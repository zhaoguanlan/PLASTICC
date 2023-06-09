import math, re, random, torch, time, os, sys
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

import tempfile
import argparse
from random import choice
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from distributed_utils import init_distributed_mode, dist, cleanup
from train_eval_utils import train_one_epoch, evaluate
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet

'''Transformer Encoder'''
def get_attn_pad_mask(seq_q, seq_k):
    """
    mask大小和(len_q,len_k)一致,
    是为了在点积注意力中,与torch.matmul(Q,K)的大小一致
    """
    # (seq_q, seq_k): (dec_inputs, enc_inputs)
    # dec_inputs:[batch_size, tgt_len] # [1,5]
    # enc_inputs:[batch_size, src_len] # [1,6]
    batch_size, len_q = seq_q.size()  # 1,5
    batch_size, len_k = seq_k.size()  # 1,6
    """Tensor.data.eq(element)
    eq即equal,对Tensor中所有元素进行判断,和element相等即为True,否则为False,返回二值矩阵
    Examples:
        >>> tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> tensor.data.eq(1) 
        tensor([[ True, False, False],
                [False, False, False]])
    """
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 升维 enc: [1,6] -> [1,1,6]
    # 矩阵扩充: enc: pad_attn_mask: [1,1,6] -> [1,5,6]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size, len_q, len_k

def Scaled_Dot_Product_Attention(Q, K, V, attn_mask):
    # Q_s: [batch_size, n_heads, len_q, d_k] # [1,8,5,64]
    # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,6,64]
    # attn_mask: [batch_size, n_heads, len_q, len_k] # [1,8,5,6]

    """torch.matmul(Q, K)
        torch.matmul是tensor的乘法,输入可以是高维的.
        当输入是都是二维时,就是普通的矩阵乘法.
        当输入有多维时,把多出的一维作为batch提出来,其他部分做矩阵乘法.
        Exeamples:
        a = torch.ones(3,4)
        b = torch.ones(4,2)
        torch.matmul(a,b).shape
            torch.Size([3,2])

        a = torch.ones(5,3,4)
        b = torch.ones(4,2)
        torch.matmul(a,b).shape
            torch.Size([5,3,2])

        a = torch.ones(2,5,3)
        b = torch.ones(1,3,4)
        torch.matmul(a,b).shape
            torch.Size([2,5,4])
        """
    # [1,8,5,64] * [1,8,64,6] -> [1,8,5,6]
    # scores : [batch_size, n_heads, len_q, len_k]
    scores = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(d_k)  # divided by scale
    '''Attention = Softmax(Q * K^T) * V '''
    """scores.masked_fill_(attn_mask, -1e9) 
    由于scores和attn_mask维度相同,根据attn_mask中的元素值,把和attn_mask中值为True的元素的
    位置相同的scores元素的值赋为-1e9
    """
    scores.masked_fill_(attn_mask, -1e9)

    # 'P'的scores元素值为-1e9, softmax值即为0
    softmax = nn.Softmax(dim=-1)  # 求行的softmax
    attn = softmax(scores)  # [1,8,6,6]
    # [1,8,6,6] * [1,8,6,64] -> [1,8,6,64]
    context = torch.matmul(attn, V)  # [1,8,6,64]
    return context, attn

class MultiHeadAttention(nn.Module):
    # dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # (512, 64*8) # d_q必等于d_k
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # (512, 64*8) # 保持维度不变
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # (512, 64*8)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # dec_outputs: [batch_size, tgt_len, d_model] # [1,5,512]
        # enc_outputs: [batch_size, src_len, d_model] # [1,6,512]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len] # [1,5,6]
        # q/k/v: [batch_size, len_q/k/v, d_model]
        residual, batch_size = Q, len(Q)
        '''
        用n_heads=8把512拆成64*8,在不改变计算成本的前提下,让各注意力头相互独立,更有利于学习到不同的特征
        '''
        # Q_s: [batch_size, len_q, n_heads, d_q] # [1,5,8,64]
        # new_Q_s: [batch_size, n_heads, len_q, d_q] # [1,8,5,64]
        Q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,6,64]
        K_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V_s: [batch_size, n_heads, len_k, d_v] # [1,8,6,64]
        V_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [1,5,6] -> [1,1,5,6] -> [1,8,5,6]
        # attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        # context: [1,8,5,64] attn: [1,8,5,6]
        context, attn = Scaled_Dot_Product_Attention(Q_s, K_s, V_s, attn_mask)
        """
        contiguous() 连续的
        contiguous: view只能用在连续(contiguous)的变量上.
        如果在view之前用了transpose, permute等,
        需要用contiguous()来返回一个contiguous copy
        """
        # context: [1,8,5,64] -> [1,5,512]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # context: [1,5,512] -> [1,5,512]
        output = self.linear(context)
        """nn.LayerNorm(output) 样本归一化
        和对所有样本的某一特征进行归一化的BatchNorm不同,
        LayerNorm是对每个样本进行归一化,而不是一个特征

        Tips:
            归一化Normalization和Standardization标准化区别:
            Normalization(X[i]) = (X[i] - np.min(X)) / (np.max(X) - np.min(X))
            Standardization(X[i]) = (X[i] - np.mean(X)) / np.var(X)
        """
        output = self.layer_norm(output + residual)
        return output, attn

class Position_wise_Feed_Forward_Networks(nn.Module):
    def __init__(self):
        super().__init__()
        '''输出层相当于1*1卷积层,也就是全连接层'''
        """nn.Conv1d
        in_channels应该理解为嵌入向量维度,out_channels才是卷积核的个数(厚度)
        """
        # 512 -> 2048
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 2048 -> 512
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # enc_outputs: [batch_size, source_len, d_model] # [1,6,512]
        residual = inputs
        relu = nn.ReLU()
        # output: 512 -> 2048 [1,2048,6]
        output = relu(self.conv1(inputs.transpose(1, 2)))
        # output: 2048 -> 512 [1,6,512]
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention()
        self.pos_ffn = Position_wise_Feed_Forward_Networks()

    def forward(self, enc_outputs, enc_attn_mask):
        # enc_attn_mask: [1,6,6]
        # enc_outputs to same Q,K,V
        # enc_outputs: [batch_size, source_len, d_model] # [1, 6, 512]
        enc_outputs, attn = self.enc_attn(enc_outputs, \
                                          enc_outputs, enc_outputs, enc_attn_mask)
        # enc_outputs: [batch_size , len_q , d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

def gelu(x):
    "Implementation of the gelu activate_ation function by Hugging Face"
    # erf(x):计算x的误差函数,
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

'''数据预处理'''

def pre_process(path):
    data = pd.read_csv(path)
    sentences_dict = {}
    mjd_dict = {}
    numb = -1
    id_list = []
    id_dict = {}
    """
        id_list:[3,4,5,6,7,8,...]
        id_dict:{'3':1,'4':2,...}

        sentences_list = {14:[[flux-passend1],
                              [flux-passend2],
                              [flux-passend3],
                              [flux-passend4],
                              [flux-passend5],
                              [flux-passend6]]}
    """

    for row_id in range(data.shape[0]):
        lines = data.iloc[row_id]
        id = lines[0]
        mjd = lines[1]
        passend = int(lines[2])
        flux = transfer(lines[3])

        '''
        注意我这里假设了数据中，同一个id一定是一起出现的情况，所以没有使用
        if id not in id_list
        为了减少这一步查找的计算量(还需要验证一下)
        '''

        if id not in id_list:

            id_list.append(id)
            numb = numb + 1
            id_dict[id] = numb
            sentences_dict[numb] = [[], [], [], [], [], []]
            sentences_dict[numb][passend - 1].append(flux)
            mjd_dict[numb] = [[], [], [], [], [], []]
            mjd_dict[numb][passend - 1].append(int(mjd-59575))
        else:
            sentences_dict[numb][passend - 1].append(flux)
            mjd_dict[numb][passend - 1].append(int(mjd-59575))

    for i, w in enumerate(id_list):
        id_dict[w] = i

    return sentences_dict, mjd_dict, id_dict, id_list

def transfer(sentence_word):
    if 0 < sentence_word < 1:
        sentence_word = 1 + 200
    elif -1 < sentence_word < 0:
        sentence_word = -1 + 200
    elif sentence_word > 0:
        sentence_word = min(399, int(math.log(sentence_word, 1.1)) + 2 + 200)
    elif sentence_word < 0:
        sentence_word = max(3, -int(math.log(-sentence_word, 1.1)) - 2 + 200)
    else :
        sentence_word = 200
    return sentence_word

def make_batch(sentences_dict, mjd_dict, vocab_size=400, batch_size=8):
    batch = []
    batch_number = 0
    #     positive = negative = 0
    a_b = []
    ''' sentences_dict, mjd_dict, id_dict, id_list, vocab_size=400, batch_size=8 '''
    '''由于随机选取句子,所以可能依照pt文件中已训练的模型也误差较大'''
    while batch_number < batch_size:
        """random.randrange(start, stop=None, step=1) 
        在(start, stop)范围内随即返回一个数
        Args:
            start: 指定范围内的开始值,包含在范围内
            stop: 指定范围内的结束值,不包含在范围内,若无参数stop则默认范围为0~start
            step: 指定递增基数,默认为1
        Examples:
            >>> random.randrange(1, 100, 2) 
            67 # 以2递增,只能返回奇数
        """
        # 在样本集中随机选取a和b两个句子
        passend_list = [0, 1, 2, 3, 4, 5]

        sentence_a_index = random.randrange(len(sentences_dict) * 6)
        sentence_a_id = int(sentence_a_index / 6)
        sentence_a_passend = sentence_a_index - sentence_a_id * 6

        #         if random.random() < 0.5:
        #             sentence_b_id = sentence_a_id
        #             passend_list.remove(sentence_a_passend)
        #             sentence_b_passend = choice(passend_list)
        #             sentence_b_index = 6 * sentence_b_id + sentence_b_passend
        #         else:
        #             sentence_b_index = random.randrange(len(sentences_dict) * 6)
        #             sentence_b_id = int(sentence_b_index / 6)
        #             sentence_b_passend = sentence_b_index - sentence_b_id * 6

        # sentence_a，b是包含一个句子中所含单词的序号列表

        if sentence_a_index not in a_b:  # 屏蔽已经选择过的
            a_b.append(sentence_a_index)
        elif len(a_b) < batch_size:  # 达到要求的样本值
            continue
        else:
            break

        sentence_a = sentences_dict[str(sentence_a_id)][sentence_a_passend]

        mjd_a = mjd_dict[str(sentence_a_id)][sentence_a_passend]

        # 加入分类符和分隔符
        input_ids = [1] + sentence_a
        # segment_ids用来标识a和b两个句子
        passend_ids = [sentence_a_passend] * (1 + len(sentence_a))
        mjd_ids = [1] + mjd_a

        # MASK LM
        # 把句子15%单词准备用于mask
        # max_words_pred(=5): 最大预测单词数  n_pred: 预测个数
        n_pred = int(0.15 * len(input_ids))
        n_pred = min(max_words_pred, max(1, n_pred))  # 预测个数不能为0,也不能大于最大预测单词数
        # 构建候选词列表(随机打乱后只有前n_pred个会被选择)
        # 要屏蔽cls和seq,所以不能用torch.arange(len(input_ids))
        candidate_mask_tokens = [i for i, token in enumerate(input_ids) if token != 1
                                 and token != 2]  # [1,2,3......](不包含0([CLS])和[SEP])

        # 随机打乱candidate_mask_tokens
        random.shuffle(candidate_mask_tokens)

        # random.shuffle打乱候选表后,直接选择前n_pred个
        masked_tokens, masked_pos = [], []
        for pos in candidate_mask_tokens[:n_pred]:
            # 不管是被mask还是被随机还是不变,都会加入masked列表(masked中的不会被替换,替换的是input_ids)
            # 记录原始单词,用于和模型输出比对计算loss
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random.random() < 0.8:  # 80%  -> MASK
                input_ids[pos] = 0  # mask
            elif random.random() < 0.5:  # 10% -> random
                random_index = random.randint(0, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = random_index  # replace

        '''把所有存储单词的变量都填充至最大长度,有利于统一处理.'''

        # Zero Paddings
        n_pad = max_len - len(input_ids)  # max_len: 允许的句子最大长度
        input_ids.extend([0] * n_pad)  # word_dict['Pad'] = 0
        mjd_ids.extend([0] * n_pad)
        passend_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_words_pred > n_pred:  # max_words_pred(=5): 最大预测长度
            n_pad = max_words_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 判断句间关系
        batch.append([input_ids, masked_tokens, masked_pos, passend_ids, mjd_ids])
        batch_number = batch_number + 1

    # batch: [6,5]
    return batch

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        # vocab_size: len(input_ids)
        # vocab_size: 29  max_len: 30

        '''
        我觉得这里其实不能用embedding函数了，要直接用linear projection
        '''

        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.passend_embed = nn.Embedding(n_passend, d_model)
        self.mjd_embed = nn.Embedding(time_range, d_model)

        '''
        这里的mjd_embed中的max_len需要更改一下
        '''

        self.norm = nn.LayerNorm(d_model)  # 对每个样本的特征归一化

    def forward(self, x, mjd, passend):  # embedding(input_ids, segment_ids) # [6,30] [6,30]
        #         seq_len = x.size(1)  # 30
        # torch.arange(n): 生成0~n-1的一维张量,默认类型为int
        # 加入序列信息,不用transformer的positional encoding
        #         pos = torch.arange(seq_len, dtype=torch.long).to(device)  # [0,1,2,3...seq-1]
        #         '''
        #         expand_as/expand拓展后,每一个(1,seq_len)都是同步变化的,不能单独修改某一个
        #         单独修改某一个只能用repeat(重复倍数)
        #         '''
        #         pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len) [6,30]

        '''
        虽然tok_embed是(29,768),但embedding首参数是指单词类别,x虽然长度=30>29,
        但是单词总类别依然肯定不大于29,所以tok_embed可以作为x的嵌入矩阵
        seg_embed同理
        '''
        # [6,30,768], [6,30,768], [6,30,768], [6,30,768]
        '''论文图2, Embedding: input = word + position + segment'''
        # token_input = self.tok_linear_projection(x.unsqueeze(1))
        embedding = self.tok_embed(x) \
                    + self.passend_embed(passend) \
                    + self.mjd_embed(mjd)

        return self.norm(embedding)

'''构建模型'''

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        self.activate_1 = gelu
        self.activate_2 = nn.Tanh()

        # decoder is shared with embedding layer
        token_embed_weight = self.embedding.tok_embed.weight
        vocab_size, n_dim = token_embed_weight.size()  # 29, 768
        self.decoder = nn.Linear(n_dim, vocab_size, bias=False)
        self.decoder.weight = token_embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, mjd_ids, passend_ids, masked_pos):
        # 按照论文图2构建输入encoder的嵌入矩阵, Embedding: input = word + position + segment
        output = self.embedding(input_ids, mjd_ids, passend_ids).to(device)
        # padding mask, 使其在softmax中为0而不影响注意力机制
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.layers:
            # output : [batch_size, max_len, d_model] [6,30,768]
            # attn : [batch_size, n_heads, d_mode, d_model]
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        '''Task1'''
        # masked_pos ->  [batch_size, max_words_pred, d_model] [6,5,768]
        # [:, :, None]在第三个维度增加一维 masked_pos: [6,5] -> [6,5,768]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # expand(-1)表示不变
        # 从最终输出output中提取mask信息
        """torch.gather() 收集输入的特定维度指定位置的数值
        Args:
            input(tensor): 待操作数.不妨设其维度为(x1, x2, ···, xn)
            dim(int): 待操作的维度
            index(LongTensor): 如何对input进行操作
        """
        # predict_masked: [batch_size, max_words_pred, d_model] # [6,5,768]
        # output:[6,30,768] -> predict_masked:[6,5,768]
        predict_masked = torch.gather(output, 1, masked_pos)
        predict_masked = self.norm(self.activate_1(self.fc(predict_masked)))  # activate_1: gelu
        # (self.decoder:nn.Linear(768,29), self.decoder_bias: nn.Parameter(torch.zeros(29)))
        # logits_lm: [batch_size, max_words_pred, vocab_size] [6,5,29]
        '''所谓的语言模型就是vocab_size个单词的嵌入矩阵'''
        logits_lm = self.decoder(predict_masked) + self.decoder_bias

        return logits_lm

'''使用multi gpu训练'''
def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,)

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    # 实例化模型
    model = BERT().to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()

device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]
# BERT Parameters

max_len = 100 # 允许的句子最大长度:9+9+1+2=21
batch_size = 8  # batch中的句子对个数,因为positive数最多为5,所以batch_size最大为10 # 必须是偶数, 否则make_batch陷入死循环
max_words_pred = 8  # 最大标记预测长度 200*0.15=30
n_layers = 12  # number of Encoder of Encoder Layer
n_heads = 12  # number of heads in Multi-Head Attention
d_model = 768  # Embedding Size
d_ff = 4 * d_model  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # ab两个句子
n_passend = 6
time_range = 1200
vocab_size = 400
#passend_list = [0, 1, 2, 3, 4, 5]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="/home/wz/data_set/flower_data/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
