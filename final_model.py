"""
Task: 基于BERT的完形填空和句间关系判断
Author: ChengJunkai @github.com/Cheng0829
Email: chengjunkai829@gmail.com
Date: 2022/09/21
Reference: Tae Hwan Jung(Jeff Jung) @graykode
"""

import math, re, random, torch, time, os, sys
import json
import torch.optim as optim
from random import choice
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


def get_attn_pad_mask(seq_q, seq_k):
    """
    mask大小和(len_q,len_k)一致,
    是为了在点积注意力中,与torch.matmul(Q,K)的大小一致
    """
    # (seq_q, seq_k): (dec_inputs, enc_inputs)
    # dec_inputs:[batch_size, tgt_len] # [1,150]
    # enc_inputs:[batch_size, src_len] # [1,150]
    batch_size, len_q = seq_q.size()  # 1,150
    batch_size, len_k = seq_k.size()  # 1,150
    """Tensor.data.eq(element)
    eq即equal,对Tensor中所有元素进行判断,和element相等即为True,否则为False,返回二值矩阵
    Examples:
        >>> tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
        >>> tensor.data.eq(1) 
        tensor([[ True, False, False],
                [False, False, False]])
    """
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # 升维 enc: [1,150] -> [1,1,150]
    # 矩阵扩充: enc: pad_attn_mask: [1,1,6] -> [1,5,6]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size, len_q, len_k

    """
    a通过expand（）函数扩展某一维度后自身不会发生变化
    """
    # Attention = Softmax(Q * K^T) * V



def Scaled_Dot_Product_Attention(Q, K, V, attn_mask):
    # Q_s: [batch_size, n_heads, len_q, d_k] # [1,8,150,64]
    # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,150,64]
    # attn_mask: [batch_size, n_heads, len_q, len_k] # [1,8,150,150]

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
    # [1,8,150,64] * [1,8,64,150] -> [1,8,150,150]
    # scores : [batch_size, n_heads, len_q, len_k]
    scores = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(d_k)  # divided by scale

    '''np.sqrt(B):求B的开方（算数平方根）'''

    """scores.masked_fill_(attn_mask, -1e9) 
    由于scores和attn_mask维度相同,根据attn_mask中的元素值,把和attn_mask中值为True的元素的
    位置相同的scores元素的值赋为-1e9
    """
    scores.masked_fill_(attn_mask, -1e9)

    # 'P'的scores元素值为-1e9, softmax值即为0
    softmax = nn.Softmax(dim=-1)  # 求行的softmax
    attn = softmax(scores)  # [1,8,150,150]
    # [1,8,150,150] * [1,8,150,64] -> [1,8,150,64]
    context = torch.matmul(attn, V)  # [1,8,150,64]
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
        # dec_outputs: [batch_size, tgt_len, d_model] # [1,150,512]
        # enc_outputs: [batch_size, src_len, d_model] # [1,150,512]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len] # [1,150,150]
        # q/k/v: [batch_size, len_q/k/v, d_model]
        residual, batch_size = Q, len(Q)
        '''
        用n_heads=8把512拆成64*8,在不改变计算成本的前提下,让各注意力头相互独立,更有利于学习到不同的特征
        '''
        # Q_s: [batch_size, len_q, n_heads, d_q] # [1,150,8,64]
        # new_Q_s: [batch_size, n_heads, len_q, d_q] # [1,8,150,64]
        Q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K_s: [batch_size, n_heads, len_k, d_k] # [1,8,150,64]
        K_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V_s: [batch_size, n_heads, len_k, d_v] # [1,8,150,64]
        V_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [1,150,150] -> [1,1,150,150] -> [1,8,150,150]
        # attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        '''
        squeeze(1)和squeeze(-1)作用：两者的效果一样，都是给张量tensor降维，如果在第2个维度为1，那么则消去这个维度
        unsqueeze(1)则相反,比如原来张量的维度为(2,3)的话那么使用该函数之后,则变成了(2,1,3)维
        
        repeat函数:在指定维度重复几次
        
        '''

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        # context: [1,8,150,64] attn: [1,8,150,150]
        context, attn = Scaled_Dot_Product_Attention(Q_s, K_s, V_s, attn_mask)

        """contiguous() 连续的
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
        # enc_outputs: [batch_size, source_len, d_model] # [1,150,512]
        residual = inputs
        relu = nn.ReLU()
        # output: 512 -> 2048 [1,2048,150]
        output = relu(self.conv1(inputs.transpose(1, 2)))
        # output: 2048 -> 512 [1,150,512]
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention()
        self.pos_ffn = Position_wise_Feed_Forward_Networks()

    def forward(self, enc_outputs, enc_attn_mask):
        # enc_attn_mask: [1, 150, 150]
        # enc_outputs to same Q,K,V
        # enc_outputs: [batch_size, source_len, d_model] # [1, 6, 512]
        enc_outputs, attn = self.enc_attn(enc_outputs, enc_outputs, enc_outputs, enc_attn_mask)
        # enc_outputs: [batch_size , len_q , d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


"""*********以上为Transformer架构代码*******************"""

'''数据预处理'''

def pre_process(path):
    data = pd.read_csv(path)
    sentences_dict = {}
    mjd_dict = {}
    numb = -1
    # id_list = []
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

        if id not in id_dict:

            numb = numb + 1
            id_dict[id] = numb
            sentences_dict[numb] = [[], [], [], [], [], []]
            sentences_dict[numb][passend - 1].append(flux)
            mjd_dict[numb] = [[], [], [], [], [], []]
            mjd_dict[numb][passend - 1].append(int(mjd-59575))
        else:
            sentences_dict[numb][passend - 1].append(flux)
            mjd_dict[numb][passend - 1].append(int(mjd-59575))

    return sentences_dict, mjd_dict, id_dict

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

def gelu(x):
    "Implementation of the gelu activate_ation function by Hugging Face"
    # erf(x):计算x的误差函数,
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

'''根据句子数据,构建词元的输入向量'''

#sentences_dict, mjd_dict, id_dict, id_list, vocab_size=400, batch_size=8
def make_batch(sentences_dict, mjd_dict, vocab_size=400, batch_size=16):
    batch = []
    positive = negative = 0

    '''由于随机选取句子,所以可能依照pt文件中已训练的模型也误差较大'''
    while (positive + negative) < batch_size:
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
        passend_list = [0, 1, 2, 3, 4, 5]
        sentence_a_index = random.randrange(len(sentences_dict) * 6)
        sentence_a_id = int(sentence_a_index / 6)
        sentence_a_passend = sentence_a_index - sentence_a_id * 6
        # 在样本集中随机选取a和b两个句子
        if random.random() < 0.75:  # 80%  -> MASK
            sentence_b_index = random.randrange(len(sentences_dict) * 6)
            sentence_b_id = int(sentence_b_index / 6)
            sentence_b_passend = sentence_b_index - sentence_b_id * 6
            Next = False
            negative = negative + 1
        else:
            sentence_b_id = sentence_a_id
            passend_list.remove(sentence_a_passend)
            sentence_b_passend = choice(passend_list)
            sentence_b_index = 6 * sentence_b_id + sentence_b_passend
            Next = True
            positive = positive + 1

        if sentence_a_index not in a_b:  # 屏蔽已经选择过的
            a_b[sentence_a_index] = 0
        else:
            continue

        if sentence_b_index not in a_b:  # 屏蔽已经选择过的
            a_b[sentence_b_index] = 0
        else:
            continue

        # sentence_ab是包含一个句子中所含单词的序号列表

        sentence_a, sentence_b = sentences_dict[str(sentence_a_id)][sentence_a_passend], \
                                 sentences_dict[str(sentence_b_id)][sentence_b_passend]
        mjd_a, mjd_b = mjd_dict[str(sentence_a_id)][sentence_a_passend], mjd_dict[str(sentence_b_id)][sentence_b_passend]

        # 加入分类符和分隔符
        input_ids = [1] + sentence_a + [2] + sentence_b + [2]
        # segment_ids用来标识a和b两个句子
        segment_ids = [0] * (1 + len(sentence_a) + 1) + [1] * (len(sentence_b) + 1)
        passend_ids = [sentence_a_passend] * (1 + len(sentence_a) + 1) + [sentence_b_passend] * (len(sentence_b) + 1)
        mjd_ids = [1] + mjd_a + [2] + mjd_b + [2]

        # MASK LM
        # 把句子15%单词准备用于mask
        # max_words_pred(=5): 最大预测单词数  n_pred: 预测个数
        n_pred = int(0.15 * len(input_ids))
        n_pred = min(max_words_pred, max(1, n_pred))  # 预测个数不能为0,也不能大于最大预测单词数
        # 构建候选词列表(随机打乱后只有前n_pred个会被选择)
        # 要屏蔽cls和seq,所以不能用torch.arange(len(input_ids))
        candidate_mask_tokens = [i for i, token in enumerate(input_ids) if token != -1
                                 and token != 1]  # [1,2,3......](不包含0([CLS])和[SEP])

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
        segment_ids.extend([0] * n_pad)
        mjd_ids.extend([0] * n_pad)
        passend_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_words_pred > n_pred:  # max_words_pred(=5): 最大预测长度
            n_pad = max_words_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # # 判断句间关系
        # if sentence_a_index == sentence_b_index and positive < batch_size / 2:  # 论文要求严格保持50% 50%
        #     batch.append([input_ids, segment_ids, masked_tokens, masked_pos, passend_ids, mjd_ids, True])  # IsNext
        #     positive += 1
        # elif sentence_a_index != sentence_b_index and negative < batch_size / 2:  # 论文要求严格保持50% 50%
        batch.append([input_ids, segment_ids, masked_tokens, masked_pos, passend_ids, mjd_ids, Next])  # NotNext

    # batch: [6,5]
    return batch

'''论文图2所设计的输入encoder的嵌入向量'''

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        '''
        我觉得这里其实不能用embedding函数了，要直接用linear projection
        '''

        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        # self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.passend_embed = nn.Embedding(n_passend, d_model)
        self.mjd_embed = nn.Embedding(time_range, d_model)

        '''
        这里的mjd_embed中的max_len需要更改一下
        '''
        self.norm = nn.LayerNorm(d_model)  # 对每个样本的特征归一化

    def forward(self, x, seg, mjd, passend):  # embedding(input_ids, segment_ids) # [6,30] [6,30]
        # seq_len = x.size(1)  # 30
        # torch.arange(n): 生成0~n-1的一维张量,默认类型为int
        # 加入序列信息,不用transformer的positional encoding
        # pos = torch.arange(seq_len, dtype=torch.long).to(device)  # [0,1,2,3...seq-1]
        '''
        expand_as/expand拓展后,每一个(1,seq_len)都是同步变化的,不能单独修改某一个
        单独修改某一个只能用repeat(重复倍数)
        '''
        # pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len) [6,30]
        '''
        虽然tok_embed是(29,768),但embedding首参数是指单词类别,x虽然长度=30>29,
        但是单词总类别依然肯定不大于29,所以tok_embed可以作为x的嵌入矩阵
        seg_embed同理
        '''
        # [6,30,768], [6,30,768], [6,30,768], [6,30,768]
        '''论文图2, Embedding: input = word + position + segment'''
        # token_input = self.tok_linear_projection(x.unsqueeze(1))
        embedding = self.tok_embed(x) \
                    + self.seg_embed(seg) \
                    + self.passend_embed(passend) \
                    + self.mjd_embed(mjd)

        return self.norm(embedding)

'''2.构建模型'''

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

    def forward(self, input_ids, segment_ids, mjd_ids, passend_ids, masked_pos):
        # 按照论文图2构建输入encoder的嵌入矩阵, Embedding: input = word + position + segment
        output = self.embedding(input_ids, segment_ids, mjd_ids, passend_ids).to(device)
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

        '''Task2'''
        '''只训练第一列? CLS能训练?   input是cls,但输出不是'''
        # 从第一个CLS值得到结果
        # output[:, 0]取output第一列的数据 [6,30,768] -> [6,768]
        # h_pooled: [batch_size, d_model] [6,768]
        h_pooled = self.activate_2(self.fc(output[:, 0]))  # activate_2: Tanh
        # 全连接分类 (classifier:768->2)
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]
        return logits_lm, logits_clsf


if __name__ == '__main__':

    device = ['cuda:6' if torch.cuda.is_available() else 'cpu'][0]
    # BERT Parameters
    max_len = 150 # 允许的句子最大长度:9+9+1+2=21
    batch_size = 16  # batch中的句子对个数,因为positive数最多为5,所以batch_size最大为10 # 必须是偶数, 否则make_batch陷入死循环
    max_words_pred = 20  # 最大标记预测长度 21*0.15=3.15
    n_layers = 12  # number of Encoder of Encoder Layer
    n_heads = 12  # number of heads in Multi-Head Attention
    d_model = 512  # Embedding Size
    d_ff = 4 * d_model  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2  # ab两个句子
    n_passend = 6
    time_range = 1200

    vocab_size = 400

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print("开始运行程序,数据预处理中:...")
    # path = "/home/guanlanzi/new_data/test_set0_5000000.csv"

    # '''1.数据预处理'''
    # start = time.time()
    # sentences_dict, mjd_dict, id_dict, id_list = pre_process(path)
    # print("data_processing finished , len of id_dict is {}".format(len(id_list)))
    # end = time.time()
    # print("data_processing的运行时间为：{}".format(end-start))

    '''2.构建模型'''
    model = BERT()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    '''lr如果是0.001,将会很快陷入局部收敛!'''

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # AdamW: 论文采用的Adam扩展版本

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*' * 30, '*' * 30))
    loss_record = []

    s_d = open("sentences_dict.json", "r")
    sentencedict = json.load(s_d)
    print("length of sentence_dict is {}".format(len(sentencedict)))

    m_d = open("mjd_dict.json", "r")
    mjddict = json.load(m_d)

    # if os.path.exists('model_param.pt') == True:
    #     # 加载模型参数到模型结构
    #     model.load_state_dict(torch.load('model_param.pt', map_location=device))

    """map(function, iterable, ...)
    map()会根据提供的函数对指定序列做映射。第一个参数function以参数序列中的每一个元素调用function函数,
    返回包含每次function函数返回值的新列表。
    Args:
        function: 函数
        iterable: 一个或多个序列
    """
    """zip([iterable, ...]) 
    用于将可迭代的对象作为参数,将对象中对应的元素打包成一个个元组,然后返回由这些元组组成的列表.
    Examples
        >>> a = [1,2,3]
        >>> b = [4,5,6]
        >>> list(zip([a,b]))
        [(1, 4), (2, 5), (3, 6)]

        >>> c = [a,b] # [[1,2,3],[4,5,6]]
        >>> list(zip(*c))
        [(1, 4), (2, 5), (3, 6)]
        >>> list(zip(c))  
        [([1, 2, 3],), ([4, 5, 6],)]
    """

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*' * 30, '*' * 30))
    loss_record = []
    for epoch in range(5):
        start = time.time()
        a_b = {}

        for i in range(200000):

            batch = make_batch(sentencedict, mjddict, vocab_size, batch_size)

            input_ids, segment_ids, masked_tokens, masked_pos, passend_ids, mjd_ids, isNext = map(torch.LongTensor,
                                                                                                  zip(*batch))

            input_ids, segment_ids, masked_tokens, masked_pos, passend_ids, mjd_ids, isNext = \
                input_ids.to(device), segment_ids.to(device), masked_tokens.to(device), masked_pos.to(device), \
                passend_ids.to(device), mjd_ids.to(device), isNext.to(device)

            optimizer.zero_grad()
            logits_lm, logits_clsf = model(input_ids, segment_ids, mjd_ids, passend_ids, masked_pos)
            # Task1: 对比完形填空准确率
            # logits_lm:[6,5,29]->[6,29,5] masked_tokens:[6,5]

            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
            # Task2: 对比句间关系分类准确率
            # logits_clsf:[6,2] isNext:[6,]
            loss_clsf = criterion(logits_clsf, isNext)
            loss = loss_lm + loss_clsf
            loss.backward()
            optimizer.step()

            if i % 50000 == 0:
                print('in batch _{}'.format(i) ,'Loss = {:.6f}'.format(loss))
                loss_record.append(loss.item())

        end = time.time()
        print("一个epoch的运行时间为：{}".format(end-start))

        if (epoch + 1) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 'model_param_MLM_IDT3.pt')
