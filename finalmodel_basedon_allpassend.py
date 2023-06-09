import math
import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from sklearn import model_selection

'''Transformer Encoder'''


def get_attn_pad_mask(seq_q, seq_k):
    """
    mask大小和(len_q,len_k)一致,
    是为了在点积注意力中,与torch.matmul(Q,K)的大小一致
    """
    # (seq_q, seq_k): (dec_inputs, enc_inputs)
    # dec_inputs:[batch_size, tgt_len] # [1,5]
    # enc_inputs:[batch_size, src_len] # [1,6]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
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


def Scaled_Dot_Product_Attention(Q, K, V, attn_mask, dropout=None):
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
    if dropout is not None:
        softmax = dropout(softmax)
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
        self.dropout = nn.Dropout(p=dropout)

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
        context, attn = Scaled_Dot_Product_Attention(Q_s, K_s, V_s, attn_mask, dropout=self.dropout)
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


#
#
# class Position_wise_Feed_Forward_Networks(nn.Module):
#     def __init__(self, dropout=0.1):
#         super().__init__()
#         '''输出层相当于1*1卷积层,也就是全连接层'''
#         """nn.Conv1d
#         in_channels应该理解为嵌入向量维度,out_channels才是卷积核的个数(厚度)
#         """
#         # 512 -> 2048
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         # 2048 -> 512
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model)
#
#     def forward(self, inputs):
#         # enc_outputs: [batch_size, source_len, d_model] # [1,6,512]
#         residual = inputs
#         relu = nn.ReLU()
#         # output: 512 -> 2048 [1,2048,6]
#         output = relu(self.conv1(inputs.transpose(1, 2)))
#         output = self.dropout(output)
#
#         # output: 2048 -> 512 [1,6,512]
#         output = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output + residual)

class PositionwiseFeedForward(nn.Module):

    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # mean(-1) 表示 mean(len(x)), 这里的-1就是最后一个维度，也就是最里面一层的维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        # Add and Norm
        return x + self.dropout(sublayer(self.norm(x)))


#
# class EncoderLayer(nn.Module):
#     def __init__(self):
#         super(EncoderLayer, self).__init__()
#         self.enc_attn = MultiHeadAttention()
#         self.pos_ffn = Position_wise_Feed_Forward_Networks()
#
#     def forward(self, enc_outputs, enc_attn_mask):
#         # enc_attn_mask: [1,6,6]
#         # enc_outputs to same Q,K,V
#         # enc_outputs: [batch_size, source_len, d_model] # [1, 6, 512]
#         enc_outputs, attn = self.enc_attn(enc_outputs, enc_outputs, enc_outputs, enc_attn_mask)
#         # enc_outputs: [batch_size , len_q , d_model]
#         enc_outputs = self.pos_ffn(enc_outputs)
#         return enc_outputs, attn

class EncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.feed_forward = PositionwiseFeedForward()
        self.input_sublayer = SublayerConnection()
        self.output_sublayer = SublayerConnection()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


def gelu(x):
    # erf(x):计算x的误差函数,
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


'''数据预处理'''


def pre_process(path):
    data_ = pd.read_csv(path)
    sentences_dict = {}
    mjd_dict = {}
    numb = -1
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

    for row_id in range(data_.shape[0]):
        lines = data_.iloc[row_id]
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
            mjd_dict[numb][passend - 1].append(int(mjd - 59575))
        else:
            sentences_dict[numb][passend - 1].append(flux)
            mjd_dict[numb][passend - 1].append(int(mjd - 59575))

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
    else:
        sentence_word = 200
    return sentence_word


def make_batch(sentences_dict, mjd_dict, index):
    sentence_a_id = index

    input_ids = [1] + sentences_dict[str(sentence_a_id)][0] + [2] + sentences_dict[str(sentence_a_id)][1] + [2] + sentences_dict[str(sentence_a_id)][2] + \
                [2] + sentences_dict[str(sentence_a_id)][3] + [2] + sentences_dict[str(sentence_a_id)][4] + [2] + sentences_dict[str(sentence_a_id)][5] + [2]

    passend_ids = [0] * (1 + len(sentences_dict[str(sentence_a_id)][0]) + 1) + [1] * (
            len(sentences_dict[str(sentence_a_id)][1]) + 1) + \
                  [2] * (len(sentences_dict[str(sentence_a_id)][2]) + 1) + [3] * (
                          len(sentences_dict[str(sentence_a_id)][3]) + 1) + \
                  [4] * (len(sentences_dict[str(sentence_a_id)][4]) + 1) + [5] * (
                          len(sentences_dict[str(sentence_a_id)][5]) + 1)

    mjd_ids = [1] + mjd_dict[str(sentence_a_id)][0] + [2] + mjd_dict[str(sentence_a_id)][1] + [2] + \
              mjd_dict[str(sentence_a_id)][2] + \
              [2] + mjd_dict[str(sentence_a_id)][3] + [2] + mjd_dict[str(sentence_a_id)][4] + [2] + \
              mjd_dict[str(sentence_a_id)][5] + [2]

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

    return input_ids, masked_tokens, masked_pos, passend_ids, mjd_ids


class Embedding(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        # vocab_size: len(input_ids)
        # vocab_size: 29  max_len: 30

        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.passend_embed = nn.Embedding(n_passend, d_model)
        self.mjd_embed = nn.Embedding(time_range, d_model)
        self.dropout = nn.Dropout(p=dropout)

        '''
        这里的mjd_embed中的max_len需要更改一下
        '''

        self.norm = nn.LayerNorm(d_model)  # 对每个样本的特征归一化

    def forward(self, x, mjd, passend):  # embedding(input_ids, segment_ids) # [6,30] [6,30]

        embedding = self.tok_embed(x) \
                    + self.passend_embed(passend) \
                    + self.mjd_embed(mjd)

        return self.dropout(self.norm(embedding))


'''构建模型'''


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

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

        predict_masked = torch.gather(output, 1, masked_pos)
        predict_masked = self.norm(self.activate_1(self.fc(predict_masked)))  # activate_1: gelu
        # (self.decoder:nn.Linear(768,29), self.decoder_bias: nn.Parameter(torch.zeros(29)))
        # logits_lm: [batch_size, max_words_pred, vocab_size] [6,5,29]
        '''所谓的语言模型就是vocab_size个单词的嵌入矩阵'''
        logits_lm = self.decoder(predict_masked) + self.decoder_bias

        return logits_lm


def confusion_matrix(preds, labels, con_matrix):
    if len(preds) == 1:
        con_matrix[preds.item(), labels.item()] += 1
        return con_matrix
    for i in range(len(preds)):
        con_matrix[preds[i].item(), labels[i].item()] += 1
    return con_matrix


class MyDataset(Dataset):
    # data1,data2 分别是_train_time_data和label
    def __init__(self, data1, data2, transform=None):
        super(MyDataset, self).__init__()

        # data1 = list(data1)
        # data2 = list(data2)
        # print(data1)
        # print(data2)
        #
        # self.data = np.array([*zip(data1, data2)])
        self.len = len(data1)

        data1 = list(data1)
        data2 = list(data2)
        self.data = np.array(zip(data1, data2))
        self.transform = transform

    def __getitem__(self, index):
        datas = self.data[index][0]


        if self.transform is not None:
            datas = self.transform(datas)

        return datas, labels

    def __len__(self):

        return self.len


if __name__ == '__main__':
    device = ['cuda:5' if torch.cuda.is_available() else 'cpu'][0]

    # BERT Parameters
    max_len = 450  # 允许的句子最大长度:9+9+1+2=21
    batch_size = 16  # batch中的句子对个数,因为positive数最多为5,所以batch_size最大为10 # 必须是偶数, 否则make_batch陷入死循环
    max_words_pred = 50  # 最大标记预测长度 200*0.15=30
    n_layers = 8  # number of Encoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    d_model = 512  # Embedding Size
    d_ff = 4 * d_model  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2  # ab两个句子
    n_passend = 6
    time_range = 1200
    vocab_size = 400
    dropout = 0.1
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    '''2.构建模型'''
    model = BERT()
    model.to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    '''lr如果是0.001,将会很快陷入局部收敛!'''

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # AdamW: 论文采用的Adam扩展版本
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=10, verbose=True, eps=1e-05)

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*' * 30, '*' * 30))
    loss_record = []
    #
    # s_d = open("sentences_dict.json", "r")
    # sentencedict = json.load(s_d)
    # print("len of sentencedict is {}".format(len(sentencedict)))
    #
    # m_d = open("mjd_dict.json", "r")
    # mjddict = json.load(m_d)
    #
    # ALL_data = []
    # label1 = []
    #
    # for i in range(len(sentencedict)):
    #     input_ids, masked_tokens, masked_pos, passend_ids, mjd_ids = make_batch(sentencedict, mjddict, i)
    #     label1.append((masked_tokens))
    #
    # ALL_data = np.array(ALL_data)
    # label1 = np.array(label1)
    # np.save('label1.npy',label1)
    # np.save('ALL_data.npy',ALL_data)

    ALL_data = np.load('ALL_data.npy', allow_pickle=True)
    label = np.load('label.npy', allow_pickle=True)
    print(ALL_data[0, :])

    label = np.expand_dims(label, axis=1)
    ALL_data = np.expand_dims(ALL_data, axis=1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(ALL_data, label,
                                                                        test_size=0.1,
                                                                        random_state=2)

    print('shape :', X_train.shape)
    print('shape :', y_train.shape)

    train_dataset = MyDataset(X_train, y_train, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)

    test_dataset = MyDataset(X_test, y_test, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    print("num of train", len(train_dataset))
    test_num = len(test_dataset)

    iteration = 50
    for epoch in range(iteration):
        start = time.time()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            input_ids, passend_ids, mjd_ids, mask_pos = zip(*data)
            optimizer.zero_grad()

            logits_lm = model(input_ids, passend_ids, mjd_ids, mask_pos)
            loss = criterion1(logits_lm.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()

        total = 0.0
        loss_total = 0.0
        optimizer.zero_grad()

        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            input_ids, passend_ids, mjd_ids, mask_pos = zip(*data)

            prediction = model(input_ids, passend_ids, mjd_ids, mask_pos)
            loss = criterion1(prediction.transpose(1, 2), labels)

            total = total + 1
            loss_total = loss_total + loss

        loss = loss_total / total
        scheduler.step(loss)

        end = time.time()
        loss_record.append(loss.item())
        print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))

        if epoch + 1 == 50:
            torch.save(model.state_dict(), 'model_param_allpassend4.pt')
