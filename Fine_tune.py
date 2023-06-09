import math, re, random, torch, time, os, sys
import json
import torch.optim as optim
from random import choice
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict

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


'''Attention = Softmax(Q * K^T) * V '''


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


def lable(path):
    data = pd.read_csv(path)
    label_ = data['target']


def gelu(x):
    "Implementation of the gelu activate_ation function by Hugging Face"
    # erf(x):计算x的误差函数,
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


'''根据句子数据,构建词元的输入向量'''


# sentences_dict, mjd_dict, id_dict, id_list, vocab_size=400, batch_size=8
def make_batch(sentences_dict, mjd_dict, target, vocab_size=400, batch_size=16):
    batch = []
    label = 0
    nub = 0
    '''由于随机选取句子,所以可能依照pt文件中已训练的模型也误差较大'''
    while nub <= batch_size:

        passend_list = [0, 1, 2, 3, 4, 5]
        sentence_a_index = random.randrange(len(sentences_dict) * 3)
        sentence_a_id = int(sentence_a_index / 3)
        sentence_a_passend = 2 * (sentence_a_index - sentence_a_id * 3)

        sentence_b_index = sentence_a_index
        sentence_b_id = sentence_a_id
        sentence_b_passend = 2 * (sentence_b_index - sentence_b_id * 3) + 1

        if sentence_a_index not in a_b:  # 屏蔽已经选择过的
            a_b[sentence_a_index] = 0
            nub = nub + 1
        else:
            continue

        # sentence_ab是包含一个句子中所含单词的序号列表
        sentence_a, sentence_b = sentences_dict[str(sentence_a_id)][sentence_a_passend], \
            sentences_dict[str(sentence_b_id)][sentence_b_passend]
        mjd_a, mjd_b = mjd_dict[str(sentence_a_id)][sentence_a_passend], mjd_dict[str(sentence_b_id)][
            sentence_b_passend]

        # 加入分类符和分隔符
        input_ids = [1] + sentence_a + [2] + sentence_b + [2]
        # segment_ids用来标识a和b两个句子
        segment_ids = [0] * (1 + len(sentence_a) + 1) + [1] * (len(sentence_b) + 1)
        passend_ids = [sentence_a_passend] * (1 + len(sentence_a) + 1) + [sentence_b_passend] * (len(sentence_b) + 1)
        mjd_ids = [1] + mjd_a + [2] + mjd_b + [2]
        label = target[sentence_a_id]
        '''把所有存储单词的变量都填充至最大长度,有利于统一处理.'''
        # Zero Paddings
        n_pad = max_len - len(input_ids)  # max_len: 允许的句子最大长度
        input_ids.extend([0] * n_pad)  # word_dict['Pad'] = 0
        segment_ids.extend([0] * n_pad)
        mjd_ids.extend([0] * n_pad)
        passend_ids.extend([0] * n_pad)

        batch.append([input_ids, segment_ids, passend_ids, mjd_ids, label])  # NotNext

    return batch


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


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

        freeze(self)  # 冻结以上所有层

    #         self.fc2 = nn.Linear(d_model, 14)
    #         self.fc2.weight = nn.Parameter(torch.randn(d_model, 14))
    #         self.fc2.bias = nn.Parameter(torch.randn(14))

    def forward(self, input_ids, segment_ids, mjd_ids, passend_ids):
        # 按照论文图2构建输入encoder的嵌入矩阵, Embedding: input = word + position + segment
        output = self.embedding(input_ids, segment_ids, mjd_ids, passend_ids).to(device)
        # padding mask, 使其在softmax中为0而不影响注意力机制
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.layers:
            # output : [batch_size, max_len, d_model] [6,30,768]
            # attn : [batch_size, n_heads, d_mode, d_model]
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        #         output = softmax(self.fc2(np.squeeze(output[:,0,:])))
        return output


class fine_tuneBERT(BERT):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, d_model)
        self.fc1.weight = nn.Parameter(torch.randn(128, d_model))
        self.fc1.bias = nn.Parameter(torch.randn(128))

        self.norm2 = nn.LayerNorm(14)
        self.fc2 = nn.Linear(14, 128)
        self.fc2.weight = nn.Parameter(torch.randn(14, 128))
        self.fc2.bias = nn.Parameter(torch.randn(14))
        self.activate_1 = gelu
        self.activate_2 = nn.Tanh()



    def forward(self, in_put):
        #         # 按照论文图2构建输入encoder的嵌入矩阵, Embedding: input = word + position + segment
        #         output = self.embedding(input_ids, segment_ids, mjd_ids, passend_ids).to(device)
        #         # padding mask, 使其在softmax中为0而不影响注意力机制
        #         enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        #         for layer in self.layers:
        #             # output : [batch_size, max_len, d_model] [6,30,768]
        #             # attn : [batch_size, n_heads, d_mode, d_model]
        #             output, enc_self_attn = layer(output, enc_self_attn_mask)

        output = np.squeeze(in_put[:, 0, :])
        output = self.norm1(self.activate_1(self.fc1(output)))
        output = self.norm2(self.activate_2(self.fc2(output)))
        #         output = torch.transpose(output, dim0=0, dim1=1)

        output = softmax(output)
        return output

def confusion_matrix(preds, labels, con_matrix):
    if len(preds) == 1:
        con_matrix[preds.item(), labels.item()] += 1
        return con_matrix
    for i in range(len(preds)):
        con_matrix[preds[i].item(), labels[i].item()] += 1
    return con_matrix


if __name__ == '__main__':

    device = ['cuda:6' if torch.cuda.is_available() else 'cpu'][0]
    # BERT Parameters
    max_len = 150  # 允许的句子最大长度:9+9+1+2=21
    batch_size = 16  # batch中的句子对个数,因为positive数最多为5,所以batch_size最大为10 # 必须是偶数, 否则make_batch陷入死循环
    max_words_pred = 20  # 最大标记预测长度 21*0.15=3.15
    n_layers = 8  # number  of Encoder Layer
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

    '''2.构建模型'''
    model1 = BERT()
    model = fine_tuneBERT()

    model1.to(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    '''lr如果是0.001,将会很快陷入局部收敛!'''
    learning_rate = 0.0002

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW: 论文采用的Adam扩展版本

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*' * 30, '*' * 30))
    loss_record = []

    #     s_d = open("sentences_dict.json", "r")
    #     sentencedict = json.load(s_d)
    #     print(len(sentencedict))

    #     m_d = open("mjd_dict.json", "r")
    #     mjddict = json.load(m_d)

    s_d = open("sentences_dict_train.json", "r")
    sentencedict = json.load(s_d)
    print("length of sentence_dict is {}".format(len(sentencedict)))

    m_d = open("mjd_dict_train.json", "r")
    mjddict = json.load(m_d)

    metadata = pd.read_csv("/home/guanlanzi/data/training_set_metadata.csv")
    target_ = metadata['target']
    target = []

    labels = [90, 42, 65, 16, 15, 62, 88, 92, 67, 52, 95, 6, 64, 53]
    for i in range(len(target_)):
        target.append(labels.index(target_[i]))

    #
    # list1 = ["layers.2.enc_attn.W_Q.weight", "layers.2.enc_attn.W_Q.bias", "layers.2.enc_attn.W_K.weight",
    #         "layers.2.enc_attn.W_K.bias", "layers.2.enc_attn.W_V.weight", "layers.2.enc_attn.W_V.bias",
    #         "layers.2.enc_attn.linear.weight", "layers.2.enc_attn.linear.bias", "layers.2.enc_attn.layer_norm.weight",
    #         "layers.2.enc_attn.layer_norm.bias", "layers.2.pos_ffn.conv1.weight", "layers.2.pos_ffn.conv1.bias",
    #         "layers.2.pos_ffn.conv2.weight", "layers.2.pos_ffn.conv2.bias", "layers.2.pos_ffn.layer_norm.weight",
    #         "layers.2.pos_ffn.layer_norm.bias", "layers.3.enc_attn.W_Q.weight", "layers.3.enc_attn.W_Q.bias",
    #         "layers.3.enc_attn.W_K.weight", "layers.3.enc_attn.W_K.bias", "layers.3.enc_attn.W_V.weight",
    #         "layers.3.enc_attn.W_V.bias", "layers.3.enc_attn.linear.weight", "layers.3.enc_attn.linear.bias",
    #         "layers.3.enc_attn.layer_norm.weight", "layers.3.enc_attn.layer_norm.bias", "layers.3.pos_ffn.conv1.weight",
    #         "layers.3.pos_ffn.conv1.bias", "layers.3.pos_ffn.conv2.weight", "layers.3.pos_ffn.conv2.bias",
    #         "layers.3.pos_ffn.layer_norm.weight", "layers.3.pos_ffn.layer_norm.bias"]
    #
    # new_state_dict = OrderedDict()
    # state_dict = torch.load('model_param_MLM_IDT1.pt')
    # for k, v in state_dict.items():
    #     name = k
    #     if k in list1:
    #         continue
    #     new_state_dict[name] = v
    #
    # model1.load_state_dict(new_state_dict)

    if os.path.exists('model_param_MLM_IDT2.pt') == True:
        # 加载模型参数到模型结构
        model1.load_state_dict(torch.load('model_param_MLM_IDT2.pt', map_location=device))

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*' * 30, '*' * 30))
    loss_record = []
    iteration_list = []
    accuracy_list = []
    iteration = 100
    l = int(len(sentencedict) * 3/5)
    array = np.random.randint(1, int(len(sentencedict) * 3), size = l)

    for epoch in range(iteration):

        a_b = {}
        total = 0
        correct = 0

        for i in range(l):
            a_b[array[i]] = 0

        # iteration_list = iteration_list.to(device)
        # accuracy_list = accuracy_list.to(device)

        for i in range(1000):
            batch = make_batch(sentencedict, mjddict, target, vocab_size, batch_size)

            input_ids, segment_ids, passend_ids, mjd_ids, label = map(torch.LongTensor, zip(*batch))

            input_ids, segment_ids, passend_ids, mjd_ids, label = \
                input_ids.to(device), segment_ids.to(device), passend_ids.to(device), mjd_ids.to(device), label.to(
                    device)

            optimizer.zero_grad()
            output1 = model1(input_ids, segment_ids, mjd_ids, passend_ids)
            # print(output.shape)
            output = model(output1)

            predict = torch.argmax(output.data, 1).to(device)
            total += label.size(0)
            correct += (predict == label).sum()

            accuary = (correct / total) / 100 * 100
            # Task1: 对比完形填空准确率
            # logits_lm:[6,5,29]->[6,29,5] masked_tokens:[6,5]

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        print('in epoch _{}'.format(epoch), 'Loss = {:.6f}'.format(loss))
        accuracy_list.append(accuary)
        iteration_list.append(epoch)

    # iteration_list = iteration_list.to('cpu')
    # accuracy_list = accuracy_list.to('cpu')

    # 可视化 accuracy

    iteration_list = torch.tensor(iteration_list, device='cpu')
    accuracy_list = torch.tensor(accuracy_list , device='cpu')

    plt.plot(iteration_list, accuracy_list, color='r')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Accuracy')
    plt.title('MLM')
    plt.show()
    plt.savefig('MLM_accuracy.png')
    plt.close()

    '''4.测试'''
    a_b = {}
    print('{}\nTest\n{}'.format('*' * 30, '*' * 30))
    print("The learning rate is {}".format(learning_rate))
    print("The iteration times is {}".format(iteration))

    # print('text:\n%s' % text)
    correct = 0.0
    total = 0.0

    a_b = {}
    for i in range(len(sentencedict) * 3):
        if i not in array:
            a_b[i] = 0

    label_kinds = 14
    conf_matrix = torch.zeros(label_kinds, label_kinds)

    for j in range(200):

        batch = make_batch(sentencedict, mjddict, target, vocab_size, batch_size)

        input_ids, segment_ids, passend_ids, mjd_ids, label = map(torch.LongTensor, zip(*batch))

        input_ids, segment_ids, passend_ids, mjd_ids, label = \
            input_ids.to(device), segment_ids.to(device), passend_ids.to(device), mjd_ids.to(device), label.to(device)

        output1 = model1(input_ids, segment_ids, mjd_ids, passend_ids)
        output = model(output1)

        predict = torch.argmax(output.data, 1).to(device)
        total += label.size(0)
        correct += (predict == label).sum()

        conf_matrix = confusion_matrix(predict, label, conf_matrix)

        accuary = (correct / total) / 100 * 100
        print("accuary is {}".format(accuary))

    # 首先定义一个 分类数*分类数 的空混淆矩阵

    conf_matrix = np.array(conf_matrix)  # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

    test_num = 3200

    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
    print(conf_matrix)

    # 获取每种Emotion的识别准确率
    print("每种类别总个数：", per_kinds)
    print("每种类别预测正确的个数：", corrects)
    print("每种类别的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    # 绘制混淆矩阵
    Emotion = 14  # 这个数值是具体的分类数，大家可以自行修改
    # 每种类别的标签
    labels = [
        "90",
        "42",
        "65",
        "16",
        "15",
        "62",
        "88",
        "92",
        "67",
        "52",
        "95",
        "6",
        "64",
        "53"
    ]
    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(Emotion):
        for y in range(Emotion):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(Emotion), labels)
    plt.xticks(range(Emotion), labels, rotation=45)  # X轴字体倾斜45°
    plt.show()
    plt.savefig('confuse_matrix.png')
    plt.close()

    '''
    loss function的修改
    把双通道变成多通道
    设置学习率的衰减系数
    用cross validation
    '''

