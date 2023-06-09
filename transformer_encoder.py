import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

'''
readme:

x 的维度是:(batch_size,vab_size,d_model)
batch_size = 16
vab_size还不确定
d_model = 512

'''

class Attention(nn.Module):
    """
    Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax得到概率得分p_atten,
        p_attn = F.softmax(scores, dim=-1)
        # 如果有 dropout 就随机 dropout 比例参数
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
        # 在计算的时候把缺失的值，进行填-1e9，以保证softmax之后还是为 0


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        # h 表示模型个数
        super().__init__()
        assert d_model % h == 0

        # d_k 表示 key长度，d_model表示模型输出维度，需保证为h得正数倍
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model =&gt; h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # 对一个Tensor执行contiguous()，如果该Tensor语义相邻、内存也相邻，则contiguous()不会有任何操作；
        # 但如果该Tensor语义相邻，内存上不相邻，则会重新开辟一块内存空间存放此Tensor的数据，使得语义和内存上都相邻。
        # 为什么需要Tensor在内存上相邻呢？因为torch.view是需要Tensor在内存上是相邻的。
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """
    在 google github中的BERT的代码实现中用Gaussian Error Linear Unit代替了RelU作为激活函数

    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

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

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # Add and Norm
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # 多头注意力模型
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        # PFFN
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        # 输入层
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # 输出层
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


if __name__ == '__main__':
    x = [[[1.0 for j in range(512)] for i in range(150)] for k in range(16)]
    x = torch.tensor(x, dtype=float)
    torch.set_default_tensor_type(torch.DoubleTensor)
    model = TransformerBlock(512, 8, 2048, 0.1)
    y = model.forward(x, mask=None)
    print(y)
    #输入维度为16，150，512，输出维度同样为16，150，512
