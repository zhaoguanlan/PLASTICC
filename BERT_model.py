import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformer_encoder


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size=768, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.Flux = FluxEmbedding(vocab_size)
        self.Time = TimeEmbedding(vocab_size)
        self.Passend = PassendEmbedding()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        output = self.Flux(sequence[:, :, 0]) + self.Passend(sequence[:, :, 1]) + self.Time(sequence[:, :, 2])

        return self.dropout(output)


class TimeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=768):
        # 3个新词
        super().__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.linear_layers = nn.ModuleList([nn.Linear(1, embed_size) for _ in range(vocab_size)])

    def forward(self, sequence):
        output = [l(x).view(-1, self.vocab_size, self.embed_size) for l, x in zip(self.linear_layers, sequence)]
        return output


class PassendEmbedding(nn.Module):
    def __init__(self, max_len=512, d_model=768):
        super().__init__()
        # 这里的max_len限制的是vocab_size的最大长度
        # time embedding这是使用的就是固定的embedding，对比于segment embedding和token embedding
        # Compute the positional encodings once in log space.
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, sequence):
        # position = torch.arange(0, max_len).float().unsqueeze(1)
        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.require_grad = False

        div_term = (torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)).exp()
        passend = sequence[:, :, 1]

        pe[:, 0::2] = torch.sin(passend * div_term)
        pe[:, 1::2] = torch.cos(passend * div_term)
        # 对数据维度进行扩充，扩展第0维
        pe = pe.unsqueeze(0)
        # 添加一个持久缓冲区pe,缓冲区可以使用给定的名称作为属性访问
        self.register_buffer('pe', pe)

        return self.pe[:, :sequence.size(1)]


class FluxEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.linear_layers = nn.ModuleList([nn.Linear(1, embed_size) for _ in range(vocab_size)])

    def forward(self, sequence):
        output = [l(x).view(-1, self.vocab_size, self.embed_size) for l, x in zip(self.linear_layers, sequence)]
        return output


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: 所有字的长度
        :param hidden: BERT模型隐藏层大小
        :param n_layers: Transformer blocks(layers)数量
        :param attn_heads: 多头注意力head数量
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # 嵌入层, positional + segment + token
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # 多层transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # 多个transformer 堆叠
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class SourceDiscrimination(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        # 这里采用了logsoftmax代替了softmax，
        # 当softmax值远离真实值的时候梯度也很小，logsoftmax的梯度会更好些
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        # self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)
        # self.next_sentence(x),


if __name__ == '__main__':
    x = [[[1.0 for j in range(512)] for i in range(150)] for k in range(16)]
    x = torch.tensor(x, dtype=float)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # model = TransformerBlock(512, 8, 2048, 0.1)
    # y = model.forward(x, mask=None)
    # print(y)
    # 输入维度为16，150，512，输出维度同样为16，150，512
