import math, re, random, torch, time, os, sys
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
'''1.数据预处理'''

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

        if numb == 10000:
            print("the id number in id_dict has beyond 10000")

    return sentences_dict, mjd_dict, id_dict

if __name__ == "__main__":

    device = ['cuda:6' if torch.cuda.is_available() else 'cpu'][0]
    # BERT Parameters

    max_len = 200 # 允许的句子最大长度:9+9+1+2=21
    batch_size = 8  # batch中的句子对个数,因为positive数最多为5,所以batch_size最大为10 # 必须是偶数, 否则make_batch陷入死循环
    max_words_pred = 25  # 最大标记预测长度 200*0.15=30
    n_layers = 4  # number of Encoder Layer
    n_heads = 12  # number of heads in Multi-Head Attention
    d_model = 768  # Embedding Size
    d_ff = 4 * d_model  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_passend = 6
    time_range = 1200
    vocab_size = 400

    print("开始运行程序,数据预处理中:...")
    path = '/home/guanlanzi/data/test_set.csv'

    #1.数据预处理
    start = time.time()
    sentences_dict, mjd_dict, id_dict = pre_process(path)
    s_d = open("sentences_dict.json", "w")
    json.dump(sentences_dict,s_d)
    s_d.close()

    m_d = open("mjd_dict.json", "w")
    json.dump(mjd_dict,m_d)
    m_d.close()

    i_d = open("id_dict.json", "w")
    json.dump(id_dict,i_d)
    i_d.close()

    print("data_processing finished , len of id_dict is {}".format(len(id_dict)))

    end = time.time()
    print("进行data_processing的时间为：{}".format(end-start))