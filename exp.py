import pandas as pd
import numpy as np
import torch
import math

# # 获取文件总行数
# def spilt_file(path):
#     df = pd.read_csv(path)
#     # 确定每个小文件要包含的数据量
#     step = 5000000
#     row_num = len(df)
#     for start in range(0, row_num-step, step):
#         stop = start + step
#         filename = "/home/guanlanzi/new_data/test_set{}_{}.csv".format(start, stop)
#         d = df[start: stop]
#         print("Saving file : " + filename + ", data size : " + str(len(d)))
#         d.to_csv(filename, index=None)
#
#     d = df[stop:row_num]
#     filename = "/home/guanlanzi/new_data/test_set{}_{}.csv".format(start, stop)
#     print("Saving file : " + filename + ", data size : " + str(len(d)))
#     d.to_csv(filename, index=None)
#
# path = "/home/guanlanzi/data/test_set.csv"
# spilt_file(path)

# A = {1:[1,2,3,4],2:3,3:4}
# B = torch.tensor([[1,2,3,4],[2,3,4,5]])
# print([A.get(list(A.keys())[-1])])
# print(len(A))
# print(math.log(4,1.1))
# print(B.unsqueeze(0))
# print(B.unsqueeze(1))
# print(B.unsqueeze(2))
#
# print(math.log(1.21,1.1))
# print('Real isNext :', bool([1,0,0,1,1]))
# path = "home/guanlanzi/data/test_set_batch1"
# print("home/guanlanzi/new_data/" + path.split("/")[-1])

list = [1,2,3,4]
print(list[2:])


