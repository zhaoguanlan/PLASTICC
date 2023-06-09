import pandas as pd
import matplotlib.pyplot as plt
import json

# data = pd.read_csv('/home/guanlanzi/data/test_set.csv')
s_d = open("sentences_dict.json", "r")
sentencedict = json.load(s_d)

max_len = 0
min_len = 100
for i in range(len(sentencedict)):
    for j in range(6):
        max_len = max(max_len,len(sentencedict[str(i)][j]))
        min_len = min(min_len,len(sentencedict[str(i)][j]))

print("len of sentence dict is_{}".format(len(sentencedict)))
print("max_len of test_data is_{}".format(max_len))
print("min_len of test_data is_{}".format(min_len))

m_d = open("mjd_dict.json", "r")
mjddict = json.load(m_d)

max_value = 0
for i in range(len(mjddict)):
    for j in range(6):
        max_value = max(max_value, max(mjddict[str(i)][j]))

print("max_time of test_data is_{}".format(max_value))

#
# # 1.求日期的最大值、最小值
# mjd_max = data['mjd'].max()
# print("时间的最大值为:{}".format(mjd_max))
# mjd_min = data['mjd'].min()
# print("时间的最小值为:{}".format(mjd_min))
#
# # 1.flux的最大值、最小值
# flux_max = data['flux'].max()
# print("flux的最大值为:{}".format(flux_max))
# flux_min = data['flux'].min()
# print("flux的最小值为:{}".format(flux_min))

"""
时间的最大值为:60674.363
时间的最小值为:59580.0338
flux的最大值为:13675792.0
flux的最小值为:-8935484.0
"""