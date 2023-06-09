# from astropy.io import fits
# import os
#
# #读取文件
# lamost=fits.open('data/ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits')
# #查看HDU
# print("info ", lamost.info())
# #查看第1个HDU的header
# print("header", lamost[0].header)
# #查看第1个HDU的data
# print("data ", lamost[0].data)
# #按照关键字搜索第1个HDU的数据（例子中是查看lamost数据的赤经）
# #lamost[0].data.field('ra')
# lamost.close()

import dask.dataframe as dd
import pyarrow.parquet as pq
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotfits(path):
    output = path
    name = path.split("/")[-1]
    # data = pf.getdata(output)
    data = fits.getdata(output)
    print("data.type", type(data))
    print("data.shape:", data.shape)
    # 显示一个数据的范围，因为有些太大的影响图片的效果，可调节一下
    vmax = np.percentile(data, 95)
    vmin = np.percentile(data, 5)
    plt.title(name)
    plt.imshow(data, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.show()

# df = pq.read_table('0/field0697').to_pandas()
# ddf = dd.read_parquet('[01]/field*/*parquet', engine='pyarrow')

if __name__ == '__main__' :
    plotfits("data/ibe-data-ztf-products-cal-2022-0218-bias-00-ccd02-q2/ztf_20220218_00_c02_q2_bias.fits")
    plotfits("data/ibe-data-ztf-products-cal-2022-0218-bias-00-ccd02-q2/ztf_20220218_00_c02_q2_biascmask.fits")
    plotfits("data/ibe-data-ztf-products-cal-2022-0218-bias-00-ccd02-q2/ztf_20220218_00_c02_q2_biasunc.fits")
    plotfits("data/ztf_001596_zr_c16_q2_refimg.fits")
    plotfits("data/ztf_20180411467847_000535_zr_c11_o_q3_sciimg.fits")

    print("-----read_parquet-----")
    ddf = dd.read_parquet('data/ztf_000302_zg_c01_q1_dr13.parquet', engine='pyarrow')
    pd.set_option('display.width', None)
    print(ddf.head())


# 打印最大值最小值平均值和标准差
# print("min:", np.min(data))
# print("Max:", np.max(data))
# print("Mean:", np.mean(data))
# print("Stdev:", np.std(data))






