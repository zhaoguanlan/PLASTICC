# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

#使用url把数据下载下来
data = pd.read_parquet("https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_dr13/0/field000355/ztf_000355_zg_c01_q1_dr13.parquet")
pd.set_option('display.max_columns', None)
print(data.head(5))

time = data['hmjd'][0]
mag = data['mag'][0]

plt.scatter(time,mag)
plt.show()
'''

objectid      int64

filterid       int8
Filter for the lightcurve photometry: 1 => g; 2 => r; 3 => i.

fieldid       int16
The survey Field Identifier. 
Lightcurves in DR13 are spread across 1,154 fields. 
This identifier can be used to retrieve additional archive metadata.

rcid           int8
objra       float32
objdec      float32
nepochs       int64
hmjd         object
mag          object
magerr       object
clrcoeff     object
catflags     object
dtype: object

'''