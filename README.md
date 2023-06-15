# PLASTICC&ZTF

在这个文档中我们完成了使用BERT模型对光变曲线数据分类的任务.

数据来源:<https://www.kaggle.com/competitions/PLAsTiCC-2018>

> Help some of the world's leading astronomers grasp the deepest properties of the universe.The human eye has been the arbiter for the classification of astronomical sources in the night sky for hundreds of years. 
> But a new facility -- the Large Synoptic Survey Telescope (LSST) -- is about to revolutionize the field, discovering 10 to 100 times more astronomical sources that vary in the night sky than we've ever known. 
> Some of these sources will be completely unprecedented!

> The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) asks Kagglers to help prepare to classify the data from this new survey. 
> Competitors will classify astronomical sources that vary with time into different classes, scaling from a small training set to a very large test set of the type the LSST will discover.

## 任务目标：使用光度数据进行瞬变源分类

## 任务背景
1. 光度曲线是通过对同一天空区域在不同时刻进行拍摄，对图像的差分的得到的
2. 光通量变化的具体方式（变亮的时间长度、物体在不同通道中变亮的方式、以及消退时间等）是判定物体类型的良好指标。
## 方案
在PLAsTiCC中，要求将光源分为14类以及一类未知的类别，所以我们使用NLP中的BERT模型对时间序列数据进行分类，一方面使用conformal inference对out of distribution sources进行判定

## 数据特征：
1. 缺失性和稀疏性, 数据本身是天文学家的观测结果, 所以对于感兴趣的光源就会 多次就行观察, 反之则可能仅仅只有几次观测数据 (在有 label 的数据集中观测次数最多 的达到了 72 次, 而最少的仅仅只有 2 次) 
2. 不规则的时间间隔, 天文观测中时间变量为 mjd, 不能保证时间的等间隔.
3. 有标签数据的稀缺, 在 PLASTicc 数据集中, 有标签的数据仅仅只有 500M, 没有标 签的观测数据有 20G

![image](/picture/1.png)

## 模型结构：
在模型中，主要是使用transformer的编码器进行掩码学习，在此我们引入了一个Same light source discrimination的loss函数，因为不同频段的信号会有所联系，所以我们将同一光源不同通道之间视为正样本对，不同光源之间视为负样本对。在make batch的时侯构造正负样本对。

![image](/picture/2.png)

## 代码
代码部分final_model.py文件为预训练模型的文件

1. 在这个文件中数据预处理部分将时间属性mjd离散化，最终转换到0-1199的整数范围中
2. 光通量通过底数为1.1的log函数进行转换，最终转换为0-399的整数范围
3. 最大的passend长度为72，所以我们设计的max_len为75
4. 最终每一个光源的光度数据转换为一个(6,75)的矩阵
5. make_batch的时候是随机取一个点光源的某一个passend作为输入
6. 后来在final_model_all_passend.py文件中把make_batch的方法改为随机选取一个,将其六个通道的数据拼到一起,max_len为450

Fine_tuneallpassend_3.py为fine_tune的文件, 我设计了两层全连接网络作为fine_tune层，抽取出0.2的数据作为验证集,最后达到了75%的精度

## 其他数据
其他光变曲线的无标签数据有ZTF的光度数据,因时间有限来不及处理，下载链接为
<https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_dr13>
若有不清楚的地方欢迎与我讨论~



   

