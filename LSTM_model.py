import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
"""
读入数据
"""

train_metadata = pd.read_csv("training_set_metadata.csv")
train_data = train_metadata.copy()
train_time_data_ = pd.read_csv("training_set.csv")
train_time_data = train_time_data_.copy()

# train_metadata.head()
# print(train_metadata.info())
# print("----------ddf---------")
# print(train_metadata["ddf"].value_counts())
# print("-------describe-------")
# print(train_metadata.describe())
# train_metadata.hist(bins=50,figsize=(20,15))
# plt.show()

"""
缺失数据填补：
"""

# 1.method = 'ffill'/'pad'：用前一个非缺失值去填充该缺失值
# data.fillna(method='ffill')
# 2.method = 'bflii'/'backfill'：用下一个非缺失值填充该缺失值
# data.fillna(method='bfill')
# 3.用常数填充
# data.fillna(100)

# print(train_metadata["distmod"].head(50))
train_metadata["distmod"] = train_metadata["distmod"].fillna(method='ffill')
train_metadata["distmod"] = train_metadata["distmod"].fillna(method='bfill')
# print(train_metadata["distmod"])
# print(train_metadata.info())

# 把经度纬度可视化
# train_data.plot(kind="scatter", x="ra", y="decl", alpha=0.2)
# corr_matrix = train_data.corr()
# 查看变量与target的相关系数
# print(corr_matrix["target"].sort_values(ascending=False))
#
# attributes = ["target", "hostgal_specz", "distmod", "hostgal_photoz"]
# scatter_matrix(train_data[attributes], figsize=(12, 8))
# plt.show()
# 单独保存target，并读入时间数据

training_data = train_data.drop("target", axis=1)
training_labels = train_data["target"].copy()
# print("train_time_data.head()",train_time_data.head())
# print("train_metadata.head()",train_data.head())

#
# print(train_time_data.info())
# print("----------detected---------")
# print(train_time_data["detected"].value_counts())
# print("-------describe-------")
# print(train_time_data.describe())

# 把label和数据一一对应起来
# Flux

"""
将同一id的数据放进同一列表中，并将同一个通道的数据按时间顺序进行排列
"""

obj_id = []
for i in range(train_time_data.shape[0]):
    if train_time_data["object_id"][i] not in obj_id:
        obj_id.append(train_time_data["object_id"][i])
# print(len(obj_id))

# pd.set_option("display.max_rows", 1000)  # 可显示1000行
# train_time_data[train_time_data["object_id"]==obj_id[0]][["mjd","passband"]]

Flux = [[[] for j in range(6)] for i in range(len(obj_id))]
for i in range(train_time_data.shape[0]):
    l = obj_id.index(train_time_data["object_id"][i])
    passband_num = train_time_data["passband"][i]
    Flux[l][passband_num - 1].append(train_time_data["flux"][i])

# print(Flux[0])
# print("----id_1-----")
# print(np.array(Flux[1][0]).shape)
# print(np.array(Flux[1][1]).shape)
# print(np.array(Flux[1][2]).shape)
# print(np.array(Flux[1][3]).shape)
# print(np.array(Flux[1][4]).shape)
# print(np.array(Flux[1][5]).shape)
# print("----id_0-----")
# print(np.array(Flux[0][0]).shape)
# print(np.array(Flux[0][1]).shape)
# print(np.array(Flux[0][2]).shape)
# print(np.array(Flux[0][3]).shape)
# print(np.array(Flux[0][4]).shape)
# print(np.array(Flux[0][5]).shape)
# print("----id_2-----")
# print(np.array(Flux[2][0]).shape)
# print(np.array(Flux[2][1]).shape)
# print(np.array(Flux[2][2]).shape)
# print(np.array(Flux[2][3]).shape)
# print(np.array(Flux[2][4]).shape)
# print(np.array(Flux[2][5]).shape)
# print("----id_3-----")
# print(np.array(Flux[3][0]).shape)
# print(np.array(Flux[3][1]).shape)
# print(np.array(Flux[3][2]).shape)
# print(np.array(Flux[3][3]).shape)
# print(np.array(Flux[3][4]).shape)
# print(np.array(Flux[3][5]).shape)

"""
数据分类完成之后发现，长度不一样，所以直接选择最大长度零填充，可以改进！
"""

min_len = 999
max_len = 0
for i in range(len(Flux)):
    for j in range(6):
        min_len = min(min_len, len(Flux[i][j]))
        max_len = max(max_len, len(Flux[i][j]))
# print("min_length is ", min_len)
# print("max_length is ", max_len)
# print(np.array(Flux).shape)
# 下下策
for i in range(len(Flux)):
    for j in range(6):
        Flux[i][j].extend(0 for _ in range(abs(max_len - len(Flux[i][j]))))
# print(np.array(Flux).shape)

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28,28)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     #概率默认0.5
#     transforms.RandomRotation(90),
#     transforms.RandomGrayscale(0.1),
#     transforms.ColorJitter(0.3, 0.3, 0.3),
#     transforms.ToTensor()
# ])
# Flux
# obj_id
# train_data

"""
target数据转换成【0 - 14】的列表
"""

label = train_data["target"].copy()
labels = [90, 42, 65, 16, 15, 62, 88, 92, 67, 52, 95, 6, 64, 53]
for i in range(len(label)):
    label[i] = labels.index(label[i])

"""
训练测试数据分割
"""

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Flux, label,
                                                                            test_size=0.2,
                                                                            random_state=1)

# print(len(X_train))
# print(len(X_train[0]))
# print(len(X_train[0][0]))
# print(len(y_train))
# print(len(X_test))
# print(len(X_test[0]))
# print(len(X_test[0][0]))

"""
定义dataset类把数据转换成Dataload的格式
"""

class MyDataset(Dataset):
    # data1,data2 分别是_train_time_data和label
    def __init__(self,
                 data1,
                 data2,
                 transform=None):
        super(MyDataset, self).__init__()

        data = []
        data1 = list(data1)
        data2 = list(data2)
        data = [*zip(data2, data1)]
        # for i in range(len(data1)):
        # data.append([data2[i], data1[i]])
        # pass
        # exit()

        self.data = np.array(data)
        self.transform = transform
        # self.loader = loader

    def __getitem__(self, index):
        labels, datas = self.data[index]
        if self.transform is not None:
            datas = self.transform(np.array(datas))
        # datas = torch.from_numpy(datas).long()
        # labels = torch.from_numpy(labels).long()
        return datas, labels
    def __len__(self):
        return len(self.data)

train_dataset = MyDataset(X_train,
                          y_train,
                          transform=transforms.ToTensor())
test_dataset = MyDataset(X_test,
                         y_test,
                         transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=16,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=16,
                         shuffle=False)

print("num of train", len(train_dataset))
test_num = len(test_dataset)

# 3、查看数据集大小shape
# print(trainsets.data.shape)
# print(trainsets.targets.shape)

# 4、定义超参数
# BASH_SIZE = 16  # 每批读取的数据大小
# EPOCHS = 10 #训练十轮
# 创建数据集的可迭代对象，也就是说一个batch一个batch的读取数据
# train_loader = torch.utils.data.DataLoader(dataset = trainsets, batch_size = BASH_SIZE,shuffle = True)
# test_loader = torch.utils.data.DataLoader(dataset = testsets, batch_size = BASH_SIZE,shuffle = True)
# 查看一批batch的数据
# 此处需要自己创建DataLoader

# images, labels = next(iter(test_loader))
# print(images.shape)

# 6、定义函数，显示一批数据
# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406]) # 均值
#     std = np.array([0.229, 0.224, 0.225]) # 标准差
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1) # 限速值限制在0-1之间
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
# 网格显示
# out = torchvision.utils.make_grid(images)

# imshow(out)

# 7. 定义RNN模型

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device = 'cuda'):
        super(LSTM_Model, self).__init__()  # 初始化父类中的构造方法
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 构建LSTM模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层：
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, device = 'cuda'):
        # 初始化隐藏层装态全为0
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device).requires_grad_()
        # 初始化cell state

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device).requires_grad_()

        x = x.to(torch.float32)
        #         c0 = c0.to(torch.float32)
        #         h0 = h0.to(torch.float32)

        # 分离隐藏状态，以免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out = out.to(device)
        # hn = hn.to(device)
        # cn = cn.to(device)

        # 只需要最后一层隐层的状态
        out = self.fc(out[:, -1, :])
        return out


# 8. 初始化模型
input_dim = 72  # 输入维度
hidden_dim = 128  # 隐藏的维度
layer_dim = 1  # 1 层
output_dim = 14  # 输出维度
BASH_SIZE = 16  # 每批读取的数据大小

# 实例化模型传入参数
model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 9、定义损失函数
criterion = nn.CrossEntropyLoss()

# 10、定义优化函数
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 11、输出模型参数
length = len(list(model.parameters()))

# 12、循环打印模型参数
for i in range(length):
    print('参数： %d' % (i + 1))
    print(list(model.parameters())[i].size())

# 13 、模型训练
sequence_dim = 6  # 序列长度
loss_list = []  # 保存loss
accuracy_list = []  # 保存accuracy
iteration_list = []  # 保存循环次数
iter = 0
EPOCHS = 50
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # 声明训练
        # 一个batch的数据转换为LSTM的输入维度
        images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(torch.float32)

        images = images.to(device)
        labels = labels.to(device)
        # 梯度清零（否则会不断增加）
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # outputs = outputs.to(device)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数自动加一
        iter += 1
        # 模型验证
        if iter % 500 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集、获取数据、预测
            for images, labels in test_loader:
                images = images.view(-1, sequence_dim, input_dim).to(device)
                # 模型预测
                outputs = model(images)
                # 获取预测概率的最大值的下标
                #                print(outputs)
                predict = torch.argmax(outputs.data, 1).to(device)
                labels = labels.to(device)
                #                 new_predict = np.zeros((32, 14))
                #                 new_predict[np.arange(32),predict] = 1

                # 统计测试集的大小
                total += labels.size(0)
                # 统计判断/预测正确的数量
                #                 print(predict)
                #                 print(labels.shape)
                correct += (predict == labels).sum()
            # 计算 accuracy
            accuracy = (correct / total) / 100 * 100
            # 保存accuracy， loss iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("epoch : {}, Loss : {}, Accuracy : {}".format(iter, loss.item(), accuracy))
# 可视化 loss

plt.plot(iteration_list, loss_list)
plt.xlabel('Number of Iteration')
plt.ylabel('Loss')
plt.title('LSTM')
plt.show()

# 可视化 accuracy
plt.plot(iteration_list, accuracy_list, color='r')
plt.xlabel('Number of Iteration')
plt.ylabel('Accuracy')
plt.title('LSTM')
plt.savefig('LSTM_accuracy.png')
plt.show()

"""
# 首先定义一个 分类数*分类数 的空混淆矩阵
label_kinds = 14

conf_matrix = torch.zeros(label_kinds, label_kinds)


# 使用torch.no_grad()可以显著降低测试用例的GPU占用

def confusion_matrix(preds, labels, con_matrix):
    if len(preds) == 1:
        con_matrix[preds.item(), labels.item()] += 1
        return con_matrix
    for i in range(len(preds)):
        con_matrix[preds[i].item(), labels[i].item()] += 1
    return con_matrix


with torch.no_grad():
    for step, (imgs, targets) in enumerate(test_loader):
        targets = targets.squeeze()  # [50,1] ----->  [50]
        imgs = imgs.view(-1, sequence_dim, input_dim).to(device)

        # 将变量转为gpu
        #         targets = targets.cuda()
        #         imgs = imgs.cuda()
        # print(step,imgs.shape,imgs.type(),targets.shape,targets.type())

        out = model(imgs)
        predict = torch.argmax(out.data, 1)
        #         #记录混淆矩阵参数
        #         print(predict)
        #         print(targets)
        #         print("----------------------")
        conf_matrix = confusion_matrix(predict, targets, conf_matrix)
#         conf_matrix=conf_matrix.cpu()


conf_matrix = np.array(conf_matrix)  # 将混淆矩阵从gpu转到cpu再转到np
corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

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
plt.close()

"""
