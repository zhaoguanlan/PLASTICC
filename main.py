# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
----------Loss-----------

# 在Pytorch中 CrossEntropyLoss()等于NLLLoss+ softmax，因此如果用CrossEntropyLoss最后一层就不用softmax了
criterion = nn.NLLLoss(ignore_index=0)

# 2-1. NLL(negative log likelihood) loss of is_next classification result
next_loss = criterion(next_sent_output, data["is_next"])

# 2-2. NLLLoss of predicting masked token word
mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

# 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
loss = next_loss + mask_loss

"""
import BERT_model
import transformer_encoder
import pandas as pd

train_time_data_ = pd.read_csv("training_set.csv")
train_time_data = train_time_data_.copy()


# 8. 初始化模型
input_dim = 192  # 输入维度
hidden_dim = 768  # 隐藏的维度
layer_dim = 1  # 1 层
BASH_SIZE = 16  # 每批读取的数据大小

# 实例化模型传入参数
model = BERT_model(vocab_size = input_dim)

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 9、定义损失函数
# 在Pytorch中 CrossEntropyLoss()等于NLLLoss+ softmax，因此如果用CrossEntropyLoss最后一层就不用softmax了
criterion = nn.NLLLoss(ignore_index=0)
# 2-1. NLL(negative log likelihood) loss of is_next classification result
next_loss = criterion(next_sent_output, data["is_next"])
# 2-2. NLLLoss of predicting masked token word
mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
# 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
loss = next_loss + mask_loss

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
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(train_time_data.head())


