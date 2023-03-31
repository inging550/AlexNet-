import torch.types
from torchvision import datasets, transforms
import torch.optim as optioms
from model import AlexNet
import torch.nn as nn
import torch.utils.data as data
import scipy.io as io
import cv2
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # 打印当前使用的设备（GPU(cuda:0)还是CPU）
num_epochs = 10  # 设定的迭代次数上限
learning_rate = 1E-2  # 初始学习率
best_test_loss = 20.0
# 1、定义数据集
transform = transforms.ToTensor()
transform1 = transforms.Normalize(0.5, 0.5)


def dp():
    path = "data.mat"  # 定义路径
    matr = io.loadmat(path)  # 关键步骤 采用io的loadmat包将数据读入缓存
    total_data = matr['sample_features']
    total_label = matr['sample_category']
    train_x = torch.zeros(4000, 1, 224, 224)
    train_y = torch.zeros(4000, 1)
    test_x = torch.zeros(1000, 1, 224, 224)
    test_y = torch.zeros(1000, 1)
    for i in range(1, 6):
        class_i_data = total_data[(i - 1) * 1000:i * 1000, :]
        class_i_label = total_label[(i - 1) * 1000:i * 1000]
        train_y[(i - 1) * 800:i * 800] = i - 1
        test_y[(i - 1) * 200:i * 200] = i - 1
        for k in range(0, 800):
            now_picture = class_i_data[k, :]
            now_picture = cv2.resize(now_picture, (224, 224))
            now_picture = transform(now_picture)

            now_picture = now_picture.unsqueeze(0)
            # now_picture = transform1(now_picture)
            train_x[(i - 1) * 800 + k, :, :, :] = now_picture
        for k in range(0, 200):
            now_picture = class_i_data[800 + k, :]
            now_picture = cv2.resize(now_picture, (224, 224))
            now_picture = transform(now_picture)

            now_picture = now_picture.unsqueeze(0)
            # now_picture = transform1(now_picture)
            test_x[(i - 1) * 200 + k, :, :, :] = now_picture

    print(' train_x.shape:\t', train_x.shape, '\n',
          'train_y.shape:\t', train_y.shape, '\n',
          'test_x.shape:\t', test_x.shape, '\n',
          'test_y.shape:\t', test_y.shape)

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = dp()

train_dataset = data.TensorDataset(train_x, train_y)
test_dataset = data.TensorDataset(test_x, test_y)

train_loader = data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=2,  # mini batch size
    shuffle=True  # 要不要打乱数据 (打乱比较好)
)
test_loader = data.DataLoader(
    dataset=test_dataset,  # torch TensorDataset format
    batch_size=2,  # mini batch size
    shuffle=True  # 要不要打乱数据 (打乱比较好)
)

# 2、定义网络结构并设置为CUDA
net = AlexNet(NUM_CLASS=5, init_weight=False)  # NUM_CLASS为当前数据集的类别总数
net.to(device)

# 打印模型信息
for k, v in net.named_parameters():
    print(k)

# 3、定义损失函数及优化器，损失函数设置为CUDABCEWithLogitsLoss

loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
# loss_function = nn.MSELoss()
optimizer = optioms.SGD(params=net.parameters(), lr=learning_rate)  # SGD随机梯度下降
loss_function.to(device)

# 4、开始训练
for epoch in range(num_epochs):
    net.train()  # 网络有Dropout，BatchNorm层时一定要加
    if epoch == 2:
        learning_rate = 1E-2
    if epoch == 4:
        learning_rate = 1E-3
    if epoch == 8:
        learning_rate = 1E-4
    for param_group in optimizer.param_groups:  # 其中的元素是2个字典；optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
        # optimizer.param_groups[1]： 好像是表示优化器的状态的一个字典；
        param_group['lr'] = learning_rate  # 更改全部的学习率
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    net.eval()
    for i, (images, target) in enumerate(train_loader):  # image为图片，target为其对应的标签（类别名）
        images, target = images.cuda(), target.cuda()  # 设置为CUDA
        pred = net(images)  # 图片输入网络得到预测结果
        # time.sleep(3)
        # print(pred)
        # print(target)
        if i % 200 == 0:
            print(pred)
            print(target)
        target = target.squeeze()
        target = target.long()
        loss = loss_function(pred, target)  # 将预测结果与实际标签比对（计算两者之间的损失值）
        total_loss += loss.item()

        optimizer.zero_grad()  # 将梯度归零，有助于梯度 ssa下降
        loss.backward()  # 反向传播 计算梯度
        optimizer.step()  # 根据梯度 更新模型参数
        # if (i + 1) % 5 == 0:
        #     print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
        #                                                                          i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
    validation_loss = 0.0
    for i, (images, target) in enumerate(test_loader):  # 导入dataloader 说明开始训练了  enumerate 建立一个迭代序列
        images, target = images.cuda(), target.cuda()
        pred = net(images)  # 将图片输入
        target = target.squeeze()
        target = target.long()
        loss = loss_function(pred, target)
        validation_loss += loss.item()  # 累加loss值  （固定搭配）
    validation_loss /= len(test_loader)  # 计算平均loss
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'AlexNet1.pth')  # 保存模型参数



