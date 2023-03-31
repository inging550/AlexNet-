import cv2
import torch
from model import AlexNet
from torchvision import transforms
import scipy.io as io
import time
# 初始化一些参数
net = AlexNet(NUM_CLASS=5, init_weight=False)  # 给神经网络实例化对象
net.load_state_dict(torch.load("AlexNet1.pth"))  # 导入权重参数
net.eval()
transform = transforms.ToTensor()
transform1 = transforms.Normalize(0.5, 0.5)
# 定义数据集
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
        class_i_data = total_data[(i-1)*1000:i*1000, :]
        class_i_label = total_label[(i-1)*1000:i*1000]
        train_y[(i-1)*800:i*800] = i - 1
        test_y[(i - 1) * 200:i * 200] = i - 1
        for k in range(0, 800):
            now_picture = class_i_data[k, :]
            now_picture = cv2.resize(now_picture, (224, 224))
            now_picture = transform(now_picture)
            now_picture = now_picture.unsqueeze(0)
            # now_picture = transform1(now_picture)
            train_x[(i-1)*800+k, :, :, :] = now_picture
        for k in range(0, 200):
            now_picture = class_i_data[800 + k, :]
            now_picture = cv2.resize(now_picture, (224, 224))
            now_picture = transform(now_picture)
            now_picture = now_picture.unsqueeze(0)
            # now_picture = transform1(now_picture)
            test_x[(i-1)*200+k, :, :, :] = now_picture

    print(' train_x.shape:\t', train_x.shape, '\n',
          'train_y.shape:\t', train_y.shape, '\n',
          'test_x.shape:\t', test_x.shape, '\n',
          'test_y.shape:\t', test_y.shape)

    return train_x, train_y, test_x, test_y


_, _, test_x, test_y = dp()
# 各类别名字，根据自己使用的数据集更改
labels_name = ['一个泡', '两个泡', '三个泡', '环流', '层流']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)   # 打印当前使用的设备（GPU(cuda:0)还是CPU）
# 进行预测
class Predict:
    def __init__(self, Test_y, Test_x, net1): # image_root->需要预测的图片路径，net1->网络结构
        self.test_x = Test_x.to(device)
        self.test_y = Test_y.to(device)
        self.model = net1.to(device)
        self.now_correct_num = 0
        self.total_correct_num = 0
        self.correct_class_i = []

    # 统计结果
    def result(self):
        print(len(self.test_x))
        self.model.to(device)
        for i in range(0, len(self.test_x)):
            input = self.test_x[i].unsqueeze(0)
            pred = self.model(input)
            _, class_i = torch.max(pred, 1)
            if class_i == self.test_y[i]:
                self.total_correct_num += 1
                self.now_correct_num += 1
            if i % 199 == 0:
                self.correct_class_i.append(self.now_correct_num)
                self.now_correct_num = 0
        print("分类正确的个数为%d", self.total_correct_num)
        print("各类别的正确率为", self.correct_class_i)
        print("正确率为", float(self.correct_class_i)/200)


if __name__ == "__main__":
    Pre = Predict(test_y, test_x, net)
    result = Pre.result()

