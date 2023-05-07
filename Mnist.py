import torch
import torch.nn as nn  # 神经网络的各层操作
import torchvision  # 图像处理, 常用数据集, 模型, 转换函数等
from torch.autograd import Variable
import matplotlib.pyplot as plt  # 绘图包
import torch.nn.functional as F  # 神经网络的各层操作
import torch.utils.data as Data  # 数据加载器

EPOCH = 5
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True  # 本地有数据集, 不用下载
# 对于 DOWNLOAD_MNIST 这个变量，是函数的torchvision.datasets.MNIST()函数里面的一个参数，如果为True表示从网上下载该数据集并放进指定目录
# 这是定义了一个tuple元组, 定义后不可修改

# torchvision.datasets 是一个PyTorch的一个接口, 对于常用的数据集都有自己的API
train_data = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,  # true为训练集, false为测试集
    transform=torchvision.transforms.ToTensor(),  # 将图像转为[0,1]的Tensor
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[50].numpy(), cmap="gray")
# plt.title('%i' % train_data.targets[50])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./MNIST', train=False)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[
         :2000] / 255  # shape from (2000, 28, 28) to (2000, 1, 28, 28) value in range(0,1)
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 有序容器, 简化网络, 简化forward
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # b_x = x
        # b_y = y
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0)) # 这是啥
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(cnn, 'cnn_mnist.pkl')
print('finish training')