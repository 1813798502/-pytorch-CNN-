"""
运用CNN进行猫狗识别，进行100回合，准确率为95，7%
1、先对图片进行预处理操作
    a、裁剪统一尺寸
    b、旋转一定角度
    c、随机翻转
    b、c只有训练集有
2、运用datasets.ImageFolder对图片进行数据读取
3、取出部分图片查看是否正常读取
4、定义网络
    a、卷积层
    b、池化层
    c、连接层
5、定义损失函数、优化器
6、定义训练函数
7、定义测试函数
8、保存模型
9、验证模型
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# 图片地址
train_datadir = 'E:\\python 项目\\pinn\\CNN_picture\\big_data\\train\\'
test_datadir = 'E:\\python 项目\\pinn\\CNN_picture\\big_data\\val\\'

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.RandomRotation(degrees=(-15, 15)),  # 随机旋转，-10到10度之间随机选
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
    # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转（效果可能会变差）
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),  # 随机视角
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

train_data = datasets.ImageFolder(train_datadir, transform=train_transforms)
test_data = datasets.ImageFolder(test_datadir, transform=test_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"


def im_convert(tensor):
    """ 展示数据"""
    #     tensor.clone()  返回tensor的拷贝，返回的新tensor和原来的tensor具有同样的大小和数据类型
    #     tensor.detach() 从计算图中脱离出来。
    image = tensor.to("cpu").clone().detach()

    #     numpy.squeeze()这个函数的作用是去掉矩阵里维度为1的维度
    image = image.numpy().squeeze()
    #     将npimg的数据格式由（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）,
    #     进行格式的转换后方可进行显示
    image = image.transpose(1, 2, 0)
    #     和标准差操作正好相反即可
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    #     使用image.clip(0, 1) 将数据 限制在0和1之间
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 20))
columns = 2
rows = 2

dataiter = iter(train_loader)
for idx, (inputs, classes) in enumerate(train_loader):
    # 只展示一个固定数量的图像样例
    if idx >= columns * rows:
        break
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    # 将Tensor移动到CPU上，并展示图像及其对应的标签
    plt.imshow(im_convert(inputs[idx]))
    if classes[idx] == 0:
        ax.set_title("cat", fontsize=20)
    else:
        ax.set_title("dog", fontsize=20)
plt.tight_layout()
plt.savefig('pic1.jpg', dpi=600)  # 保存图像
plt.show()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为53*53，所以全连接层的输入是16*53*53
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在需要应用Dropout的地方调用Dropout层
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 在需要应用Dropout的地方调用Dropout层
        x = self.fc3(x)
        return x


model = LeNet().to(device)
# print(model)
# 定义损失函数，优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    model.train()
    losses = []  # 用于收集损失值
    for epoch in range(epochs):
        epoch_loss = 0.0  # 每个epoch的损失值
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 计算预测误差
            pred = model(X)
            loss = loss_fn(pred, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"epoch: [{epoch + 1}/{epochs}], loss: {avg_epoch_loss:.6f}")
    # 画出损失曲线
    plt.plot(losses, label='Training loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over iterations')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1000
train(train_loader, model, loss_fn, optimizer, epochs)
test(test_loader, model, loss_fn)

# 保存整个模型的状态字典
torch.save(model.state_dict(), 'cnn.pth')
