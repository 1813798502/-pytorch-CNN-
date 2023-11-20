import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


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


# 加载模型权重
model = LeNet()
model.load_state_dict(torch.load('model1.pth'))
model.eval()  # 设置模型为评估模式

# 对图像进行与训练时相同的预处理
preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 读取待识别的图片并进行预处理
# img_path = "E:\\python 项目\\pinn\\CNN_picture\\original_pic\\cat\\flickr_cat_000002.jpg"
data_dir = "E:\\python 项目\\pinn\\CNN_picture\\small_data\\train"
dataset = ImageFolder(root=data_dir, transform=preprocess)
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 预测并计算准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {100 * accuracy:.2f}%')
