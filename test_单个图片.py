import os
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


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
data_dir = "E:\\python 项目\\pinn\\CNN_picture\\test"
file_list = os.listdir(data_dir)
# 筛选出图片文件的地址
image_paths = [os.path.join(data_dir, file) for file in file_list if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
# 循环处理每张图片
acc = 0
# 定义图表的行列数
rows = 6  # 6行
cols = 6  # 6列
total_images = rows * cols
fig = plt.figure(figsize=(10, 6))  # 设置图表大小
i = 0
for img_path in image_paths:
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 增加一个维度作为 batch
    # 使用模型进行推理
    with torch.no_grad():
        output = model(img_tensor)
        # 这里根据你的需求为图片分配标签
        # 假设图片名称包含 'cat' 则标签为 0，否则为 1
        if 'cat' in img_path:
            label = 0
        else:
            label = 1
        predicted_class = torch.argmax(output).item()
        if predicted_class == label:
            acc += 1
        if predicted_class == 0:
            string = 'cat'
        else:
            string = 'dog'

    fig.add_subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f'prediction:{string}')
    i += 1
plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图表

print(f'准确率：{acc / len(image_paths)*100}%')
