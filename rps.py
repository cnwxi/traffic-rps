import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter
from PIL import Image
# 升腾环境
# import torch_npu 
# from torch_npu.contrib import transfer_to_npu 

#对训练集做数据归一化及增强处理
TRAINING_DIR="./rps/train/"
# 定义图像预处理和增强
def remove_alpha_channel(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    else:
        return image
transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel), # 去除Alpha通道
    transforms.Resize((150, 150)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(40),  # 随机旋转
    transforms.RandomAffine(degrees=0,
                            translate=(0.2, 0.2),
                            shear=0.2,
                            scale=(0.8, 1.2)),  # 随机仿射变换，包括平移、剪切、缩放
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize([1. / 255, 1. / 255, 1. / 255],
                         [1, 1, 1])  # 对每个通道进行归一化，均值为1/255，标准差为1
])
TRAINING_DIR="./rps/train/"
train_dataset = datasets.ImageFolder(TRAINING_DIR, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#对验证集做数据归一化处理
VALIDATION_DIR="./rps/val/"
# 定义验证集的图像预处理
validation_transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel), # 去除Alpha通道
    transforms.Resize((150, 150)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize([1. / 255, 1. / 255, 1. / 255],
                         [1, 1, 1])  # 对每个通道进行归一化
])
VALIDATION_DIR="./rps/val/"
validation_dataset = datasets.ImageFolder(VALIDATION_DIR, transform=validation_transform)

# 创建验证集的数据加载器
validation_loader = DataLoader(validation_dataset,
                               batch_size=32,
                               shuffle=False)

class RPSNet(nn.Module):

    def __init__(self):
        super(RPSNet, self).__init__()
        # 第一组“卷积层+池化层+Dropout层” 输入尺寸 3*150*150
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)  # 64*148*148
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64*74*74
        self.dropout1 = nn.Dropout(0.3)
        # 第二组“卷积层+池化层+Dropout层”
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)  # 64*72*72
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64*36*36
        self.dropout2 = nn.Dropout(0.3)
        # 第三组“卷积层+池化层+Dropout层”
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 128*34*34
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128*17*17
        self.dropout3 = nn.Dropout(0.3)
        # 第四组“卷积层+池化层+Dropout层”
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)  # 128*15*15
        self.activation4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128*7*7
        self.dropout4 = nn.Dropout(0.3)
        # 拉伸层和Dropout层
        self.flatten = nn.Flatten()  # 128*7*7=6272
        self.dropout5 = nn.Dropout(0.5)
        # 全连接层和Dropout层
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout6 = nn.Dropout(0.5)
        # 输出层
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.activation1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.activation2(self.conv2(x))))
        x = self.dropout3(self.pool3(self.activation3(self.conv3(x))))
        x = self.dropout4(self.pool4(self.activation4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout5(x)
        x = F.relu(self.fc1(x))
        x = self.dropout6(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# 创建模型实例并打印模型概要
model = RPSNet()
print(model)

# 检查CUDA是否可用，据此设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): # 英伟达GPU
    print("Using GPU for training")
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # Apple M芯片
        print("Using NPU for training")
        device = torch.device("mps")
else: # CPU
    print("Using CPU for training")
    device = torch.device("cpu")
model = model.to(device)  # 将模型移动到指定的设备上

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

# 创建保存模型权重的文件夹
if not os.path.exists("./weight"):
    os.makedirs("./weight")

# 训练网络模型
def train_model(model, train_loader, validation_loader, epochs):
    writer = SummaryWriter('./runs/rps_log')  # 初始化SummaryWriter
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        with tqdm(total=len(train_loader),
                  ncols=150,
                  desc=f"epoch:{epoch+1}/{epochs}") as pBar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pBar.update()
                pBar.set_description(
                    f"epoch:{epoch+1}/{epochs}|running_loss:{loss.item()/len(inputs)}"
                )
            epoch_loss = running_loss / len(train_loader)/32
            pBar.set_description(
                f"epoch:{epoch+1}/{epochs}|epch_loss:{epoch_loss}")
        writer.add_scalar('Loss/Training', epoch_loss, epoch)  # 记录训练损失
        # 验证阶段
        model.eval()
        total = 0
        correct = 0
        acc = 0.0
        best_acc = 0.0
        with torch.no_grad(), tqdm(total=len(validation_loader),
                                   ncols=150,
                                   desc=f"validate:{epoch}/{epochs}") as pBar:
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pBar.update()
            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "./weight/best_rps.pth")
            pBar.set_description(f"validate:{epoch+1}/{epochs}|acc:{acc:.2f}")
        writer.add_scalar('Validation Accuracy', acc, epoch)  # 记录验证准确率

# 假设 train_loader 和 validation_loader 已经定义
train_model(model, train_loader, validation_loader, epochs=10)

# 保存模型
torch.save(model.state_dict(), "./weight/rps.pth")


# 加载模型
model = RPSNet()
model.load_state_dict(torch.load("./weight/best_rps.pth"))
model = model.to(device)
model.eval()


test_transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel), # 去除Alpha通道
    transforms.Resize((150, 150)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize([1. / 255, 1. / 255, 1. / 255],
                         [1, 1, 1])  # 对每个通道进行归一化
])

# 对单张图像进行预测
def predict_image(model, image_path):
    image = Image.open(image_path)
    image = test_transform(image).to(device)
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


label_map = {0: "paper", 1: "rock", 2: "scissors"}
with torch.no_grad():
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        image_path = f"./rps/test/{i+1}.png"
        img=mpimg.imread(f"./rps/test/{i+1}.png")
        plt.imshow(img)
        plt.title(label_map[predict_image(model, image_path)])
        plt.axis('Off')
    plt.show()
