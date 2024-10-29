import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

# 数据预处理和加载
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 输入尺寸为224x224
        transforms.ToTensor(),
    ]
)


# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# 定义 ResNet101 网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 实例化 ResNet101
def resnet101(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes)


model = resnet101()

# 将模型移到 GPU（如果可用）
try:
    model.load_state_dict(torch.load("models/0.pth", weights_only=True))
except:
    pass

if __name__ == "__main__":
    for image in os.listdir("unknown"):
        image = cv2.imread("unknown/" + image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(Image.fromarray(image))
        print(image)
        plt.imshow(image.permute(1, 2, 0).cpu())  # 转换通道顺序为(H, W, C)
        plt.title(f"Predicted: {model(image.unsqueeze(0))}")
        plt.axis("off")
        plt.show()

    # 定义超参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # 加载数据集
    train_dataset = ImageFolder(root="dataset", transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # 实例化模型
    model = resnet101()

    # 将模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load("models/0.pth", weights_only=True))
    except:
        pass
    model = model.to(device)

    images, labels = next(iter(train_loader))
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs.data, 1)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 设置 L1 正则化的超参数
    l1_lambda = 0.00005  # 调整该值以控制正则化强度

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(x := tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.argmax(dim=-1)
            loss = criterion(outputs, labels)

            if i % 64 == 0:
                # 计算 L1 正则化项
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm  # 添加 L1 正则化项

            loss.backward()
            acc = (logits == labels).float().mean()
            optimizer.step()
            x.set_postfix(loss=loss.item(), acc=acc.item())

            # 保存模型为 .pth 文件
            if i % 64 == 0:
                torch.save(model.state_dict(), "models/%d.pth" % epoch)
    exit(0)


# 实例化模型
model = resnet101()

# 将模型移到 GPU（如果可用）
try:
    model.load_state_dict(torch.load("models/0.pth", weights_only=True))
except:
    pass


def check_image(image: np.ndarray) -> bool:
    """
    :param image: OpenCV图像数组。
    :return: 如果图像中有奶龙，返回True；否则返回False。
    """
    from time import time

    origin = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(Image.fromarray(image))
    image = image.unsqueeze(0)
    output = model(image).squeeze(0)
    print(image, output)

    if (output[0] - output[1]).abs() < 5:
        cv2.imwrite("unknown/%d.png" % time(), origin)
    return output.argmax() == 0
