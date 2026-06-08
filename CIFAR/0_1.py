import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

# ===================== 全局配置（CPU运行）=====================
device = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
NUM_CLASSES = 100  # CIFAR100分类数

# ===================== 1. 数据预处理 & 数据集加载 =====================
def get_dataloader():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

# ===================== 2. 基础组件：ResNet 基础块 =====================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# ===================== 3. 四大改进模块 =====================
# 3.1 Coordinate Attention (CA)
class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden = max(8, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.fc_h = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        self.fc_w = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.avg_pool_h(x).permute(0,1,3,2)
        x_w = self.avg_pool_w(x)
        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = torch.relu(self.bn1(self.fc1(x_cat)))
        x_h, x_w = torch.split(x_cat, [H, W], dim=2)
        x_h = self.fc_h(x_h.permute(0,1,3,2))
        x_w = self.fc_w(x_w)
        att = torch.sigmoid(x_h + x_w)
        return x * att

# 3.2 FcaNet 频域通道注意力
class FcaLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.gap(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

# 3.3 Res2Net 多尺度卷积块
class Res2Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, scale=4):
        super().__init__()
        self.scale = scale
        self.width = planes // scale
        self.conv1 = nn.Conv2d(in_planes, self.width*scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.width, self.width, 3, stride, 1, bias=False),
                nn.BatchNorm2d(self.width)
            ) for _ in range(scale-1)
        ])
        self.conv2 = nn.Conv2d(self.width*scale, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        out_list = []
        for i in range(self.scale-1):
            if i ==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            out_list.append(sp)
        out_list.append(spx[-1])
        out = torch.cat(out_list, dim=1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# 3.4 Conv2former Transformer风格卷积块
class Conv2formerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ===================== 4. 多版本模型 =====================
# 4.1 基线：精简版ResNet18
class ResNet_Slim(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 4.2 ResNet + CA
class ResNet_CA(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2,2,2,2])
        self.ca = CoordAttention(512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = torch.relu(self.backbone.bn1(x))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.ca(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.backbone.linear(x)
        return x

# 4.3 ResNet + FCA
class ResNet_FCA(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2,2,2,2])
        self.fca = FcaLayer(512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = torch.relu(self.backbone.bn1(x))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.fca(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.backbone.linear(x)
        return x

# 4.4 Res2Net
def Res2Net_Slim():
    return ResNet_Slim(Res2Block, [2,2,2,2])

# 4.5 ResNet + Conv2former
class ResNet_Conv2former(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNet_Slim(BasicBlock, [2,2,2,2])
        self.conv2f = Conv2formerBlock(512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = torch.relu(self.backbone.bn1(x))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.conv2f(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.backbone.linear(x)
        return x

# ===================== 5. 训练 & 测试 =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def test(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

# ===================== 6. 主运行 =====================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()
    model_list = [
        ("Base_ResNet", ResNet_Slim(BasicBlock, [2,2,2,2])),
        ("ResNet_CA", ResNet_CA()),
        ("ResNet_FCA", ResNet_FCA()),
        ("Res2Net", Res2Net_Slim()),
        ("ResNet_Conv2former", ResNet_Conv2former())
    ]

    for name, model in model_list:
        print(f"\n========== 训练模型: {name} ==========")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc = test(model, test_loader, criterion)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"Epoch {epoch+1:2d} | Train Loss:{train_loss:.3f} | Train Acc:{train_acc:.2f}% | Test Acc:{test_acc:.2f}%")
        
        print(f"\n【{name}】最终最优测试准确率 = {best_acc:.2f}%\n")