import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def create_model_instance(dataset_type, model_type, class_num=10):
    # return VGG9()
    return EMNIST_CNN1(),EMNIST_CNN2()
    # return AlexNet_DF1(),AlexNet_DF2()
    return IMAGE100_VGG16_1(),IMAGE100_VGG16_2()
    return TinyIMAGE_VGG16_1(), TinyIMAGE_VGG16_2()
    # return TinyIMAGE_ResNet8_1(), TinyIMAGE_ResNet8_2()
    return TinyIMAGE_ResNet18_1(), TinyIMAGE_ResNet18_2()

class AlexNet_DF1(nn.Module):
    def __init__(self):
        super(AlexNet_DF1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class AlexNet_DF2(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet_DF2, self).__init__()
        
        # self.f2= nn.Sequential(
            
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    def forward(self, x):
        # x = self.f2(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x

class EMNIST_CNN1(nn.Module):
    def __init__(self):
        super(EMNIST_CNN1,self).__init__()

        self.conv1 = nn.Sequential(        
            nn.Conv2d(1,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv2 = nn.Sequential(        
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        return out_conv2

class EMNIST_CNN2(nn.Module):
    def __init__(self):
        super(EMNIST_CNN2,self).__init__()
        self.fc1 = nn.Linear(7*7*64,512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self,out_conv2):
        output = out_conv2.view(-1,7*7*64)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output

class IMAGE100_VGG16_1(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),
            
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class IMAGE100_VGG16_2(nn.Module):
    def __init__(self, class_num=100):
        super(IMAGE100_VGG16_2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512*25, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = x.view(x.size(0), 512*25)
        x = self.classifier(x)
        return x


class TinyIMAGE_VGG16_1(nn.Module):
    def __init__(self, class_num=100):
        super(TinyIMAGE_VGG16_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,padding=1),
            
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class TinyIMAGE_VGG16_2(nn.Module):
    def __init__(self, class_num=100):
        super(TinyIMAGE_VGG16_2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 200)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample:
#             identity = self.downsample(x)
#         return F.relu(out + identity)

class TinyIMAGE_ResNet8_1(nn.Module):
    def __init__(self):
        super(TinyIMAGE_ResNet8_1, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return x
    

class TinyIMAGE_ResNet8_2(nn.Module):
    def __init__(self, class_num=200):
        super(TinyIMAGE_ResNet8_2, self).__init__()
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1  # ResNet18 不扩展通道

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class TinyIMAGE_ResNet18_1(nn.Module):
    def __init__(self, block=BasicBlock):
        super(TinyIMAGE_ResNet18_1, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入较小，不用stride=2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # layer 配置：ResNet18 每层重复2个 BasicBlock
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class TinyIMAGE_ResNet18_2(nn.Module):
    def __init__(self, class_num=200):
        super(TinyIMAGE_ResNet18_2, self).__init__()
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x