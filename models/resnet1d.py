import torch.nn as nn
import torchvision.transforms as transforms
import math

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=(1, 1))
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, stride=stride, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
    
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet1d(nn.Module):

    def __init__(self, num_classes=4):
        super(ResNet1d, self).__init__()
        self.num_classes = num_classes
        self.downsample = nn.MaxPool1d(2, 2) 
        self.conv1 = nn.Conv1d(in_channels = 2, out_channels=32, kernel_size=1, padding = 0) # 2 * 1024
        self.block1 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        self.block2 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        self.block3 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        self.block4 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))

        # self.block4 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        # self.block5 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        # self.block6 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=2, downsample=self.downsample))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.SELU())
        self.fc2 = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.SELU())
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
