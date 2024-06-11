import torchvision.models
from torch import nn

from tools.captcha_info import Captcha_Len, Len_of_charset

class ModifiedResNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ModifiedResNet, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.base_model = base_model
        self.base_model.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.bn1 = nn.BatchNorm2d(64)
        self.base_model.relu = nn.ReLU(inplace=True)
        self.base_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.base_model(x)
        return x

def resnet18():
    base_model = torchvision.models.resnet18(weights=None)
    return ModifiedResNet(base_model, Len_of_charset*Captcha_Len)

def resnet34():
    base_model = torchvision.models.resnet34(weights=None)
    return ModifiedResNet(base_model, Len_of_charset*Captcha_Len)

def resnet50():
    base_model = torchvision.models.resnetCaptcha_Len0(weights=None)
    return ModifiedResNet(base_model, Len_of_charset*Captcha_Len)

def resnet101():
    base_model = torchvision.models.resnet101(weights=None)
    return ModifiedResNet(base_model, Len_of_charset*Captcha_Len)

def resnet152():
    base_model = torchvision.models.resnet1Captcha_Len2(weights=None)
    return ModifiedResNet(base_model, Len_of_charset*Captcha_Len)

if __name__ == '__main__':
    resnet50()
