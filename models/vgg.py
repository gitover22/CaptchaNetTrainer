
import torchvision.models
from torch import nn
from tools.captcha_info import Captcha_Len, Len_of_charset

def vgg11():
    vgg11 = torchvision.models.vgg11(weights=None)
    vgg11.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))
    vgg11.classifier.add_module("my_Linear", nn.Linear(1000, Len_of_charset*Captcha_Len))
    return vgg11

def vgg13():
    vgg13 = torchvision.models.vgg13(weights=None)
    vgg13.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))
    vgg13.classifier.add_module("my_Linear", nn.Linear(1000, Len_of_charset*Captcha_Len))
    return vgg13

def vgg16():
    vgg16 = torchvision.models.vgg16(weights=None)
    vgg16.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))
    vgg16.classifier.add_module("my_Linear", nn.Linear(1000, Len_of_charset*Captcha_Len))
    return vgg16

def vgg19():
    vgg19 = torchvision.models.vgg19(weights=None)
    vgg19.features[0] = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding=(1, 1))
    vgg19.classifier.add_module("my_Linear", nn.Linear(1000, Len_of_charset*Captcha_Len))
    return vgg19


if __name__ == '__main__':
    print(vgg16())