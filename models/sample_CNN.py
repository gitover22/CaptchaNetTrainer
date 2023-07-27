import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tools import captcha_info


class my_CNN(nn.Module):

    def __init__(self):
        super(my_CNN, self).__init__()
        # 层级序列
        self.layers1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 第一层 输入通道数为1，输出通道数为32，卷积核大小为3，padding大小为1,stride为1
            nn.BatchNorm2d(32),  # 批量归一化层
            # nn.Dropout(0.5),  # 以0.5的概率随机丢弃神经元，防止过拟合
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 第二层 输入通道数为32，输出通道数为32，卷积核大小为3，padding大小为1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 第三层 最大池化层，池化核大小为2, 池化步长默认等于池化核大小
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第四层 输入通道数为32，输出通道数为64，卷积核大小为3，padding大小为1
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),  # 以0.5的概率随机丢弃神经元，防止过拟合
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第五层 输入通道数为64，输出通道数为64，卷积核大小为3，padding大小为1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 第六层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 第七层,输入通道数为64，输出通道数为128，卷积核大小为3，padding大小为1
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),  # 以0.5的概率随机丢弃神经元，防止过拟合   4_20对4_12的改进
            nn.ReLU(),
            nn.MaxPool2d(2)  # 第八层
        )
        # 全连接层
        self.layers2 = nn.Sequential(
            nn.Linear(128 * 10 * 25, 1024),  # 输入层有7*20*128个神经元，输出层有1024个神经元
            nn.ReLU(),
            nn.Dropout(0.5)  # 使用dropout技术，防止过拟合
        )
        # 全连接层
        self.layers3 = nn.Sequential(
            # 输入层有1024个神经元，输出层有5*62个神经元
            nn.Linear(1024, captcha_info.Captcha_Len * captcha_info.Len_of_charset),
        )

    # 前向传播函数
    def forward(self, input):
        out = self.layers1(input)
        out = out.view(out.size(0), -1)  # 展开
        out = self.layers2(out)
        out = self.layers3(out)
        return out


if __name__ == "__main__":
    my_model = my_CNN()
    # print(my_model)
    # 利用torch产生输入数据 ，BCHW
    input = torch.ones((64, 1, 80, 200))
    out = my_model(input)

    # 可视化操作
    writer = SummaryWriter("./logs")
    writer.add_graph(my_model, input)
    writer.close()
    print(out.shape)
    torch.set_printoptions(edgeitems=100000, threshold=100000)
    print(out)
