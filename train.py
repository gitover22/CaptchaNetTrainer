# 实现神经网络的训练

import argparse
import time
from tools.my_dataset import *
from models.sample_CNN import *
from tools.my_dataset import Get_train_Dataloader
import tqdm
from test import *
from torch.autograd import Variable

'''
captcha_train.py用于训练模型,模型会被保存在model_CNN.pkl文件中
作者：邹国强
'''


def parse_option():
    """
    tool function
    命令行参数解析器，并定义了三个可选的命令行参数，
    分别是 epochs、batch_size 和 learning_rate.
    它们的默认值分别是 20、256 和 0.00007
    @rtype: object
    """
    parser = argparse.ArgumentParser('Arguments for training')
    parser.add_argument('--epochs', type=int, default=25, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    # 解析命令行参数，并将解析结果赋值给变量 opt
    opt = parser.parse_args()
    # print(type(opt))
    return opt


def count_img_nums():
    """
    tool function
    用来统计参与训练的图片数量
    """
    path = "./dataset/train"
    count = 0  # count用来统计./dataset/train下的图片数量
    for _ in os.listdir(path):  # _ 表示的是文件名
        count = count + 1
    # print("参与本次训练的图片共有%d张" % count)


def train_and_test():
    # 初始化一个CNN网络模型
    model = my_CNN()
    print("正常加载CNN网络模型")

    # 检查是否有可用的CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("当前训练设备为:", device)

    # 解析命令行参数
    opt = parse_option()

    # 获取训练数据集
    train_dataloader = Get_train_Dataloader()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

    my_loss_fun = nn.MultiLabelSoftMarginLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        my_loss_fun = my_loss_fun.cuda()
    # 训练模型
    for epoch in range(opt.epochs):
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)
            labels = labels.to(device)
            predict_labels = model(images)
            # 计算损失
            loss = my_loss_fun(predict_labels, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), "saved_model/test_model.pkl")
        print("epoch:{}    trainloss:{}".format((epoch + 1), loss))
        time.sleep(2)

        # 计算模型在测试集上的准确率和损失值
        accuracy, loss2 = test_acc()

        # 自定义阈值
        theta = 0.96
        if accuracy > theta:
            break

    # 再次保存模型参数
    torch.save(model.state_dict(), "saved_model/test_model.pkl")
    print("------------------------------训练测试完成，模型已存储------------------------------")


if __name__ == '__main__':
    # 参与训练的图片数
    count_img_nums()
    # 记录开始时间
    train_and_test()

