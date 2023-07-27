# 实现神经网络的训练

import argparse
import time
from tools.my_dataset import *
from tools.my_dataset import Get_train_Dataloader
from test import *
from torch.autograd import Variable
from models.vgg import *
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


def train_model(model_creater,saved_name):
    # 初始化一个CNN网络模型
    model = model_creater()
    print("正常加载网络模型")

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

        torch.save(model.state_dict(), "saved_model/"+saved_name+".pkl")
        print("epoch:{}    trainloss:{}".format((epoch + 1), loss))
        time.sleep(2)

    # 再次保存模型参数
    torch.save(model.state_dict(),  "saved_model/"+saved_name+".pkl")
    print("------------------------------训练测试完成，模型已存储------------------------------")


if __name__ == '__main__':
    file_name = input("请输入存储本次训练模型的文件名：(like:cnn_model)")
    model_names = {
        "my_CNN": my_CNN,
        "vgg11": vgg11,
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19
    }
    model_name = input("请选择您要训练的模型(可选择：my_CNN,vgg11，vgg13，vgg16，vgg19)：")
    if model_name in model_names:
        selected_model = model_names[model_name]
        train_model(selected_model,file_name)
    else:
        print("选择的模型名称无效，请重新运行并输入正确的模型名称。")
