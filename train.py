import argparse
import time
import torch
import tqdm
from models.sample_CNN import my_CNN
from tools.my_dataset import *
from tools.my_dataset import Get_train_Dataloader
from test import *
from torch.autograd import Variable
from models.vgg import *
from models.resnet import *
def parse_option():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser('Arguments for training')
    parser.add_argument('--epochs', type=int, default=25, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    opt = parser.parse_args()
    return opt


def count_img_nums():
    """
    Used to count the number of pictures involved in the training
    """
    path = "./dataset/train"
    count = 0
    for _ in os.listdir(path): 
        count = count + 1
    print("The number of pictures involved in the training : %d\n" % count)


def train_model(model_creater,saved_name):
    model = model_creater()
    print("load model success")
    # CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("working device:", device)
    opt = parse_option()

    train_dataloader = Get_train_Dataloader()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)

    my_loss_fun = nn.MultiLabelSoftMarginLoss()
    my_loss = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        my_loss_fun = my_loss_fun.cuda()
    # train model 
    for epoch in range(opt.epochs):
        batch_loss = 0
        count = 0
        for i, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = Variable(images)
            labels = Variable(labels.float())
            images = images.to(device)
            labels = labels.to(device)
            predict_labels = model(images)
            # loss
            # loss = my_loss_fun(predict_labels, labels)
            loss = my_loss(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = batch_loss + loss.cpu().item()
            count = count + 1

        torch.save(model.state_dict(), "saved_model/"+saved_name+".pkl")
        print("epoch:{}    trainloss:{}".format((epoch + 1), batch_loss/count))
        time.sleep(2)

    # saved
    torch.save(model.state_dict(),  "saved_model/"+saved_name+".pkl")
    print("------------------------------The training is complete and the model is stored------------------------------")


if __name__ == '__main__':
    file_name = input("Please enter the file name to store this training model: (like:cnn_model)")
    model_names = {
        "diyCNN": my_CNN,
        "vgg11": vgg11,
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }
    model_name = input("Select the type of model you want to train(pools: diyCNN,vggN,resnetN):")
    if model_name in model_names:
        selected_model = model_names[model_name]
        train_model(selected_model,file_name)
    else:
        print("The model name you selected is invalid. Please re-run and enter the correct model name.")
