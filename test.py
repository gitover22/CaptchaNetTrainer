# 实现神经网络的测试
from models.sample_CNN import *
from tools import trans
from tools.my_dataset import Get_test_Dataloader
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from tools import captcha_info
from models.vgg import *
def test_acc(model_select,model_name):
    # 创建CNN模型并加载保存的模型
    model = model_select()
    model.eval()
    model.load_state_dict(torch.load("saved_model/"+model_name+".pkl"))

    # 判断是否支持GPU，如果支持则将模型和损失函数转移到GPU上
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 判断是否支持GPU
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fun = nn.MultiLabelSoftMarginLoss().cuda()
    else:
        loss_fun = nn.MultiLabelSoftMarginLoss()

    # 输出开始测试信息
    print("\n开始测试!!!")

    # 获取测试集数据加载器
    test_dataloader = Get_test_Dataloader()

    # 初始化正确分类和总数
    correct = 0
    total = 0

    # 对测试集进行迭代
    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        # 将图像转化为可变变量并移到GPU上（如果支持）
        image = images
        vimage = Variable(image)
        vimage = vimage.to(device)

        # 预测输出并计算损失值
        predict_label = model(vimage)
        labels = Variable(labels.float())
        labels = labels.to(device)
        loss = loss_fun(predict_label, labels)
        predict_label.argmax(1)
        # 将预测值转化为字符并计算准确率
        c0 = captcha_info.Char_Set[
            np.argmax(predict_label[0, 0:captcha_info.Len_of_charset].data.cpu().numpy())]
        c1 = captcha_info.Char_Set[
            np.argmax(
                predict_label[0, captcha_info.Len_of_charset:2 * captcha_info.Len_of_charset].data.cpu().numpy())]
        c2 = captcha_info.Char_Set[
            np.argmax(predict_label[0,
                      2 * captcha_info.Len_of_charset:3 * captcha_info.Len_of_charset].data.cpu().numpy())]
        c3 = captcha_info.Char_Set[
            np.argmax(predict_label[0,
                      3 * captcha_info.Len_of_charset:4 * captcha_info.Len_of_charset].data.cpu().numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = trans.decode(labels.cpu().numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1

    # 计算并输出准确率和损失值
    accuracy = correct / total
    print('\n当前 %d 张图片测试的准确率为: %f %%   loss为: %f ' % (total, 100 * correct / total, loss))
    return accuracy, loss

if __name__ == '__main__':
    file_name = input("请输入本次要测试的模型文件名(like:cnn_model)：")
    model_names = {
        "my_CNN": my_CNN,
        "vgg11": vgg11,
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19
    }
    model_name = input("请选择您要测试的模型类型(like:my_CNN,vgg11，vgg13，vgg16，vgg19)：")
    if model_name in model_names:
        selected_model = model_names[model_name]
        test_acc(selected_model, file_name)
    else:
        print("选择的模型名称无效，请重新运行并输入正确的模型名称。")
