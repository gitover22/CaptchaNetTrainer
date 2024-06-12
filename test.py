from models.resnet import *
from models.sample_CNN import *
from tools import trans
from tools.my_dataset import Get_test_Dataloader
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from tools import captcha_info
from models.vgg import *
def test_acc(model_select,model_name):
    model = model_select()
    model.eval()
    model.load_state_dict(torch.load("saved_model/"+model_name+".pkl"))
    # gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fun = nn.MultiLabelSoftMarginLoss().cuda()
    else:
        loss_fun = nn.MultiLabelSoftMarginLoss()

    print("\n start test!!!")

    test_dataloader = Get_test_Dataloader()

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(tqdm(test_dataloader)):
        image = images
        vimage = Variable(image)
        vimage = vimage.to(device)

        # Predict the output and calculate the loss
        predict_label = model(vimage)
        labels = Variable(labels.float())
        labels = labels.to(device)
        loss = loss_fun(predict_label, labels)
        predict_label.argmax(1)
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
        print("predict: %s   true: %s\n"%(predict_label,true_label))
        if predict_label == true_label:
            correct += 1

    accuracy = correct / total
    print('\n The accuracy of the test is: %f %%   loss is: %f ' % (100 * correct / total, loss))
    return accuracy, loss

if __name__ == '__main__':
    file_name = input("Please enter the name of the model you want to test (like:cnn_model):")
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
        test_acc(selected_model, file_name)
    else:
        print("The model name you selected is invalid. Please re-run and enter the correct model name.")
