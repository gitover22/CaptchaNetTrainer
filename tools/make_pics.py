# 生成训练数据集和测试数据集
import random
import time
from captcha.image import ImageCaptcha  # pip install captcha
from captcha_info import *
from PIL import Image


def random_chars():
    """
    :return:返回长度为Captcha_Len的字符串
    """
    captcha_text = []
    for _ in range(Captcha_Len):
        rc = random.choice(Char_Set)
        captcha_text.append(rc)
    return ''.join(captcha_text)
def generate_pic(num,path):
    """
    生成图片
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for _ in range(num):
        now = str(random.randint(10001,99999))
        text = random_chars()
        image = Image.open(ImageCaptcha(width=Pic_Width,height=Pic_Height).generate(text))
        file_name = text+'_'+now+'.png'
        image.save(path+os.path.sep+file_name)
        print("第%d张已保存,文件名为%s"%(_+1,file_name))
if __name__ == '__main__':
    number = input("请输入生成图片的数量（整数）:")
    path = "..\\dataset\\"
    son_path = input("请输入要生成图片的类型（train或test）:")
    path+=son_path
    generate_pic(int(number),path)