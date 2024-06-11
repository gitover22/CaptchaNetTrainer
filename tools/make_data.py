import random
import time
from captcha.image import ImageCaptcha
from captcha_info import *
from PIL import Image

# captcha generator

def random_chars():
    """
    :return: return a string of length Captcha_Len
    """
    captcha_text = []
    for _ in range(Captcha_Len):
        rc = random.choice(Char_Set)
        captcha_text.append(rc)
    return ''.join(captcha_text)
def generate_pic(num,path):
    if not os.path.exists(path):
        os.makedirs(path)
    for _ in range(num):
        now = str(random.randint(10001,99999))
        text = random_chars()
        image = Image.open(ImageCaptcha(width=Pic_Width,height=Pic_Height).generate(text))
        file_name = text+'_'+now+'.png'
        image.save(path+os.path.sep+file_name)
        print("number %d saved,file name:%s"%(_+1,file_name))
if __name__ == '__main__':
    number = input("input number (int):")
    path = "C:\\Projects\\python_pro\\captcha_recognition\\dataset\\"
    son_path = input("type (train or test):")
    path+=son_path
    generate_pic(int(number),path)