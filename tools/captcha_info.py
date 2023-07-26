# 设置验证码相关信息
import os

# 生成验证码长度
Captcha_Len = 5

# 设置生成图片尺寸
Pic_Height = 80
Pic_Width = 200

# 可识别字符集合
Char_Set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
    , 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
            '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 记录长度
Len_of_charset = len(Char_Set)

# 设置路径
Train_Dataset = 'dataset\\train'
Test_Dataset = 'dataset\\test'

# 测试
if __name__ == '__main__':
    print(Len_of_charset)
    print(os.path.sep)
