import os

Captcha_Len = 4
# size
Pic_Height = 80
Pic_Width = 200

# Char_Set = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
Char_Set = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Char_Set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
#     , 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
#             '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Char_Set = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
#             '1', '2', '3', '4', '5', '6', '7', '8', '9']

Len_of_charset = len(Char_Set) #  10 or 26 or 36 or 62

Train_Dataset = 'dataset\\train'
Test_Dataset = 'dataset\\test'

# test
if __name__ == '__main__':
    print(Len_of_charset)
    print(os.path.sep)
