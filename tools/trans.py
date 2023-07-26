# 实现转化，vector<->text
import numpy as np
from tools import captcha_info


def encode(text):
    """
    将输入的验证码文本 text 转换成一个大小为 5 * (26+26+10)的向量 vector
    @param text: 字符串文本，such as:"9BK7H"
    @return:一个（10+26+26）*5维度的张量，前62位代表第一个字符，依次类推
    """
    # 定义一个大小为 captcha_info.ALL_CHAR_SET_LEN * captcha_info.MAX_CAPTCHA 的全 0 向量 vector，用于存储标记结果
    vector = np.zeros(captcha_info.Captcha_Len * captcha_info.Len_of_charset, dtype=float)

    def char2pos(c):
        """
        将字符映射到其对应的位置，用于在向量中标记出该字符
        @param c:
        @return:
        """
        # 将下划线字符 '_' 映射到 62 的位置
        if c == '_':
            k = 62
            return k
        # 将数字字符 '0' ~ '9' 映射到 0 ~ 9 的位置(0的ascii是48)
        k = ord(c) - 48
        # 将大写字母字符 'A' ~ 'Z' 映射到 10 ~ 35 的位置(A的ASCII码为65)
        if k > 9:
            k = ord(c) - 65 + 10
            # 将小写字母字符 'a' ~ 'z' 映射到 36 ~ 61 的位置(a的ascii是97)
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

    # 遍历验证码文本的每个字符，并在 vector 中标记出该字符所对应的位置
    for i, c in enumerate(text):
        # 计算该字符在 vector 中的位置
        idx = i * captcha_info.Len_of_charset + char2pos(c)
        # 将该位置上的值设置为 1，表示该字符在该位置上出现
        vector[idx] = 1.0
    return vector


def decode(vec):
    """
    将输入的向量 vec 转换回原始的验证码文本 text
    """
    # 使用 nonzero() 函数获取 vec 中非零元素的索引位置，得到每个字符在向量中的位置
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        print("i: %d   c: %d"%(i,c))
        char_idx = c % captcha_info.Len_of_charset
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


if __name__ == '__main__':
    e = encode("6BK7H")
    print(type(e))
    print(e)
    print(decode(e))
    # for i in range(10):
    #     print(e)
    # print(e)
    # sor=decode(e)
    # print(sor)
