import numpy as np
from tools import captcha_info

def encode(text):
    """
    transfer text to vector
    @param text: string text, such as:"9BK7H"
    @return: a tensor of 5*(10+26+26) length
    """
    # define a captcha_info.ALL_CHAR_SET_LEN * captcha_info.MAX_CAPTCHA zero vector
    vector = np.zeros(captcha_info.Captcha_Len * captcha_info.Len_of_charset, dtype=float)

    def char2pos(c):
        """
        Maps characters to their corresponding locations
        @param c: char
        @return: pos
        """
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

    for i, c in enumerate(text):
        idx = i * captcha_info.Len_of_charset + char2pos(c)
        vector[idx] = 1.0
    return vector


def decode(vec):
    """
    transfer vec to text
    """
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
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

# test
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
