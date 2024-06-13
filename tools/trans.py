import numpy as np
from tools import captcha_info
# import captcha_info

def encode(text):
    """
    Transfer text to vector based on dynamic character set from captcha_info.
    @param text: string text, such as:"9BK7H"
    @return: a tensor of length captcha_info.Captcha_Len * captcha_info.Len_of_charset
    """
    vector = np.zeros(captcha_info.Captcha_Len * captcha_info.Len_of_charset, dtype=float)

    # Creating a character index map based on the current character set
    char2idx = {char: idx for idx, char in enumerate(captcha_info.Char_Set)}

    def char2pos(c):
        """
        Maps characters to their corresponding positions in the character set
        @param c: char
        @return: pos
        """
        if c in char2idx:
            return char2idx[c]
        else:
            raise ValueError(f"Character {c} not found in the defined character set.")

    for i, c in enumerate(text):
        idx = i * captcha_info.Len_of_charset + char2pos(c)
        vector[idx] = 1.0
    return vector

def decode(vec):
    """
    Transfer vec back to text using the character set defined in captcha_info.
    """
    char_pos = vec.nonzero()[0]
    text = []
    for pos in char_pos:
        char_idx = pos % captcha_info.Len_of_charset
        char_code = captcha_info.Char_Set[char_idx]
        text.append(char_code)
    return "".join(text)

if __name__ == '__main__':
    e = encode("6BK7")
    print(type(e))
    print(e)
    # print(decode(e))
    # for i in range(10):
    #     print(e)
    # print(e)
    # sor=decode(e)
    # print(sor)
