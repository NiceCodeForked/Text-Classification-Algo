import re


def remove_punct(text):
    text_ = []
    for char in text:
        w = re.sub(r'([^\w\s]|_)', '', char)
        text_.append(w)
    return ''.join(text_)