import re
from nltk.corpus import stopwords
from .constants import EMOTICONS_EMO, EMOJI_UNICODE


STOPWORDS = set(stopwords.words('english'))


def remove_punct(text):
    """Custom function to remove the punctuations"""
    text_ = []
    for char in text:
        w = re.sub(r'([^\w\s]|_)', '', char)
        text_.append(w)
    return ''.join(text_)


def remove_stopwords(text):
    """Custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def remove_emoji(string):
    """
    Custom function to remove emojis
    
    References
    ----------
    1. https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def convert_emojis(text):
    for emot in EMOJI_UNICODE:
        text = re.sub(
            r'('+emot+')', 
            "_".join(EMOJI_UNICODE[emot].replace(",","").replace(":","").split()), 
            text
        )
    return text


def remove_emoticons(text):
    """Custom function to remove emoticons"""
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_EMO) + u')')
    return emoticon_pattern.sub(r'', text)


def convert_emoticons(text):
    """Custom function to convert emoticons to string"""
    for emot in EMOTICONS_EMO:
        text = re.sub(
            u'('+emot+')', 
            "_".join(EMOTICONS_EMO[emot].replace(",","").split()), 
            text
        )
    return text


def remove_urls(text): 
    """Custom function to remove URLs"""
    regex_str = (
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
        r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
        r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    )
    url_pattern = re.compile(regex_str) 
    return url_pattern.sub(r'', text)


def remove_html(text):
    """Custom function to remove html tags"""
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)