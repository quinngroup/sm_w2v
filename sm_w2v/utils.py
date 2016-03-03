import re

class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)

def clean_text(text):
    clean = re.sub(r'http.*$', '', text)
    regex = re.compile('[^a-zA-Z,\.!?\'#0-9\s\-_]')
    clean = regex.sub('', clean)

    clean = clean.replace('...', '.')
    clean = clean.replace('.', ' . ')
    clean = clean.replace(',', ' , ')
    clean = clean.replace('#', ' # ')
    clean = clean.replace('?', ' ? ')
    clean = clean.replace('!', ' ! ')

    word_list = clean.split()
    words = " ".join(word_list)
    return  words

