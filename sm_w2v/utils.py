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

    clean = clean.lower()
    word_list = clean.split()
    words = " ".join(word_list)
    return  words

def clean_tweet(twt_obj, i):
    cln_twt = dict()
    cln_twt['index'] = i
    cln_twt['c_text'] = clean_text(twt_obj['text'])
    cln_twt['tags'] = ['user--' + twt_obj['user']['name']] + \
        ['#' + hashtag['text'].lower() for hashtag in twt_obj['entities']['hashtags']]
    cln_twt['weeknum'] = twt_obj['weeknum']
    #coordinates HERE
    if twt_obj['coordinates']:
        cln_twt['coordinates'] = twt_obj['coordinates']['coordinates']
    return cln_twt

