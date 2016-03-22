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

def clean_tweet(twt):
    cln_twt = dict()

    cln_twt['id'] = twt['id']
    cln_twt['c_text'] = clean_text(twt['text'])
    cln_twt['user_id_str'] = twt['user']['id_str']
    cln_twt['tags'] = ['user--' + twt['user']['name'] + '--' + cln_twt['user_id_str']] + \
        ['#' + hashtag['text'].lower() for hashtag in twt['entities']['hashtags']]
    cln_twt['weeknum'] = twt['weeknum']
    #coordinates HERE
    if twt['coordinates']:
        cln_twt['coordinates'] = twt['coordinates']['coordinates']
    return cln_twt

def relevant_twt(twt):
    key_words = ['hiv', 'aids', 'truvada', 'prep', 'prophylaxis',
                 'imtesting', 'sexwork', 'gay']
    lcase_txt = twt['text'].lower()
    # if twt['coordinates']:
    #     return True

    for word in key_words:
        if word in lcase_txt:
            return True
    return False
