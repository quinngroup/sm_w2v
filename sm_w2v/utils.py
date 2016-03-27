import re

class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)

def clean_text(text):
    clean = re.sub(r'http.*$', '', text)
    clean = re.sub(r'[^a-zA-Z,\.!?\'#0-9\s\-_]', '', clean)

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
    cln_twt['text'] = clean_text(twt['text'])
    cln_twt['user_id_str'] = twt['user']['id_str']
    cln_twt['tags'] = [twt['user']['name'] + '-*-' + cln_twt['user_id_str']] + \
        ['#' + hashtag['text'].lower() for hashtag in twt['entities']['hashtags']]
    cln_twt['weeknum'] = twt['weeknum']
    #coordinates HERE
    if twt['coordinates']:
        cln_twt['coordinates'] = twt['coordinates']['coordinates']
    return cln_twt

def relevant_twt(twt):
    key_words_lower = ['hiv', 'aids', 'truvada', 'prophylaxis',
                       'imtesting', 'sexwork', 'gay']
    key_words_w_case = ['PrEP']

    for word in key_words_w_case:
        if word in twt['text']:
            return True

    lcase_txt = twt['text'].lower()

    for word in key_words_lower:
        if word in lcase_txt:
            return True

    return False
