import json
import string
import os
from math import sqrt
import datetime
from os import listdir

# general purpose
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import gensim
from nltk.corpus import stopwords

# important data source for CDC weekly data:
# http://www.cdc.gov/mmwr/mmwr_nd/nd_data_tables.html
class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)

def obj_generator(data_dir):
    '''
    load all json objs (one per line) from file as a
    generator (cuz big data, small memory)
    '''
    filenames = [f for f in listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for fname in filenames:
        with open(data_dir + fname) as f:
            for line in f:
                obj = json.loads(line)
                yield obj

twts = MakeIter(obj_generator, data_dir="sm_w2v/r_twt_data/")

def len_iterable(iterable):
    """
    Get the length of an iterable. Yes, this is
    (time) inefficent for large iterables
    """
    return sum(1 for e in iterable)

# this is the stop words collection
eng_stopwords = stopwords.words('english')
eng_stopwords.append('httpst')
eng_stopwords = set(eng_stopwords)

def preprocess_txt(txt):
    '''
    Takes raw text and returns a list of cleaned sentences
    (list of list of word-strings)
    '''

    clean_sentences = []
    raw_sentences = txt.split('.')
    for sentence in raw_sentences:
        sentence = ''.join(
                [char for char in sentence if char not in string.punctuation]
                )
        sentence = sentence.lower()
        sentence = [word for word in sentence.split() if word not in eng_stopwords]
        clean_sentences.append(sentence)
    return clean_sentences

def clean_sentences(indir, outdir):
    '''
    Takes a directory of raw objects `dirname` (tweets or reddit comments)
    and converts them to clean sentences in `out_dir`
    '''
    for infile in os.listdir(indir):
        if not os.path.isdir(indir + infile):
            f_out = open(outdir + infile.split(".")[0] + ".txt", "w")
            for l in open(indir + infile):
                obj = json.loads(l)
                if 'text' in obj.keys():
                    txt = obj['text']
                else:
                    txt = obj['body']
                clean_sentences = preprocess_txt(txt)
                for sentence in clean_sentences:
                    f_out.write(" ".join(sentence) + " ")
                f_out.write("\n")

def get_weeknum_frm_obj(obj):
    """
    Takes an object and adds a new field (or overwrites)
    `weeknum` which is the iso calendar weeknumber of the object.

    Returns a formated YYYYWW string
    """
    if 'created_at' in obj.keys():
        date_str = obj['created_at']
        dt = datetime.datetime.strptime(date_str, "%a %b %d %H:%M:%S +%f %Y")
    elif 'created_utc' in obj.keys():
        dt = datetime.datetime.fromtimestamp(int(obj['created_utc']))
    year, week = dt.isocalendar()[0], dt.isocalendar()[1]
    return "%d%02d" % (year, week)

def write_obj(obj, outdir):
    """
    Takes an object and an output directory, and writes the object to
    a file in the outdirectory that is determined by the object's
    `weeknum` field.
    """
    # add weeknum
    ywfmt = get_weeknum_frm_obj(obj)
    obj['weeknum'] = ywfmt

    # write twt to raw tweet dir
    with open(outdir + ywfmt + ".json", "a") as f:
        f.write(json.dumps(obj) + "\n")

# delete this after running once
# This has been tested
def temp_reformat(indir, outdir):
    for filename in os.listdir(indir):
        if not os.path.isdir(indir + filename):
            for l in open(indir + filename):
                obj = json.loads(l)
                write_obj(obj, outdir)

def gen_cln_sent(data_dir):
    '''
    Takes a data directory containing the clean sentences and returns a
    generator of (YYYYWW, sentences)
    '''
    for filename in os.listdir(data_dir):
        for l in open(data_dir + filename):
            yield l.split(" ")

# make iterators of clean sentences
cleaned_sentences_twt = MakeIter(gen_cln_sent, data_dir="sm_w2v/c_twt_data/")
cleaned_sentences_red = MakeIter(gen_cln_sent, data_dir="sm_w2v/c_twt_data/")

def make_model(cleaned_sentences, fname, **kwargs):
    """
    Make a word2vec `model`
    """
    model = Word2Vec(sentences=cleaned_sentences, **kwargs)
    model.save(fname)


# Do word frequency count
def flatten_list(list_of_lists):
    out_list = []
    for l in list_of_lists:
        for word in l:
            yield word

def get_word_freq(cleaned_sentences):
    """
    Given an iterator over cleaned sentences,
    Returns (dictionary with word counts,
    a word list, count list sorted by count descending)
    """
    flat_words = flatten_list(cleaned_sentences)
    word_dict = dict()
    for word in flat_words:
        if word in word_dict.keys():
            word_dict[word] = word_dict[word] + 1
        else:
            word_dict[word] = 1
    words = []
    counts = []
    for w in sorted(word_dict, key=word_dict.get, reverse=True):
        words.append(w)
        counts.append(word_dict[w])

    return word_dict, words, counts

# A generator of tweets with coordinates
coord_twts = (twt for twt in twts if twt['coordinates'] != None)


# now we'll make a line plot of :
# (counts of tweets containing 'related word' divided by total tweets for that week)
def get_weeknum_from_filename(filename):
    return filename.split(".")[0]

def count_related_words_normalized(rel_wds, data_dir, out_fname):
    """
    Given a set of related words, and a clean data directory,
    Make a pandas dataframe that represents a frequency table.
    """
    related_words = [wd[0] for wd in rel_wds]
    filenames = [f for f in listdir(data_dir) if os.path.isfile(data_dir + f)]
    week_nums = sorted([get_weeknum_from_filename(name) for name in filenames])
    total_tweets_per_week = dict()
    count_word_week = pd.DataFrame(
            data=np.zeros((len(week_nums), len(related_words))),
            index=week_nums,
            columns=related_words)

    for fname in filenames:
        week_num = get_weeknum_from_filename(fname)
        with open(data_dir + fname) as f:
            for line in f:
                # update the tweet count (per week) for normalization purposes
                if week_num in total_tweets_per_week:
                    total_tweets_per_week[week_num] +=1
                else:
                    total_tweets_per_week[week_num] = 1

                # update related word occurence count
                for wrd in related_words:
                    if wrd in line:
                        count_word_week[wrd][week_num] += 1
    # here is the normalization
    for i, week_num in enumerate(week_nums):
        count_word_week.iloc[i,:] = (count_word_week.iloc[i,:] /
            total_tweets_per_week[week_num])

    count_word_week.to_csv(out_fname)


