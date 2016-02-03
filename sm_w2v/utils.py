import json
import string
import os
from math import sqrt
import datetime
from os import listdir
from os.path import isfile, join


# general purpose
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA

# plotting toolkits
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
import seaborn as sb


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
    filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for fname in filenames:
        with open(data_dir + fname) as f:
            for line in f:
                obj = json.loads(line)
                yield obj

twts = MakeIter(obj_generator, data_dir='raw_twt_data/')



def len_iterable(iterable):
    """
    Get the length of an iterable. Yes, this is
    (time) inefficent for large iterables
    """
    return sum(1 for e in i)

def preprocess_txt(txt):
    '''
    Takes raw text and returns a list of cleaned sentences
    (list of list of word-strings)
    '''
    clean_sentences = []
    raw_sentences = txt.split('.')
    for sentence in raw_sentences:
        sentence = ''.join([char for char in sentence if char not in string.punctuation])
        sentence = sentence.lower()
        sentence = [word for word in sentence.split() if word not in eng_stopwords]
        clean_sentences.append(sentence)
    return clean_sentences

def clean_sentences(indir, outdir):
    '''
    Takes a directory of raw objects `dirname` (tweets or reddit comments)
    and converts them to clean sentences in `out_dir`
    '''
    for infile in os.listdir(indir)
        fname = filename.split("/")[1]
        f_out = open(outdir + "/" +fname, "w")
        for l in open(infile):
            txt = json.loads(l)['txt']
            clean_sentences = preprocess_txt(txt)
            for sentence in clean_sentences:
                f_out.write(" ".join(sentence) + "\n")


def get_weeknum_frm_obj(obj):
    """
    Takes an object and adds a new field (or overwrites)
    `weeknum` which is the iso calendar weeknumber of the object.

    Returns a formated YYYYWW string
    """
    date_str = obj['created_at']
    dt = datetime.datetime.strptime(date_str, "%a %b %d %H:%M:%S +%f %Y")
    ywfmt, week = dt.isocalendar()[0], dt.isocalendar()[1]
    return "%d%02d" % (year, week)

def write_obj(obj, outdir):
    """
    Takes an object and an output directory, and writes the object to
    a file in the outdirectory that is determined by the object's
    `weeknum` field.
    """
    # add weeknum
    ywfmt = get_weeknum_frmtwt(twt)
    twt['weeknum'] = ywfmt

    # write twt to raw tweet dir
    with open(outdir + ywfmt + ".json", "a") as f:
        f.write(json.dumps(twt) + "\n")

# delete this after running once
def temp_reformat():
    for filename in os.listdir("r_twt_data"):
        for l in open(filename):
            obj = json.loads(l)
            write_obj(obj, "r_twt_data")


# this is the stop words collection
eng_stopwords = stopwords.words('english')

def preprocess_twt(twt, stopwords):
    '''
    Takes a raw sentence (string), and a list of stop words (list of strings),
    and returns a cleaned sentence (list of strings)
    '''
    clean_sentences = []
    raw_sentences = twt['text'].split('.')
    for sentence in raw_sentences:
        sentence = ''.join([char for char in sentence if char not in string.punctuation])
        sentence = sentence.lower()
        stopwords.append("httpst")
        sentence = [word for word in sentence.split() if word not in stopwords]
        clean_sentences.append(sentence)
    return clean_sentences

def clean_sentences(data_dir):
    '''
    Takes a data directory containing the clean sentences and returns a
    generator of (YYYYWW, sentences)
    '''
    for filename in os.listdir(data_dir):
        for l in open(filename):
            yield l.split(" ")

# make iterators of clean sentences
cleaned_sentences_twt = MakeIter(clean_sentences, cl_twt_data_dir)
cleaned_sentences_red = MakeIter(clean_sentences, cl_red_data_dir)


def make_model(cleaned_sentences, word_type, **kwargs):
    """
    Make a word2vec `model`
    """

    if word_type == "word":
        sentences = cleaned_sentences
    elif word_type == "phrase":
        bigram_transformed = gensim.models.Phrases(sentences)
        sentences = bigram_transformed[sentences]

    model = Word2Vec(sentences=sentences, **kwargs)
    return model

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

# do map plot
def plot_map(twts, title='default title'):
    """
    Given an iterable of tweets, make a dot map over North America.
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = Basemap(projection='merc',
        resolution = 'l',
        llcrnrlon=-136.0, llcrnrlat=24.0,
        urcrnrlon=-67.0, urcrnrlat=60.0,
        ax=ax)

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.fillcontinents(color = 'coral', alpha=0.5)
    m.drawmapboundary()

    lons = [twt['coordinates']['coordinates'][0] for twt in twts]
    lats = [twt['coordinates']['coordinates'][1] for twt in twts]
    x,y = m(lons, lats)

    m.plot(x, y, 'bo', markersize=5)
    plt.title(title)
    plt.show()


# now we'll make a line plot of :
# (counts of tweets containing 'related word' divided by total tweets for that week)
def get_weeknum_from_filename(filename):
    return filename.split(".")[0]

def count_related_words_normalized(rel_wds, data_dir):
    """
    Given a set of related words, and a clean data directory,
    Make a pandas dataframe that represents a frequency table.
    """
    related_words = [wd[0] for wd in rel_wds]
    filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    week_nums = sorted([get_weeknum_from_filename(name) for name in filenames])
    total_tweets_per_week = dict()
    count_word_week = pd.DataFrame(data=np.zeros((len(week_nums), len(related_words))),
                                  index=week_nums,
                                  columns=related_words)

    for fname in filenames:
        week_num = get_weeknum_from_filename(fname)
        with open(data_dir + fname) as f:
            for line in f:

                twt = json.loads(line)

                # update the tweet count (per week) for normalization purposes
                if week_num in total_tweets_per_week:
                    total_tweets_per_week[week_num] +=1
                else:
                    total_tweets_per_week[week_num] = 1

                # update related word occurence count
                for wrd in related_words:
                    if wrd in twt['text']:
                        count_word_week[wrd][week_num] += 1
    # here is the normalization
    for i, week_num in enumerate(week_nums):
        count_word_week.iloc[i,:] = count_word_week.iloc[i,:] / total_tweets_per_week[week_num]

    return count_word_week


def make_heatmap_w2vrelated(model, rel_wds):
    """
    Given a model (from word2vec) and a list of related words,
    make a square heatmap using the cosine similarity between the given words
    """
    n = len(rel_wds)
    names = [wd[0] for wd in rel_wds]
    data_mat = np.zeros((n,n))
    for i, word1 in enumerate(names):
        for j, word2 in enumerate(names):
            data_mat[i,j] = model.similarity(word1, word2)
            if i == j:
                data_mat[i,j] = 0

    df = pd.DataFrame(data=data_mat,
                     columns=names,
                     index=names)
    sb.clustermap(df, linewidths=.5,)

def scikit_pca(model, cluster="kmeans"):
    """
    Given a word2vec model and a cluster (choice of "kmeans" or "spectral")
    Make a plot of all word-vectors in the model.
    """
    # the word2vec vectors data matrix
    keys = list(model.vocab.keys())
    num_words_in_vocab = len(keys)
    size_of_vecs = len(model[keys[0]])

    # X is the data matrix
    X = np.zeros((num_words_in_vocab, size_of_vecs))
    for i, key in enumerate(keys):
        X[i,] = model[key]

    labels = [0] * num_words_in_vocab

    if cluster == "kmeans":
        k_means = KMeans(n_clusters=8)
        labels = k_means.fit_predict(X)

    elif cluster == "spectral":
        sp_clust = SpectralClustering()
        labels = sp_clust.fit_predict(X)

    # Standardize
    X_std = StandardScaler().fit_transform(X)

    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)

    # Plot the data
    plt.scatter(X_transf[:,0], X_transf[:,1], c=labels)
    plt.title('PCA via scikit-learn (using SVD)')
    plt.show()

    return X, sklearn_pca.explained_variance_ratio_


def make_histogram(X):
    """
    Given a numpy matrix, plot a histogram
    """
    hist, bins = np.histogram(X, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()



