import json
import sys

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream

from gensim.models import Word2Vec

from sm_w2v.tokens import CONSUMER_KEY, CONSUMER_SECRET, TOKEN_KEY, TOKEN_SECRET
from sm_w2v.utils import (
        write_obj, clean_sentences, make_model, cleaned_sentences_twt, cleaned_sentences_red,
        count_related_words_normalized
        )


# The list of relevant disease keywords to track
track = ['hiv', 'aids', 'pre-exposure', 'prep', 'prophylaxis',

# AIDS medications from (http://www.aidsmeds.com/articles/DrugChart_10632.shtml)
'Atripla', 'Complera', 'Eviplera', 'Genvoya', 'Stribild', 'Triumeq',
'Aptivus', 'Crixivan', 'Evotaz', 'Invirase', 'Kaletra' 'Aluvia',
'Lexiva', 'Telzir', 'Norvir', 'Prezcobix', 'Rezolsta', 'Prezista',
'Reyataz', 'Viracept', 'Combivir', 'Emtriva', 'Epivir', 'Epzicom',
'Retrovir', 'Trizivir', 'Truvada', 'Videx EC', 'Viread', 'Zerit',
'Ziagen', 'Edurant', 'Intelence', 'Rescriptor', 'Sustiva',
'Viramune XR', 'Fuzeon', 'Selzentry', 'Isentress', 'Tivicay'

# other infection disease keywords (can cross reference with CDC data)
'Babesiosis', 'Chlamydia', 'trachjomatis', 'Coccidioidomycosis',
'Cyptosporidiosis', 'Dengue Fever', 'Dengue',
'Ehrlichia', 'caffeenis', 'Anaplasma','phagocytophilum',
'Ehrliciosis', 'Anaplasmosis', 'Giardiasis', 'Gonorrhea',
'Haemophilus', 'influenzae', 'flu', 'Hepatitis',
'Pneumococal', 'Pneumonia', 'Legionellosis', 'Lyme disease',
'Malaria', 'Meningoccal', 'meningitis', 'giardia',
'legionnaires', 'Mumps', 'Pertussis', 'Rabies', 'Salmonellosis',
'Shiga', 'Shigellosis', 'Rickettsiosis', 'Spotted Fever',
'Syphilis', 'Varicella', 'chicken pox', 'West Nile Virus']


class StdOutListener(StreamListener):
    # write streaming data to data.json
    def on_data(self, data):
        twt = json.loads(data)
        write_obj(twt, "sm_w2v/r_twt_data/")
        return True

def run_download():
    """
    This function starts downloading and saving tweets from twitter
    """
    print("downloading...")
    while True:
        try:
            # open file to write, and authenticate
            l = StdOutListener()
            auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(TOKEN_KEY, TOKEN_SECRET)
            stream = Stream(auth, l, retry_count=10)

            # Use the POST statuses/filter endpoint
            stream.filter(track=track, languages=['en'])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(str(e))

def run_clean():
    """
    This function runs the pipeline:
        2) clean raw data (remove stopwords and punctuation)
    """
    print("cleaning...")
    clean_sentences("sm_w2v/r_twt_data/", "sm_w2v/c_twt_data/")
    #clean_sentences("sm_w2v/r_red_data", "sm_w2v/c_red_data")
    print("done cleaning.")

def run_train():
    """
    This function runs the pipeline:
        3) Train word2vec and save model
    """
    print("training...")
    make_model(cleaned_sentences_twt,
               "sm_w2v/models_freq_tables/twt.model",
               size=100, # dimension of word vecs
               window=5, # context size
               min_count=100, #words repeated less than this are discarded
               workers=5 # number of threads
              )
    make_model(cleaned_sentences_red,
              "sm_w2v/models_freq_tables/red.model",
              size=100, # dimension of word vecs
              window=5, # context size
              min_count=100, #words repeated less than this are discarded
              workers=5 # number of threads
              )
    print("done training.")

def run_wdfrq():
    """
    This function runs the pipeline:
        4) Get word frequency of `related words` and save
    """
    print("running word freq...")

    # twitter
    model_twt = Word2Vec.load("sm_w2v/models_freq_tables/twt.model")
    rel_words = model_twt.most_similar(positive=['hiv'], topn=10)
    count_related_words_normalized(rel_words,
            "sm_w2v/c_twt_data/",
            "sm_w2v/models_freq_tables/twt_hiv_wdfreq.csv")
    rel_words = model_twt.most_similar(positive=['prophylaxis'], topn=10)
    count_related_words_normalized(rel_words,
            "sm_w2v/c_twt_data/",
            "sm_w2v/models_freq_tables/twt_prophylaxis_wdfreq.csv")


    # reddit
    model_red= Word2Vec.load("sm_w2v/models_freq_tables/red.model")
    rel_words = model_red.most_similar(positive=['hiv'], topn=10)
    count_related_words_normalized(rel_words,
            "sm_w2v/c_red_data/",
            "sm_w2v/models_freq_tables/red_hiv_wdfreq.csv")
    rel_words = model_red.most_similar(positive=['prophylaxis'], topn=10)
    count_related_words_normalized(rel_words,
            "sm_w2v/c_red_data/",
            "sm_w2v/models_freq_tables/red_prophylaxis_wdfreq.csv")

    print("done with word freq.")

if __name__ == '__main__':
    run = sys.argv[1]

    if run == "download":
        run_download()
    elif run == "clean":
        run_clean()
    elif run == "train":
        run_train()
    elif run == "wdfrq":
        run_wdfrq()
    else:
        print("\n\n Sorry, that wasn't a valid choice. The choices are (download, clean, train, wdfrq)\n")
