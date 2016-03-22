import json
import sys
import datetime

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream

from sm_w2v.utils import clean_text, clean_tweet, relevant_twt
from sm_w2v.tokens import CONSUMER_KEY, CONSUMER_SECRET, TOKEN_KEY, TOKEN_SECRET

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
        sdate = twt['created_at']
        pydate = datetime.datetime.strptime(sdate, '%a %b %d %H:%M:%S %z %Y')
        yyyyww = "%d%02d" % (pydate.isocalendar()[0], pydate.isocalendar()[1])
        twt['weeknum'] = yyyyww
        with open('data/r_twitter.json', 'a') as f:
            f.write(json.dumps(twt) + '\n')
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
    with open('data/r_twitter.json') as f_in, open('data/c_twitter.json', 'w') as f_out:
        for i, l in enumerate(f_in):
            try:
                twt = json.loads(l)
            except:
                break

            if i % 2 and relevant_twt(twt):
                cln_twt = clean_tweet(twt)
                f_out.write(json.dumps(cln_twt) + '\n')
    print("done cleaning.")

if __name__ == '__main__':
    run = sys.argv[1]

    if run == "download":
        run_download()
    elif run == "clean":
        run_clean()
    else:
        print("\n\n Sorry, that wasn't a valid choice. The choices are (download, clean, train, wdfrq)\n")
