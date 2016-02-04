import json
import sys

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream

from sm_w2v.tokens import CONSUMER_KEY, CONSUMER_SECRET, TOKEN_KEY, TOKEN_SECRET
from sm_w2v.utils import write_obj, clean_sentences
from sm_w2v.config_constants import (
        track,
        r_twt_data_dir, r_red_data_dir, er_log,
        c_twt_data_dir, c_red_data_dir)

class StdOutListener(StreamListener):
    # write streaming data to data.json
    def on_data(self, data):
        twt = json.loads(data)
        write_obj(twt, "sm_w2v/" + r_twt_data_dir)
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
            with open(er_log, "a") as f:
                f.write(e)

def run_clean():
    """
    This function runs the pipeline:
        2) Build and save word2vec models
    """
    print("cleaning...")
    clean_sentences(r_twt_data_dir, c_twt_data_dir)
    clean_sentences(r_red_data_dir, c_red_data_dir)
    print("done cleaning.")

def run_train():
    """
    This function runs the pipeline:
        3) Get word frequency of `related words` and save
    """
    print("training...")
    print("done training...")

if __name__ == '__main__':
    run = sys.argv[1]

    if run == "download":
        run_download()
    elif run == "clean":
        run_clean()
    elif run == "train":
        run_train()
    else:
        print("\n\n Sorry, that wasn't a valid choice. The choices are (download, clean, train)\n")
