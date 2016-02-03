import json
import sys

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream

from smw2v.tokens import CONSUMER_KEY, CONSUMER_SECRET, TOKEN_KEY, TOKEN_SECRET
from smw2v.utils import write_obj
from smw2v.config_constants import track, r_twt_data_dir, er_log

class StdOutListener(StreamListener):
    # write streaming data to data.json
    def on_data(self, data):
        twt = json.loads(data)
        write_obj(twt, r_twt_data_dir)
        return True

def run_download():
"""
This function starts downloading and saving tweets from twitter
"""
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

def run_pipeline():
"""
This function runs the pipeline:
    1) Clean and save tweets and reddit comments as clean sentences
    2) Build and save word2vec models
    3) Get word frequency of `related words` and save
"""
    pass

if __name__ == '__main__':
    run = sys.argv[1]

    if run == "download":
        run_download()
    elif run == "pipeline":
        run_pipeline()
