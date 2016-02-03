# download_twitter_disease_data

Before running, you need two additional (untracked) files:

```
tokens.py # contains twitter access tokens
data.json # contains twitter stream data to be appended to
```

To run, set up virtualenv with `python3.4` and install the requirements:

```
virtualenv -p /usr/bin/python3.4 venv
source venv/bin/activate
pip install -r requirements.txt
```

(actually as of this writing, the version of `tweepy` on PYPI has a bug, so you have to install the latest `tweepy` from source (github)):

```
git clone https://github.com/tweepy/tweepy.git
cd tweepy
python setup.py install
```

Then run `python main.py`
