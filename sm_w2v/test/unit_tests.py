import unittest
import json
import os
import shutil
import itertools

from sm_w2v.utils import (
    get_weeknum_frm_obj,
    write_obj,
    preprocess_txt,
    clean_sentences,
    cleaned_sentences_twt,
    cleaned_sentences_red,
    make_model,
    count_related_words_normalized
    )

from gensim.models import Word2Vec

class TestUtils(unittest.TestCase):
    """
    Unit test everything in `sm_w2v.utils`
    """

    def setUp(self):
        if not os.path.exists('temp/'):
                os.makedirs('temp/')

    def test_get_weeknum_frm_obj(self):
        # tweet example
        obj = dict(created_at="Sun Jan 01 01:01:01 +000001 2001")
        weeknum = get_weeknum_frm_obj(obj)
        self.assertEqual(weeknum, "200101")

        # reddit example
        obj = dict(created_utc="1193875152")
        weeknum = get_weeknum_frm_obj(obj)
        self.assertEqual(weeknum, "200744")

    def test_write_obj(self):
        # tweet example
        obj = dict(created_at="Sun Jan 01 01:01:01 +000001 2001")
        write_obj(obj, "temp/")
        f = open("temp/200101.json")
        obj = json.loads(f.readline())
        self.assertEqual(obj["weeknum"], "200101")

        # reddit example
        obj = dict(created_utc="1193875152")
        write_obj(obj, "temp/")
        f = open("temp/200744.json")
        obj = json.loads(f.readline())
        self.assertEqual(obj["weeknum"], "200744")

    def test_preprocess_text(self):
        txt = "Sentence a the one. Grapefruit is."

        cln_txt = preprocess_txt(txt)
        self.assertEqual(cln_txt, [["sentence", "one"], ["grapefruit"], []])

    def test_clean_sentences(self):
        indir_twt = "sm_w2v/r_twt_data/test/"
        indir_red = "sm_w2v/r_red_data/test/"

        clean_sentences(indir_twt, "temp/")
        l = open("temp/twt.test").readline()
        self.assertEqual(l, "got flu \n")

        clean_sentences(indir_red, "temp/")
        l = open("temp/red.test").readline()
        print(l)
        self.assertTrue("compression ratio theoretically" in l)

    def test_train_model(self):
        twt_model_fname="temp/twt_word.model"

        make_model(itertools.islice(cleaned_sentences_twt, 0, 1000),
                   twt_model_fname,
                   size=100, # dimension of word vecs
                   window=5, # context size
                   min_count=100, #words repeated less than this are discarded
                   workers=6, # number of threads
                   seed=1
                  )
        model = Word2Vec.load(twt_model_fname)

    # this requires a model and a query word
    def test_word_freq_table(self):
        model = Word2Vec.load("temp/twt_word.model")
        rel_wds = model.most_similar(positive=['hiv'], topn=10)

        count_related_words_normalized(rel_wds,
               "sm_w2v/c_twt_data/",
               "temp/twt_hiv_wdfreq.csv")

if __name__ == "__main__":
    unittest.main()
