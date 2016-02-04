import unittest
import json

from sm_w2v.utils import (
        get_weeknum_frm_obj,
        write_obj,
        preprocess_txt,
        clean_sentences
        )
from sm_w2v.config_constants import (
        r_twt_data_dir,
        r_red_data_dir
        )

class TestUtils(unittest.TestCase):
    """
    Unit test everything in `sm_w2v.utils`
    """

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
        indir_twt = r_twt_data_dir + "test/"
        indir_red = r_red_data_dir + "test/"

        clean_sentences(indir_twt, "temp/")
        l = open("temp/twt.test").readline()
        self.assertEqual(l, "got flu\n")

        clean_sentences(indir_red, "temp/")
        l = open("temp/red.test").readline()
        self.assertEqual(l, "pale color always style\n")

class TestIntegration(unittest.TestCase):
    """
    Integration test the pipeline:
        1) download (twitter)
        2) clean (twitter and reddit)
        3) train w2v model and make frequency tables (twitter and reddit)
    """

    # TODO
    # def test_clean(self):

    # TODO
    # def test_train(self):

if __name__ == "__main__":
    unittest.main()
