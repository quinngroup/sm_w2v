import unittest

from sm_w2v.plot_utils import scatter_plot
import numpy as np

class TestPlotUtils(unittest.TestCase):
    """
    Unit test everything in `sm_w2v.utils`
    """

    def setUp(self):
        pass

    def test_scatter_plot(self):
        rand_seed=0
        np.random.seed(rand_seed)
        x = np.random.randn(100)
        y = np.random.randn(100)
        alpha_high = 1.0
        alpha_low = 0.4
        clust_labels = [1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20
        text_annotations = [""]*90 + ["odd"]*10
        down_samp_rate = 0.5
        title = "scatter_test"
        plot_lims=[-4,4,-4,4]

        scatter_plot(x, y, alpha_high, alpha_low,
                clust_labels, text_annotations,
                down_samp_rate, title, rand_seed, plot_lims)

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
