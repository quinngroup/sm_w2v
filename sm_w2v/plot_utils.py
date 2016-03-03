import numpy as np
import pandas as pd

# plotting toolkits
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import seaborn as sb

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# do map plot
def plot_map_twts(twts, title='default title'):
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

def make_data_matrix(model):
    # the word2vec vectors data matrix
    keys = list(model.vocab.keys())
    num_words_in_vocab = len(keys)
    size_of_vecs = len(model[keys[0]])

    # X is the data matrix
    X = np.zeros((num_words_in_vocab, size_of_vecs))

    return X, keys


# need to fix with new 'scatter'
def scikit_pca(model, rel_wds, plot_lims, title, cluster="kmeans"):
    """
    Given a word2vec model and a cluster (choice of "kmeans" or "spectral")
    Make a plot of all word-vectors in the model.
    """
    X, keys = make_data_matrix(model)

    for i, key in enumerate(keys):
        X[i,] = model[key]

    if cluster == "kmeans":
        k_means = KMeans(n_clusters=8)
        labels = k_means.fit_predict(X)

    elif cluster == "spectral":
        sp_clust = SpectralClustering()
        labels = sp_clust.fit_predict(X)

    # PCA
    X_std = StandardScaler().fit_transform(X)
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)

    scatter_plot(X_transf[:,0], X_transf[:,1],  rel_wds, labels, title, keys, plot_lims)

    return sklearn_pca.explained_variance_ratio_

def scatter_plot(x, y, alpha_high, alpha_low, clust_labels, text_annotations, down_samp_rate, title, rand_seed=None, plot_lims=None):

    # set up the high alpha labeled plot
    txt_labels = []
    x_high_alpha = []
    y_high_alpha = []
    labels_high_alpha = []
    text_high_alpha = []
    for i in range(len(x)):
        if(text_annotations[i] != ""):
            x_high_alpha.append(x[i])
            y_high_alpha.append(y[i])
            labels_high_alpha.append(clust_labels[i])
            text_high_alpha.append(text_annotations[i])

    # do down sample
    if(rand_seed):
        np.random.seed(rand_seed)
    take = np.random.choice(list(range(len(x))), int(len(x)*down_samp_rate), replace=False)
    x = np.array(x)[take]
    y = np.array(y)[take]
    clust_labels = np.array(clust_labels)[take]

    # Plot all the data with low alpha (in the background)
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=alpha_low, c=clust_labels, cmap=plt.get_cmap("nipy_spectral"))

    # Plot data with high alpha
    ax.scatter(x_high_alpha, y_high_alpha, c=labels_high_alpha,
            alpha=alpha_high, cmap=plt.get_cmap("nipy_spectral"))
    for i, txt in enumerate(text_high_alpha):
        ax.annotate(txt, (x_high_alpha[i], y_high_alpha[i]))
    if(plot_lims):
        plt.axis(plot_lims)
    plt.savefig(title + ".pdf", format="pdf")
    plt.show()

# need to fix with new 'scatter'
def make_tsne_plot(model, rel_wds, plot_lims, title):

    dim = 30
    X, keys = make_data_matrix(model)

    # first we actually do PCA to reduce the
    # dimensionality to make tSNE easier to calculate
    X_std = StandardScaler().fit_transform(X)
    sklearn_pca = PCA(n_components=2)
    X = sklearn_pca.fit_transform(X_std)[:,:dim]

    # do downsample
    k = 5000
    sample = []
    important_words = []
    r_wds = [word[0] for word in rel_wds]
    for i, key in enumerate(keys):
        if key in r_wds:
            sample.append(i)
    sample = np.concatenate((np.array(sample),
                np.random.choice(len(keys), k-10, replace = False),
             ))
    X = X[sample,:]
    keys = [keys[i] for i in sample]



    # Do tSNE
    tsne = TSNE(n_components=2, random_state=0, metric="cosine")
    X_transf = tsne.fit_transform(X)

    k_means = KMeans(n_clusters=8)
    labels = k_means.fit_predict(X_transf)

    scatter_plot(X_transf[:,0], X_transf[:,1],  rel_wds, labels, title, keys, plot_lims)


