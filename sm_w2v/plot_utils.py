
# plotting toolkits
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
import seaborn as sb

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



