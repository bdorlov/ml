from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_moons, make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, AffinityPropagation


X_moons, y_moons = make_moons(n_samples=200, noise=0.1)
X_blobs, y_blobs = make_blobs(n_samples=200, centers=3)


Z_moons = linkage(X_moons, 'ward')
Z_blobs = linkage(X_blobs, 'ward')
Z_moons_s = linkage(X_moons, 'single')

def show_dendrogram():
    plt.figure(figsize=(9, 6))
    plt.title("ward")
    #dendrogram(Z_blobs)
    #dendrogram(Z_moons)
    dendrogram(Z_moons_s)
    plt.show()

def cluster_ac_blobs():
    clustering_blobs = AgglomerativeClustering(n_clusters=4)
    y_blobs = clustering_blobs.fit_predict(X_blobs)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)

def cluster_ac_moons():
    clustering_moons = AgglomerativeClustering(n_clusters=2)
    y_moons = clustering_moons.fit_predict(X_moons)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)

def cluster_ac_moons_b():
    clustering_moons = AgglomerativeClustering(n_clusters=2, linkage="single")
    y_moons = clustering_moons.fit_predict(X_moons)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)

def cluster_dbscan_moons():
    clustering_moons = DBSCAN(eps=0.2, min_samples=2)
    y_moons = clustering_moons.fit_predict(X_moons)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)
    print(y_moons)

def cluster_dbscan_blobs():
    clustering_blobs = DBSCAN(eps=1, min_samples=3)
    y_blobs = clustering_blobs.fit_predict(X_blobs)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)
    print(y_blobs)

def cluster_ap_moons():
    clustering_moons = AffinityPropagation(affinity='euclidean', convergence_iter=5, damping=0.9, preference=-10.0)
    y_moons = clustering_moons.fit_predict(X_moons)
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)
    print(y_moons)

def cluster_ap_blobs():
    clustering_blobs = AffinityPropagation(affinity='euclidean', convergence_iter=5, damping=0.9, preference=-10.0)
    y_blobs = clustering_blobs.fit_predict(X_blobs)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)
    print(y_blobs)


if __name__ == '__main__':
    #cluster_ap_moons()
    cluster_ap_blobs()
    plt.show()
