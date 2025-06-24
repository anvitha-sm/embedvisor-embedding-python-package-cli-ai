import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
try:
    import hdbscan
    HDBSCAN_AVAIL = True
except ImportError:
    HDBSCAN_AVAIL = False

def cluster_embeddings(embeddings, method="kmeans", verbose=True, **kwargs):
    method = method.lower()
    if method == "kmeans":
        labels, clusters = cluster_kmeans(embeddings, **kwargs)
    elif method == "dbscan":
        labels, clusters = cluster_dbscan(embeddings, **kwargs)
    elif method == "hdbscan":
        if not HDBSCAN_AVAIL:
            raise ImportError("Install hdbscan with 'pip install hdbscan' first")
        labels, clusters = cluster_hdbscan(embeddings, **kwargs)
    elif method == "gmm":
        labels, clusters = cluster_gmm(embeddings, **kwargs)
    elif method == "spectral":
        labels, clusters = cluster_spectral(embeddings, **kwargs)
    else:
        raise ValueError(f"Cannot cluster with unknown method: {method}")
    
    if verbose:
        print(f"Found {clusters} clusters with {method}")
    return labels

def cluster_kmeans(embeddings, n_clusters=10, **kwargs):
    model = KMeans(n_clusters=n_clusters, random_state=3)
    labels = model.fit_predict(embeddings)
    return labels, len(set(labels))

def cluster_dbscan(embeddings, eps=0.5, min_samples=5, **kwargs):
    model = DBSCAN(eps = eps, min_samples = min_samples, metric = "cosine")
    labels = model.fit_predict(embeddings)
    clusters = len(set(labels)) 
    if -1 in labels:
        clusters -= 1
    return labels, clusters

def cluster_hdbscan(embeddings, min_cluster_size=5, **kwargs):
    model = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, metric = "euclidean")
    labels = model.fit_predict(embeddings)
    clusters = len(set(labels)) 
    if -1 in labels:
        clusters -= 1
    return labels, clusters

def cluster_gmm(embeddings, n_components=10, **kwargs):
    model = GaussianMixture(n_components=n_components, random_state=3)
    labels = model.fit_predict(embeddings)
    return labels, len(set(labels))

def cluster_spectral(embeddings, n_clusters=10, **kwargs):
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=3)
    labels = model.fit_predict(embeddings)
    return labels, len(set(labels))
    