import umap
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_umap(embeddings, n_samples, dim=2, labels=None, save_path=None):
    if (dim == 2):
        visualize_umap_2d(embeddings, n_samples, labels, save_path)
    elif (dim == 3):
        visualize_umap_3d(embeddings, n_samples, labels, save_path)
    else:
        print("UMAPs must be in 2D or 3D")

def visualize_umap_2d(embeddings, n_samples, labels=None, save_path=None):
    reduce = umap.UMAP(n_components=2, n_neighbors=min(15, n_samples-1), random_state=3)
    embeddings_2d = reduce.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="Spectral", s=5)
    else:   
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5)
    plt.title("2D UMAP of Embeddings")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_umap_3d(embeddings, n_samples, labels=None, save_path=None):
    reduce = umap.UMAP(n_components=3, n_neighbors=min(15, n_samples-1), random_state=3)
    embeddings_3d = reduce.fit_transform(embeddings)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if labels is not None:
        ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=labels, cmap="Spectral", s=5)
    else:   
        ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], s=5)
    ax.set_title("3D UMAP of Embeddings")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_tsne(embeddings, n_samples, dim=2, labels=None, save_path=None):
    if n_samples < 100:
        perplexity = 5
    elif n_samples < 500:
        perplexity = 20
    elif n_samples < 2000:
        perplexity = 30
    else:
        perplexity = 50

    if perplexity >= n_samples / 3:
        perplexity = max(5, n_samples // 4)
        
    if (dim == 2):
        visualize_tsne_2d(embeddings, perplexity, labels, save_path)
    elif (dim == 3):
        visualize_tsne_3d(embeddings, perplexity, labels, save_path)
    else:
        print("t-SNEs must be in 2D or 3D")

def visualize_tsne_2d(embeddings, perplexity, labels=None, save_path=None):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=labels, cmap='Spectral', s=5)
    else:
        plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=5)
    plt.title("2D t-SNE of Embeddings")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_tsne_3d(embeddings, perplexity, labels=None, save_path=None):
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=3)
    embeddings_3d = tsne.fit_transform(embeddings)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if labels is not None:
        ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=labels, cmap='Spectral', s=5)
    else:
        ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], s=5)
    ax.set_title("3D t-SNE of Embeddings")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_neighbors(embeddings, threshold=0.95, n_neighbors=10, save_path=None):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    similarities = 1 - distances[:, 1:]
    num_neighbors_close = np.sum(similarities >= threshold, axis = 1)
    plt.figure(figsize=(8, 6))
    plt.hist(num_neighbors_close, bins=20, color="orchid")
    plt.title(f"Histogram of neighbors with similarity >= {threshold}")
    plt.xlabel("Number of near neighbors per embedding")
    plt.ylabel("Count")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_norms(embeddings, save_path = None):
    norms = np.linalg.norm(embeddings, axis=1)
    plt.figure(figsize=(8, 6))
    plt.hist(norms, bins = 50, color="orchid")
    plt.title("Embedding Norm Distribution")
    plt.xlabel("Norm")
    plt.ylabel("Count of Norms")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

