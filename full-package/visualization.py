import umap
import numpy as np
import pandas as pd
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    FAISS_AVAILABLE = False
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def convert_labels_to_numeric(labels):
    labels = np.array(labels)
    labels = labels.flatten()
    if labels.dtype.kind not in {'i', 'f'}: 
        labels = labels.astype(str) 
        le = LabelEncoder()
        return le.fit_transform(labels)
    else:
        return labels

def visualize_umap(embeddings, n_samples, dim=2, labels=None, save_path=None):
    if (dim == 2):
        return visualize_umap_2d(embeddings, n_samples, labels, save_path)
    elif (dim == 3):
        return visualize_umap_3d(embeddings, n_samples, labels, save_path)
    else:
        print("UMAPs must be in 2D or 3D")

def visualize_umap_2d(embeddings, n_samples, labels=None, save_path=None):
    reduce = umap.UMAP(n_components=2, n_neighbors=min(15, n_samples-1), random_state=3)
    embeddings_2d = reduce.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("2D UMAP of Embeddings")
    if labels is not None:
        if pd.api.types.is_numeric_dtype(labels):
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap="Spectral",
                s=5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Label Value')
        else:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=pd.factorize(labels)[0],
                cmap="Spectral",       
                s=5
            )
    else:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5)
    if save_path:
        fig.savefig(save_path)
    return fig

def visualize_umap_3d(embeddings, n_samples, labels=None, save_path=None):
    reduce = umap.UMAP(n_components=3, n_neighbors=min(15, n_samples-1), random_state=3)
    embeddings_3d = reduce.fit_transform(embeddings)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    numeric_labels = convert_labels_to_numeric(labels) if labels is not None else None
    if labels is not None:
        if pd.api.types.is_numeric_dtype(labels):
            scatter = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=labels, cmap="Spectral", s=5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Label Value')
        else:
            scatter = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=pd.factorize(labels)[0], cmap="Spectral", s=5)
    else:   
        scatter = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], s=5)
    ax.set_title("3D UMAP of Embeddings")
    if save_path:
        fig.savefig(save_path)
    return fig

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
        return visualize_tsne_2d(embeddings, perplexity, labels, save_path)
    elif (dim == 3):
        return visualize_tsne_3d(embeddings, perplexity, labels, save_path)
    else:
        print("t-SNEs must be in 2D or 3D")

def visualize_tsne_2d(embeddings, perplexity, labels=None, save_path=None):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_labels = convert_labels_to_numeric(labels) if labels is not None else None
    if labels is not None:
        if pd.api.types.is_numeric_dtype(labels):
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap="Spectral",
                s=5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Label Value')
        else:
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=pd.factorize(labels)[0],
                cmap="Spectral",       
                s=5
            )
    else:
        scatter = ax.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=5)
    ax.set_title("2D t-SNE of Embeddings")

    if save_path:
        fig.savefig(save_path)
    return fig

def visualize_tsne_3d(embeddings, perplexity, labels=None, save_path=None):
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=3)
    embeddings_3d = tsne.fit_transform(embeddings)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    numeric_labels = convert_labels_to_numeric(labels) if labels is not None else None
    if labels is not None:
        if pd.api.types.is_numeric_dtype(labels):
            scatter = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=labels, cmap="Spectral", s=5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Label Value')
        else:
            scatter = ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], c=pd.factorize(labels)[0], cmap="Spectral", s=5)
    else:
        ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2], s=5)
    ax.set_title("3D t-SNE of Embeddings")

    if save_path:
        fig.savefig(save_path)
    return fig

def visualize_clusters(embeddings, n_samples, labels=None, method="umap", dim=2, save_path=None):
    method = method.lower()
    if (method == "umap"):
        return visualize_umap(embeddings, n_samples, dim, labels, save_path)
    elif (method == "tsne"):
        return visualize_tsne(embeddings, n_samples, dim, labels, save_path)

def visualize_neighbors(embeddings, threshold=0.95, n_neighbors=10, save_path=None):
    if FAISS_AVAILABLE:
        emb = embeddings.astype(np.float32)
        faiss.normalize_L2(emb)

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        distances, indices = index.search(emb, k=n_neighbors + 1)
        similarities = 1 - distances[:, 1:]
        num_neighbors_close = np.sum(similarities >= threshold, axis = 1)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        similarities = 1 - distances[:, 1:]
        num_neighbors_close = np.sum(similarities >= threshold, axis = 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(num_neighbors_close, bins=20, color="orchid")
    ax.set_title(f"Histogram of neighbors with similarity >= {threshold}")
    ax.set_xlabel("Number of near neighbors per embedding")
    ax.set_ylabel("Count")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig, similarities, num_neighbors_close

def visualize_norms(embeddings, save_path = None):
    norms = np.linalg.norm(embeddings, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(norms, bins = 50, color="orchid")
    ax.set_title("Embedding Norm Distribution")
    ax.set_xlabel("Norm")
    ax.set_ylabel("Count of Norms")

    if save_path:
        fig.savefig(save_path)
    return fig

