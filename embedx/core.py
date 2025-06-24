import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

class Embedx:
    def __init__(self, embeddings: np.ndarray, verbose: bool = True):
        self.embeddings = embeddings
        self.n_samples, self.n_dim = embeddings.shape
        self.verbose = verbose

    def basic_stats(self):
        norms = np.linalg.norm(self.embeddings, axis=1)
        if self.verbose:
            print(f"Embedding shape: {self.embeddings.shape}")
            print(f"Mean norm: {norms.mean():.4f}")
            print(f"Std of norms: {norms.std():.4f}")
        return {
            "shape": self.embeddings.shape,
            "mean_norm": norms.mean(),
            "std_norm": norms.std()
        }
    
    def find_duplicates(self, threshold=0.99, neighbors=10):
        nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine")
        nn.fit(self.embeddings)
        distances, indices = nn.kneighbors(self.embeddings)

        duplicates = []
        for i, (dist_row, index_row) in enumerate(zip(distances, indices)):
            for dist, index in zip(dist_row[1:], index_row[1:]):
                if (1 - dist) >= threshold:
                    if i < index:
                        duplicates.append((i, index, 1 - dist))
       
        if self.verbose:
            print(f"Found {len(duplicates)} near-duplicates")
        return duplicates
    
    def remove_duplicates(self, threshold=0.99, neighbors=10):
        duplicates = self.find_duplicates(threshold=threshold, neighbors=neighbors)
        removing = {index for (_, index, _) in duplicates}
        if self.verbose:
            print(f"Removing {len(removing)} near-duplicates")
        self._remove_indices(removing)

    def find_outliers(self, contamination=0.01):
        iso = IsolationForest(contamination=contamination)
        preds = iso.fit_predict(self.embeddings)
        outliers = np.where(preds == -1)[0]
        if self.verbose:
            print(f"Found {len(outliers)} outliers")
        return outliers
    
    def remove_outliers(self, contamination=0.01):
        outliers = self.find_outliers(contamination=contamination)
        if self.verbose:
            print(f"Removing {len(outliers)} outliers")
        self._remove_indices(outliers)

    def _remove_indices(self, removing):
        if (len(removing) == 0):
            if self.verbose:
                print("Zero embeddings removed.")
            return
        mask = np.ones(self.n_samples, dtype=bool)
        mask[list(removing)] = False
        self.embeddings = self.embeddings[mask]
        self.n_samples = self.embeddings.shape[0]
        if self.verbose:
            print(f"{len(removing)} embeddings have been removed.")
        return self.basic_stats()
    
    def visualize_umap(self, dim=2, labels=None, save_path=None):
        from .visualization import visualize_umap
        visualize_umap(self.embeddings, self.n_samples, dim=dim, labels=labels, save_path=save_path)

    def visualize_tsne(self, dim=2, labels=None, save_path=None):
        from .visualization import visualize_tsne
        visualize_tsne(self.embeddings, self.n_samples, dim=dim, labels=labels, save_path=save_path)

    def visualize_neighbors(self, threshold=0.95, n_neighbors=10, save_path=None):
        from .visualization import visualize_neighbors
        visualize_neighbors(self.embeddings, threshold=threshold, n_neighbors=n_neighbors, save_path=save_path)

    def visualize_norm_histogram(self, save_path=None):
        from .visualization import visualize_norms
        visualize_norms(self.embeddings, save_path=save_path)

    def center(self):
        mean = np.mean(self.embeddings, axis=0, keepdims = True)
        self.embeddings -= mean
        if self.verbose:
            print("Centered embeddings at mean")
    
    def normalize(self, method="l2"):
        if method == "l2":
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims = True)
        elif method == "l1":
            norms = np.sum(np.abs(self.embeddings), axis=1, keepdims = True)
        else:
            raise ValueError(f"Cannot normalize with unknown method: {method}")
        
        norms[norms == 0] = 1
        self.embeddings = self.embeddings/norms
        if self.verbose:
            print(f"{method} normalization applied to embeddings.")

    def whiten(self, n_components=None, whiten=True, transform=True, plot_variance=True):
        if n_components is None or n_components > self.n_dim:
            n_components = self.n_dim
            print(f"Setting n_components to default {self.n_dim}")
        pca = PCA(n_components=n_components, whiten=whiten, random_state=3)
        if transform:
            self.embeddings = pca.fit_transform(self.embeddings)
            self.n_dim = self.embeddings.shape[1]
            print(f"Post-PCA dim = {self.n_dim}")
        else:
            pca.fit(self.embeddings)

        if plot_variance:
            explained = np.cumsum(pca.explained_variance_ratio_)
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
            plt.title("PCA Total Variance Explained")
            plt.xlabel("Number of Components")
            plt.ylabel("Total Variance Explained")
            plt.grid()
            plt.show()

    def variance_plot(self, n_components=None):
        self.whiten(n_components, whiten=False, transform=False, plot_variance=True)

    def remove_low_variance(self, threshold=0.001):
        variance = np.var(self.embeddings, axis=0)
        to_keep = variance > threshold
        self.embeddings = self.embeddings[:, to_keep]
        self.n_dim = self.embeddings.shape[1]
        if self.verbose:
            print(f"Removed {np.sum(variance <= threshold)} low-variance components. New dim: {self.n_dim}")

    def save_embeddings(self, path, format="npy"):
        if format == "npy":
            np.save(path, self.embeddings)
        elif format == "csv":
            np.savetxt(path, self.embeddings, delimiter=",")
        else:
            raise ValueError(f"Cannot save to unknown format: {format}")

        if self.verbose:
            print(f"Saved embeddings to {path} as .{format}")