import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

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
            print("{len(removing)} embeddings have been removed.")
        return self.basic_stats()
