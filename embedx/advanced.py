import matplotlib.pyplot as plt
import numpy as np

def compare_models(embeddings_a, embeddings_b, plot=True, save_path=None):
    from sklearn.metrics.pairwise import cosine_similarity
    assert embeddings_a.shape == embeddings_b.shape, "Embeddings must have same shape"
    cosine_sim = cosine_similarity(embeddings_a, embeddings_b)
    avg_similarity = np.mean(cosine_sim)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(cosine_sim.flatten(), bins=50, color='orchid')
        plt.title('Cosine Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.axvline(avg_similarity, color='purple', linestyle='dashed', linewidth=1)
        plt.text(avg_similarity + 0.01, 5, f'Avg: {avg_similarity:.2f}', color='purple')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return avg_similarity

def semantic_coverage(embeddings, labels, top_n=10, plot=True, save_path=None):
    from sklearn.metrics.pairwise import cosine_distances
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    coverage_dict = {}
    for label in unique_labels:
        cluster_embeddings = embeddings[labels == label]
        if len(cluster_embeddings) > 0:
            distances = cosine_distances(cluster_embeddings)
            avg_distance = np.mean(distances)
            coverage_dict[label] = avg_distance
        else:
            coverage_dict[label] = 0.0

    sorted_coverage = sorted(coverage_dict.items(), key=lambda x: x[1], reverse=True)
    top_coverage = sorted_coverage[:top_n]

    if plot:
        plt.figure(figsize=(10, 6))
        labels, values = zip(*top_coverage)
        plt.bar(labels, values, color='orchid')
        plt.title('Top Semantic Coverage by Cluster')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Average Distance')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return {label: coverage for label, coverage in top_coverage}

def intracluster_variance(embeddings, labels, plot=True, save_path=None):
    variance_dict = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_embeddings = embeddings[labels == label]
        if len(cluster_embeddings) > 1:
            variance = np.var(cluster_embeddings, axis=0)
            variance_dict[label] = np.mean(variance)
        else:
            variance_dict[label] = 0.0

    if plot:
        plt.figure(figsize=(10, 6))
        labels, variances = zip(*variance_dict.items())
        plt.bar(labels, variances, color='orchid')
        plt.title('Intracluster Variance by Cluster')
        plt.xlabel('Cluster Labels')
        plt.ylabel('Mean Variance')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return variance_dict

def intercluster_distance(embeddings, labels, plot=True, save_path=None):
    from sklearn.metrics.pairwise import cosine_distances
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        cluster_embeddings = embeddings[labels == label]
        if len(cluster_embeddings) > 0:
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
    centroids = np.array(centroids)
    distances = cosine_distances(centroids)
    inter_distances = {}
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                inter_distances[(label_i, label_j)] = distances[i, j] 

    if plot:
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(distances, xticklabels=unique_labels, yticklabels=unique_labels,
                    cmap='OrRd', annot=True, fmt=".2f")
        plt.title('Intercluster Distance Matrix')
        plt.xlabel('Cluster')
        plt.ylabel('Cluster')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return inter_distances

def density(embeddings, threshold=0.95, n_neighbors=10, plot=True, save_path=None):
    try:
        import faiss
        FAISS_AVAILABLE = True
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        FAISS_AVAILABLE = False

    if FAISS_AVAILABLE:
        emb = embeddings.astype(np.float32)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(emb, n_neighbors + 1)
        similarities = 1 - distances[:, 1:]
        density_count = np.sum(similarities >= threshold, axis=1)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        similarities = 1 - distances[:, 1:] 
        density_count = np.sum(similarities >= threshold, axis=1)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(density_count, bins=20, color='orchid')
        plt.title(f'Embedding Density Histogram (Threshold: {threshold})')
        plt.xlabel('Number of Neighbors Above Threshold')
        plt.ylabel('Count')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return similarities, density_count

def decay_over_time(timestamps, embeddings, window_size=10, plot=True, save_path=None):
    from sklearn.metrics.pairwise import cosine_distances
    order = np.argsort(timestamps)
    if timestamps is None:
        raise ValueError("Timestamps must be provided for decay analysis")
    if len(timestamps) != len(embeddings):
        raise ValueError("Timestamps and embeddings must have the same length")
    timestamps = timestamps[order]
    embeddings = embeddings[order]

    mean_reference = np.mean(embeddings[:window_size], axis=0, keepdims=True)
    decay_scores = []
    for i in range(window_size, len(embeddings)):
        current_embedding = embeddings[i:i+1]
        distance = cosine_distances(current_embedding, mean_reference)
        decay_scores.append(distance[0][0])
        mean_reference = np.mean(embeddings[i-window_size+1:i+1], axis=0, keepdims=True)  

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps[window_size:], decay_scores, marker='o', color='orchid')
        plt.title('Embedding Decay Over Time')
        plt.xlabel('Timestamps')
        plt.ylabel('Cosine Distance from Mean Reference')
        plt.grid()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()  
    return np.array(decay_scores), timestamps[window_size:]




   