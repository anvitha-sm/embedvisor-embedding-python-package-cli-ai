import click
import numpy as np
from embedx.core import Embedx

@click.group()
def cli():
    """Embedx CLI for embedding cleaning, analysis, and visualization."""
    pass

@cli.command()
@click.option("--input", "-i", required=True, type=str, help="Path to text file with one text entry per line.")
@click.option("--model", "-m", default="all-MiniLM-L6-v2", type=str, help="Name of the sentence-transformers model to use.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the generated embeddings (.npy).")
def embed(input, model, output):
    with open(input, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    Embedx.generate_embeddings(texts, model_name=model, output_path=output)

def load_embeddings(path):
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Unknown file format: must be .npy or .csv")

@cli.group()
def stats():
    """Commands for computing statistics on embeddings."""
    pass

@stats.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
def basic(input):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.basic_stats()

@cli.group()
def clean():
    """Commands for cleaning embeddings."""
    pass

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--threshold", "-t", default=0.99, type=float, help="Threshold for duplicate removal.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def remove_duplicates(input, threshold, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.remove_duplicates(threshold=threshold)
    np.save(output, embedx.embeddings)

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--contamination", "-c", default=0.01, type=float, help="Contamination level for outlier removal.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def remove_outliers(input, contamination, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.remove_outliers(contamination=contamination)
    np.save(output, embedx.embeddings)

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def center(input, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.center()
    np.save(output, embedx.embeddings)

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--method", "-m", default="l2", type=click.Choice(["l1", "l2"]), help="Normalization method.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def normalize(input, method, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.normalize(method=method)
    np.save(output, embedx.embeddings)

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--n_components", "-n", default=50, type=int, help="Number of components for whitening.")
@click.option("--whiten", "-w", is_flag=True, help="Whether to apply whitening.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def whiten(input, n_components, whiten, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.whiten(n_components=n_components, whiten=whiten, transform=True, plot_variance=True)
    np.save(output, embedx.embeddings)

@clean.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--threshold", "-t", default=0.001, type=float, help="Threshold for low variance removal.")
@click.option("--output", "-o", required=True, type=str, help="Path to save the cleaned embeddings (.npy).")
def remove_low_variance(input, threshold, output):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.remove_low_variance(threshold=threshold)
    np.save(output, embedx.embeddings)

@cli.group()
def visualize():
    """Commands for visualizing embeddings."""
    pass

@visualize.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--dim", "-d", default=2, type=int, help="Dimensionality for visualization (2 or 3).")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def umap(input, dim, save_path=None):
    from embedx.visualization import visualize_umap
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.visualize_umap(dim=dim, save_path=save_path)

@visualize.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--dim", "-d", default=2, type=int, help="Dimensionality for visualization (2 or 3).")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def tsne(input, dim, save_path=None):
    from embedx.visualization import visualize_tsne
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.visualize_tsne(dim=dim, save_path=save_path)

@visualize.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--threshold", "-t", default=0.95, type=float, help="Threshold for neighbor visualization.")
@click.option("--n_neighbors", "-n", default=10, type=int, help="Number of neighbors to visualize.")
@click.option("--save_path", "-s", type=str, help="Path to save the neighbor visualization image.")
def neighbors(input, threshold, n_neighbors, save_path=None):
    from embedx.visualization import visualize_neighbors
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.visualize_neighbors(threshold=threshold, n_neighbors=n_neighbors, save_path=save_path)

@visualize.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--save_path", "-s", type=str, help="Path to save the norm histogram image.")
def norms(input, save_path=None):
    from embedx.visualization import visualize_norms
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    embedx.visualize_norm_histogram(save_path=save_path)

@visualize.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--cluster_method", "-c", default="kmeans", type=click.Choice(["kmeans", "dbscan", "hdbscan", "gmm", "spectral"]), help="Clustering method.")
@click.option("--viz_method", "-v", default="umap", type=click.Choice(["umap", "tsne"]), help="Visualization method.")
@click.option("--dim", "-d", default=2, type=int, help="Dimensionality for visualization (2 or 3).")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def clusters(input, cluster_method, viz_method, dim, save_path=None):
    from embedx.cluster import cluster_embeddings
    from embedx.visualization import visualize_clusters
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings, verbose=True)
    
    labels = cluster_embeddings(embedx.embeddings, method=cluster_method, verbose=embedx.verbose)
    visualize_clusters(embedx.embeddings, embedx.n_samples, labels, method=viz_method, dim=dim, save_path=save_path)

@cli.group()
def cluster():
    """Commands for clustering embeddings."""
    pass    

@cluster.command()
@click.option('--input', required=True)
@click.option('--method', default='kmeans')
@click.option('--n_clusters', type=int, default=10)
@click.option('--eps', type=float, default=0.5)
@click.option('--min_samples', type=int, default=5)
@click.option('--min_cluster_size', type=int, default=5)
@click.option('--n_components', type=int, default=10)
@click.option('--output', required=True, help="Path to save cluster labels (CSV).")
def embeddings(input, method, output, n_clusters, eps, min_samples, min_cluster_size, n_components):
    embeddings = load_embeddings(input)
    embedx = Embedx(embeddings)

    kwargs = {}
    if method == "kmeans":
        kwargs["n_clusters"] = n_clusters
    elif method == "dbscan":
        kwargs["eps"] = eps
        kwargs["min_samples"] = min_samples
    elif method == "hdbscan":
        kwargs["min_cluster_size"] = min_cluster_size
    elif method == "spectral":
        kwargs["n_clusters"] = n_clusters
    elif method == "gmm":
        kwargs["n_components"] = n_components
    labels = embedx.cluster_embeddings(method=method, **kwargs)
    np.savetxt(output, labels, delimiter=",", fmt="%d")

@cli.group()
def advanced():
    """Commands for advanced embedding analysis."""
    pass

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--second", "-s", required=True, type=str, help="Path to second embeddings file (.npy or .csv) for comparison.")
@click.option("--save_path", "-o", type=str, help="Path to save the visualization image.")
def compare(input, second, save_path=None):
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    embeddings_2 = load_embeddings(second)
    
    embedx = Embedx(embeddings, verbose=True)
    avg_similarity = embedx.compare_models(embeddings_2, plot=True, save_path=save_path)
    print(f"Average similarity: {avg_similarity}")

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--top_n", "-n", default=5, type=int, help="Number of top items to consider for semantic coverage.")
@click.option("--labels", "-l", required=True, type=str, help="Path to cluster labels (.csv)")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def semantic_coverage(input, top_n, labels, save_path=None):
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    label = np.loadtxt(labels, dtype=int)

    embedx = Embedx(embeddings, labels=label, verbose=True)
    embedx.semantic_coverage(top_n=top_n, plot=True, save_path=save_path)

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--labels", "-l", required=True, type=str, help="Path to cluster labels (.csv)")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def intracluster_variance(input, labels, save_path=None):
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    label = np.loadtxt(labels, dtype=int)
    
    embedx = Embedx(embeddings, labels=label, verbose=True)
    embedx.intracluster_variance(plot=True, save_path=save_path)

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--labels", "-l", required=True, type=str, help="Path to cluster labels (.csv)")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def intercluster_distance(input, labels, save_path=None):
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    label = np.loadtxt(labels, dtype=int)
    
    embedx = Embedx(embeddings, labels=label, verbose=True)
    embedx.intercluster_distance(plot=True, save_path=save_path)

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--threshold", "-t", default=0.95, type=float, help="Threshold for density calculation.")
@click.option("--n_neighbors", "-n", default=10, type=int, help="Number of neighbors for density calculation.")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def density(input, threshold, n_neighbors, save_path=None): 
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    
    embedx = Embedx(embeddings, verbose=True)
    embedx.density(threshold=threshold, n_neighbors=n_neighbors, plot=True, save_path=save_path)     

@advanced.command()
@click.option("--input", "-i", required=True, type=str, help="Path to input embeddings file (.npy or .csv).")
@click.option("--window_size", "-w", default=10, type=int, help="Window size for decay calculation.")
@click.option("--save_path", "-s", type=str, help="Path to save the visualization image.")
def decay(input, window_size, save_path=None):
    from embedx.core import Embedx
    embeddings = load_embeddings(input)
    
    embedx = Embedx(embeddings, verbose=True)
    embedx.decay_over_time(window_size=window_size, plot=True, save_path=save_path)  

if __name__ == "__main__":
    cli()
