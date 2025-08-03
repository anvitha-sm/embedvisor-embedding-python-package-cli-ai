# Embedvisor

An all-in-one embeddings package in Python for data preprocessing: cleaning, visualization, clustering, etc. CLI support and an LLM-powered web app to recommend data preprocessing steps customized to the user's dataset and model requirements/goals. 

## Tech Stack
Python, Streamlit, Groq, OneDrive
FAISS, sklearn, sentence-transformers

## Project Links
https://embedvisor.streamlit.app

https://pypi.org/project/embedvisor/

https://github.com/anvithasm/embedvisor

## Functions
#### Initialization
- `Embedx(embeddings: np.ndarray, verbose: bool = True, timestamps=None, labels=None)`
  - Create an Embedx instance for managing embeddings and related metadata.

#### Metadata
- `set_labels(labels)`
  - Assign labels for the embeddings.
- `set_timestamps(timestamps)`
  - Assign timestamps associated with embeddings.

#### Embedding Management
- `generate_embeddings(texts, model_name="all-MiniLM-L6-v2", output_path="embeddings.npy")`
  - Generate embeddings from a list of texts using a SentenceTransformer model and save to file.
- `load_embeddings(path)`
  - Load embeddings from `.npy` or `.csv` files.
- `save_embeddings(path, format="npy")`
  - Save current embeddings to `.npy` or `.csv`.

#### Basic Info
- `get_dims()`
  - Returns `(n_samples, n_dim)` of the embeddings array.
- `basic_stats()`
  - Compute and return basic statistics (shape, mean norm, std norm) of embeddings.

#### Preprocessing
- `find_duplicates(threshold=0.99, neighbors=10)`
  - Find pairs of embeddings that exceed similarity threshold (cosine similarity).
- `remove_duplicates(threshold=0.99, neighbors=10)`
  - Remove near-duplicate embeddings based on similarity threshold.
- `find_outliers(contamination=0.01)`
  - Detect outliers using IsolationForest algorithm.
- `remove_outliers(contamination=0.01)`
  - Remove detected outliers from embeddings.
- `center()`
  - Center embeddings by subtracting the mean vector.
- `normalize(method="l2")`
  - Normalize embeddings using L1 or L2 norm.
- `whiten(n_components=None, whiten=True, transform=True, plot_variance=True, save_path=None)`
  - Perform PCA whitening and optionally transform embeddings and plot explained variance.
- `variance_plot(n_components=None)`
  - Plot variance explained by PCA components (no whitening or transform).
- `remove_low_variance(threshold=0.001)`
  - Remove PCA components with variance below threshold.

#### Visualization
- `visualize_umap(dim=2, save_path=None, display=False)`
  - Visualize embeddings using UMAP in 2D or 3D.
- `visualize_tsne(dim=2, save_path=None, display=False)`
  - Visualize embeddings using t-SNE in 2D or 3D.
- `visualize_neighbors(threshold=0.95, n_neighbors=10, save_path=None, display=False)`
  - Visualize neighbor similarity above threshold.
- `visualize_norm_histogram(save_path=None, display=False)`
  - Plot histogram of embedding norms.

#### Clustering
- `cluster_embeddings(method="kmeans", **kwargs)`
  - Cluster embeddings with method (`kmeans`, `dbscan`, `hdbscan`, etc).
- `cluster_visualize(cluster_method="kmeans", viz_method="umap", dim=2, save_path=None, display=False, **kwargs)`
  - Cluster embeddings using specified method and visualize clusters.

#### Advanced Analytics
- `intracluster_variance(plot=True, save_path=None, display=False)`
  - Calculate and plot variance within clusters.
- `intercluster_distance(plot=True, save_path=None, display=False)`
  - Calculate and plot distance between clusters.
- `compare_models(embeddings_2, plot=True, save_path=None, display=False)`
  - Compare this embedding set to another and plot similarity.
- `semantic_coverage(plot=True, save_path=None, display=False)`
  - Measure and plot semantic coverage of embeddings.
- `density(threshold, n_neighbors, plot=True, save_path=None, display=False)`
  - Calculate density and similarity, then plot.
- `decay_over_time(window_size=10, plot=True, save_path=None, display=False)`
  - Analyze embedding changes over time based on timestamps.

## Embedvisor: The Package
Run
```
pip install embedvisor
```
to import embedvisor into a Python file or Jupyter Notebook.

## Embedvisor: The Command-Line Interface
All functions through this CLI take the same parameters as in the function list, in addition to the --input/-i option to indicate the file location for the user's input embeddings and the --output/-o option to indicate the file location for the resulting embeddings.

## Embedvisor: The Web App
The web app features OneDrive integration and a Groq-powered processing + visualization recommendation assistant. Users can either upload raw data or embeddings. If uploading raw data, users can either log in via Microsoft account to directly upload their Excel spreadsheet from OneDrive, or can upload an Excel or csv file from their local file system. Then, users can generate embeddings within the app via SentenceTransformers and select which column, if any, is the label column. If uploading embeddings, users can upload their features and labels as distinct embedding files. 

All functions listed above can be done directly through the app with customizable parameters. Users can select which columns will undergo which functions through a multi-select box. Plots are displayed directly in the app, but can also be downloaded into the user's local file system or into OneDrive. Users can save their embeddings as npy or csv files in their local file system or OneDrive.

For recommendations on what processing or visualization steps to take, the Groq-powered AI assistant analyzes characteristics about the user's input data as well as their goal for the analysis (anomaly detection, clustering, classification, etc.) and selects the best parameters and order of specific operations to yield the best results.
