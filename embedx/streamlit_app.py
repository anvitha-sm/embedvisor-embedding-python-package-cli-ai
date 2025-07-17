import streamlit as st
import pandas as pd
import numpy as np
from embedx.core import Embedx
from graph_api import start_device_flow, acquire_token, list_excel_files, download_file

st.set_page_config(page_title="Excipher", layout="wide")
st.title("Excipher: Process and Visualize Excel Data")

def get_all_embeddings(selected_columns, all_embeddings):
    if not selected_columns:
        st.error("Please select at least one column.")
        return None
    arrays = []
    for col in selected_columns:
        arr = all_embeddings.get(col)
        if arr is None:
            st.error(f"No embeddings found for column '{col}'.")
            return None
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    combined = np.hstack(arrays)
    return combined

if "df" not in st.session_state:
    st.session_state.df = None
if "show_picker" not in st.session_state:
    st.session_state.show_picker = False
if "excel_files" not in st.session_state:
    st.session_state.excel_files = []
if "token" not in st.session_state:
    st.session_state.token = None
if "device_flow_message" not in st.session_state:
    st.session_state.device_flow_message = None
if "auth_app" not in st.session_state:
    st.session_state.auth_app = None
if "auth_flow" not in st.session_state:
    st.session_state.auth_flow = None
if "awaiting_auth" not in st.session_state:
    st.session_state.awaiting_auth = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = []

with st.sidebar:
    st.header("Upload File")
    if st.button("Import from OneDrive"):
        if st.session_state.token:
            st.session_state.show_picker = True
            st.session_state.awaiting_auth = False
        else:
            try:
                app, flow, message = start_device_flow()
                st.session_state.auth_app = app
                st.session_state.auth_flow = flow
                st.session_state.device_flow_message = message
                st.session_state.awaiting_auth = True
                st.session_state.show_picker = False 
            except Exception as e:
                st.error(f"Could not start Device Code Flow:\n\n{e}")
    st.markdown("---")
    uploaded_file = st.file_uploader("Or upload Excel/csv file", type=["xlsx", "csv"], key="upload_file")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.session_state.show_picker = False
        st.session_state.awaiting_auth = False 
        st.success("File uploaded!")
    st.markdown("""<div style="height:3px;background-color:#333;margin:20px 0;"></div>""", unsafe_allow_html=True)
    st.header("Upload Precomputed Embeddings")
    uploaded_embeddings = st.file_uploader("Upload embeddings (.npy or .csv)", type=["npy", "csv"], key="upload_embeddings")
    uploaded_labels = st.file_uploader("Upload labels (.csv, optional)", type=["csv"], key="upload_labels")
    if uploaded_embeddings:
        embeddings = Embedx.load_embeddings(uploaded_embeddings)
        st.session_state.embeddings = embeddings
        st.session_state.df = pd.DataFrame(embeddings)  
        st.success(f"Embeddings uploaded! Shape: {embeddings.shape}")

        if uploaded_labels:
            labels = np.loadtxt(uploaded_labels, delimiter=",", dtype=int)
            st.session_state.labels = labels
            st.success(f"Labels uploaded! Found {len(labels)} labels.")
        else:
            st.session_state.labels = None

if st.session_state.df is not None:
    st.success(f"Loaded DataFrame with {len(st.session_state.df)} rows.")
    st.dataframe(st.session_state.df.head())
    embeddings = st.session_state.df.values
else:
    st.info("Connect to OneDrive or upload a file to get started.")
    embeddings = None

if st.session_state.awaiting_auth:
    st.info("Follow these instructions to authorize Excipher on OneDrive:")
    st.code(st.session_state.device_flow_message)

    result = acquire_token(st.session_state.auth_app, st.session_state.auth_flow)
    if "access_token" in result:
        st.session_state.token = result["access_token"]
        st.session_state.excel_files = list_excel_files(st.session_state.token)
        st.session_state.show_picker = True
        st.session_state.awaiting_auth = False
        st.success("Connected to OneDrive!")

if st.session_state.show_picker:
    st.subheader("Import from OneDrive")
    search_term = st.text_input("Search filename")
    files = st.session_state.excel_files.get("value", [])
    filtered_files = [
        f for f in files if search_term.lower() in f["name"].lower()
    ] if search_term else files

    if filtered_files:
        file_options = {f["name"]: f["id"] for f in filtered_files}
        selected_file = st.selectbox("Select file:", list(file_options.keys()))
        if st.button("Upload Selected File"):
            file_id = file_options[selected_file]
            file_data = download_file(st.session_state.token, file_id)
            df = pd.read_excel(file_data)
            st.session_state.df = df
            st.session_state.show_picker = False  
    else:
        st.info("No matching Excel files found.")

st.markdown("""<div style="height:3px;background-color:#333;margin:20px 0;"></div>""", unsafe_allow_html=True)
st.header("EmbedX Functions")
if st.session_state.df is not None:
    all_columns = st.session_state.df.columns.tolist()
    st.session_state.selected_columns = st.multiselect(
        "Select columns to process with each function:",
        all_columns,
        default=all_columns 
    )
    st.session_state.label_column = st.selectbox(
        "Select label column (optional):",
        ["None"] + all_columns
    )

    if st.button("Generate Embeddings"):
        for col in st.session_state.df.columns:
            col_dtype = st.session_state.df[col].dtype
            if col_dtype == "object":
                texts = st.session_state.df[col].dropna().astype(str).tolist()
                embeddings = Embedx.generate_embeddings(texts, model_name="all-MiniLM-L6-v2")
                st.session_state.embeddings[col] = embeddings
            else:
                st.session_state.embeddings[col] = st.session_state.df[col].dropna().values.astype(np.float64).reshape(-1, 1)
        st.success("Embeddings generated!")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("☆ Process ☆")
    st.markdown("---")
    st.markdown("### Basic Stats")
    if st.button("Run Basic Stats"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            result = embedx.basic_stats()
            df_stats = pd.DataFrame({
                "Metric": ["Shape", "Mean Norm", "Std Norm"],
                "Value": [str(result["shape"]),
                        round(result["mean_norm"], 4),
                        round(result["std_norm"], 4)]
            })
            df_stats["Value"] = df_stats["Value"].astype(str)
            st.table(df_stats)

    st.markdown("### Remove Duplicates")
    threshold = st.number_input("Threshold", value=0.99, key="duplicate_threshold")
    neighbors = st.number_input("Neighbors", value=10, key="duplicate_neighbors")
    if st.button("Run Remove Duplicates"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                embedx.remove_duplicates(threshold=threshold, neighbors=neighbors)
        st.success("Duplicates removed!")

    st.markdown("### Remove Outliers")
    contamination = st.number_input("Contamination", value=0.01, key="outlier_contamination")
    if st.button("Run Remove Outliers"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                embedx.remove_outliers(contamination=contamination)
        st.success("Outliers removed!")

    st.markdown("### Center Embeddings")
    if st.button("Center Embeddings"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                embedx.center()
        st.success("Centered!")

    st.markdown("### Normalize")
    method = st.selectbox("Method", ["l1", "l2"], key="normalize_method")
    if st.button("Run Normalize"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                embedx.normalize(method=method)
        st.success("Normalized!")

    st.markdown("### Principal Component Analysis (PCA)")
    n_components = st.number_input("Components", value=50, key="pca_components")
    whiten_flag = st.checkbox("Apply Whitening", key="pca_whiten", value=True)
    transform_flag = st.checkbox("Apply Transform", value=True, key="pca_transform")
    if st.button("Run PCA"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            fig = embedx.whiten(n_components=n_components, whiten=whiten_flag, transform=transform_flag, plot_variance=True)
            st.pyplot(fig)
        st.success("PCA applied!")

    st.markdown("### Remove Low Variance")
    var_threshold = st.number_input("Variance Threshold", value=0.001, key="var_threshold")
    if st.button("Run Remove Low Variance"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                embedx.remove_low_variance(threshold=var_threshold)
        st.success("Low variance features removed!")

with col2:
    st.subheader("☆ Visualize ☆")
    st.markdown("---")
    st.markdown("### UMAP")
    dim_umap = st.selectbox("Dimensionality", [2, 3], key="umap_dim")
    if st.button("Run UMAP"):
        if len(st.session_state.selected_columns) is 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                st.subheader(f"`{col}`")
                fig = embedx.visualize_umap(dim=dim_umap)
                st.pyplot(fig)

    st.markdown("### t-SNE")
    dim_tsne = st.selectbox("Dimensionality", [2, 3], key="tsne_dim")
    if st.button("Run t-SNE"):
        if len(st.session_state.selected_columns) is 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                st.subheader(f"`{col}`")
                fig = embedx.visualize_tsne(dim=dim_tsne)
                st.pyplot(fig)

    st.markdown("### Neighbors")
    n_neighbors = st.number_input("Neighbors", value=10, key="neighbors_count")
    threshold_neighbors = st.number_input("Threshold", value=0.95, key="neighbors_threshold")
    if st.button("Visualize Neighbors"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            fig = embedx.visualize_neighbors(threshold=threshold_neighbors, n_neighbors=n_neighbors)
            st.pyplot(fig)

    st.markdown("### Norms")
    if st.button("Visualize Norms"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            fig = embedx.visualize_norm_histogram()
            st.pyplot(fig)

    st.markdown("### Clusters")
    cluster_method = st.selectbox("Cluster Method", ["kmeans", "dbscan", "hdbscan", "gmm", "spectral"], key="cluster_method")
    viz_method = st.selectbox("Visualization Method", ["umap", "tsne"], key="viz_method")
    dim_clusters = st.selectbox("Dimensionality", [2, 3])
    if st.button("Run Clusters Visualization"):
         if len(st.session_state.selected_columns) is 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
         for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                st.subheader(f"`{col}`")
                fig = embedx.cluster_and_visualize(method=cluster_method, viz_method=viz_method, dim=dim_clusters)
                st.pyplot(fig)

with col3:
    st.subheader("☆ and Beyond!☆")
    st.markdown("---")
    st.markdown("### Cluster Embeddings")
    cluster_method = st.selectbox("Method", ["kmeans", "dbscan", "hdbscan", "spectral", "gmm"], key="cluster_method_select")
    n_clusters = st.number_input("n_clusters", value=10, key="n_clusters")
    eps = st.number_input("eps", value=0.5, key="eps_value")
    min_samples = st.number_input("min_samples", value=5, key="min_samples_value")
    min_cluster_size = st.number_input("min_cluster_size", value=5, key="min_cluster_size_value")
    n_components_cluster = st.number_input("n_components", value=10, key="n_components_cluster_value")

    if st.button("Run Cluster"):
        for col in st.session_state.selected_columns:
            embedx = st.session_state.embedxs.get(col)
            if embedx is not None:
                kwargs = {}
                if cluster_method == "kmeans":
                    kwargs["n_clusters"] = n_clusters
                elif cluster_method == "dbscan":
                    kwargs["eps"] = eps
                    kwargs["min_samples"] = min_samples
                elif cluster_method == "hdbscan":
                    kwargs["min_cluster_size"] = min_cluster_size
                elif cluster_method == "spectral":
                    kwargs["n_clusters"] = n_clusters
                elif cluster_method == "gmm":
                    kwargs["n_components"] = n_components_cluster
                labels = embedx.cluster_embeddings(method=cluster_method, **kwargs)
                st.write(labels)
            else:
                st.error("Load embeddings first!")

    st.markdown("### Compare Models")
    st.warning("Add a file input for second embeddings here!")

    st.markdown("### Semantic Coverage")
    st.warning("Add UI for labels input and top_n here!")

    st.markdown("### Intra/Inter Cluster & Others")
    st.warning("Add UI for advanced metrics like intracluster_variance, intercluster_distance, density, decay, etc.")
