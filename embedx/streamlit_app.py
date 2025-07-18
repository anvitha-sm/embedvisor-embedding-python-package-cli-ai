import streamlit as st
import pandas as pd
import numpy as np
from embedx.core import Embedx
import io
from graph_api import start_device_flow, acquire_token, list_excel_files, download_file, upload_to_onedrive

st.set_page_config(page_title="Excipher", layout="wide")
st.title("Excipher: Process and Visualize Excel Data")

def connect_to_onedrive():
    if st.session_state.token is None:
        try:
            app, flow, message = start_device_flow()
            st.session_state.auth_app = app
            st.session_state.auth_flow = flow
            st.session_state.device_flow_message = message
            st.session_state.awaiting_auth = True
            st.session_state.show_picker = False 
        except Exception as e:
            st.error(f"Could not start Device Code Flow:\n\n{e}")
    if st.session_state.awaiting_auth:
        st.info("Follow these instructions to authorize Excipher on OneDrive:")
        st.code(st.session_state.device_flow_message)

        result = acquire_token(st.session_state.auth_app, st.session_state.auth_flow)
        if "access_token" in result:
            st.session_state.token = result["access_token"]
            st.session_state.excel_files = list_excel_files(st.session_state.token)
            st.session_state.show_picker = True
            st.session_state.awaiting_auth = False

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

def update_embeddings(embedx, selected_columns):
    start = 0
    for col in selected_columns:
        original = st.session_state.embeddings[col]
        dim = 1 if original.ndim == 1 else original.shape[1]
        end = start + dim
        st.session_state.embeddings[col] = embedx.embeddings[:, start:end]
        if dim == 1:
            st.session_state.embeddings[col] = st.session_state.embeddings[col].squeeze()
        start = end

def update_session_embeddings(embedx, selected_columns):
    if embedx.embeddings.shape[1] != sum(st.session_state.embeddings[col].shape[1] if len(st.session_state.embeddings[col].shape) > 1 else 1 for col in selected_columns):
        st.warning("Skipping update: shape mismatch after transformation.")
        return
    start = 0
    for col in selected_columns:
        col_data = st.session_state.embeddings[col]
        col_dim = col_data.shape[1] if len(col_data.shape) > 1 else 1
        updated = embedx.embeddings[:, start:start+col_dim]
        if col_dim == 1:
            updated = updated.flatten()
        st.session_state.embeddings[col] = updated
        start += col_dim

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
if "labels" not in st.session_state:
    st.session_state.labels = None

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
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
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
    st.session_state.labels = st.selectbox(
        "Select label column (optional):",
        ["None"] + all_columns
    )

    col_gen, col_format, col_save = st.columns([1.5, 1, 1])

    with col_gen:
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

    with col_format:
        cola, colb = st.columns([2, 1.2])
        with cola:
            col1, col2 = st.columns([2, 3])
            with col1:
                file_format = st.radio("Format", ["npy", "csv"], horizontal=True)
            with col2:
                destination = st.radio("Destination", ["Download", "OneDrive"], horizontal=True)
            filename = st.text_input("Filename", value=f"transformed_embeddings.{file_format}")
        with colb:
            if st.button("Save Embeddings"):
                data = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
                if data is not None:
                    if file_format == "npy":
                        bytes_data = io.BytesIO()
                        np.save(bytes_data, data)
                        bytes_data.seek(0)
                    else:
                        df = pd.DataFrame(data)
                        bytes_data = io.BytesIO()
                        df.to_csv(bytes_data, index=False)
                        bytes_data.seek(0)

                    if destination == "Download":
                        st.download_button(
                            label="Download File",
                            data=bytes_data,
                            file_name=filename,
                            mime="application/octet-stream"
                        )
                    else:
                        connect_to_onedrive()
                        upload_to_onedrive(filename, bytes_data.getvalue(), st.session_state.token)
                        st.success(f"Embeddings saved as {filename} to OneDrive!")

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
            update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Remove Duplicates")
    threshold = st.number_input("Threshold", value=0.99, key="duplicate_threshold")
    neighbors = st.number_input("Neighbors", value=10, key="duplicate_neighbors")
    if st.button("Run Remove Duplicates"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                embedx.remove_duplicates(threshold=threshold, neighbors=neighbors)
                st.success("Duplicates removed!")
                update_session_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Remove Outliers")
    contamination = st.number_input("Contamination", value=0.01, key="outlier_contamination")
    if st.button("Run Remove Outliers"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                embedx.remove_outliers(contamination=contamination)
                st.success("Outliers removed!")
                update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Center Embeddings")
    if st.button("Center Embeddings"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                embedx.center()
                st.success("Centered!")
                update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Normalize")
    method = st.selectbox("Method", ["l1", "l2"], key="normalize_method")
    if st.button("Run Normalize"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                embedx.normalize(method=method)
                st.success("Normalized!")
                update_embeddings(embedx, st.session_state.selected_columns)

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
            update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Remove Low Variance")
    var_threshold = st.number_input("Variance Threshold", value=0.001, key="var_threshold")
    if st.button("Run Remove Low Variance"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                embedx.remove_low_variance(threshold=var_threshold)
                st.success("Low variance features removed!")
                update_embeddings(embedx, st.session_state.selected_columns)

with col2:
    st.subheader("☆ Visualize ☆")
    st.markdown("---")
    st.markdown("### UMAP")
    dim_umap = st.radio("Dimensionality", [2, 3], horizontal=True, key="umap_dim")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_umap"
        )

    if st.button("Run UMAP"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        if st.session_state.labels is None:
            st.error("Select a label column to visualize!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            if st.session_state.labels != "None":
                if type(st.session_state.labels) is str:
                    labels = st.session_state.df[st.session_state.labels].values
                else:
                    labels = st.session_state.labels
                embedx = Embedx(combined_embeddings, labels=labels, verbose=False)
            else:
                embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                fig = embedx.visualize_umap(dim=dim_umap)
                if save_plot_option != "None":
                    filename= f"umap_plot_{dim_umap}d.png"
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    if save_plot_option == "Download":
                        st.download_button(
                            label="Download Plot",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_button_umap"
                        )
                    elif save_plot_option == "OneDrive":
                        connect_to_onedrive()
                        upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
                st.pyplot(fig)
                update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### t-SNE")
    dim_tsne = st.radio("Dimensionality", [2, 3], horizontal=True, key="tsne_dim")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_tsne"
        )
    if st.button("Run t-SNE"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        if st.session_state.labels is None:
            st.error("Select a label column to visualize!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            if st.session_state.labels != "None":
                if type(st.session_state.labels) is str:
                    labels = st.session_state.df[st.session_state.labels].values
                else:
                    labels = st.session_state.labels
                embedx = Embedx(combined_embeddings, labels=labels, verbose=False)
            else:
                embedx = Embedx(combined_embeddings, verbose=False)
            if embedx is not None:
                fig = embedx.visualize_tsne(dim=dim_tsne)
                if save_plot_option != "None":
                    filename= f"tsne_plot_{dim_tsne}d.png"
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    if save_plot_option == "Download":
                        st.download_button(
                            label="Download Plot",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_button_tsne"
                        )
                    elif save_plot_option == "OneDrive":
                        connect_to_onedrive()
                        upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
                st.pyplot(fig)
                update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Neighbors")
    n_neighbors = st.number_input("Neighbors", value=10, key="neighbors_count")
    threshold_neighbors = st.number_input("Threshold", value=0.95, key="neighbors_threshold")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_neigh"
        )
    if st.button("Visualize Neighbors"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            fig, similarites, num_neighbors_close = embedx.visualize_neighbors(threshold=threshold_neighbors, n_neighbors=n_neighbors)
            if save_plot_option != "None":
                    filename= f"neighbors_plot.png"
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    if save_plot_option == "Download":
                        st.download_button(
                            label="Download Plot",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_button_neigh"
                        )
                    elif save_plot_option == "OneDrive":
                        connect_to_onedrive()
                        upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
            st.pyplot(fig)
            st.write(f"Similarities: {similarites}")
            st.write(f"Number of neighbors close: {num_neighbors_close}")
            update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Norms")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_norms"
        )
    if st.button("Visualize Norms"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
            fig = embedx.visualize_norm_histogram()
            if save_plot_option != "None":
                    filename= f"norms_plot.png"
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    if save_plot_option == "Download":
                        st.download_button(
                            label="Download Plot",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_button_norms"
                        )
                    elif save_plot_option == "OneDrive":
                        connect_to_onedrive()
                        upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
            st.pyplot(fig)
            update_embeddings(embedx, st.session_state.selected_columns)

with col3:
    st.subheader("☆ and Beyond!☆")
    st.markdown("---")
    st.markdown("### Cluster Embeddings")
    cluster_method = st.selectbox("Clustering Method", ["kmeans", "dbscan", "hdbscan", "spectral", "gmm"], key="cluster_method_select")
    visualization_method = st.selectbox("Visualization Method", ["None", "umap", "tsne"], key="visualization_method")
    dim_viz_clusters = st.selectbox("Dimensionality", [2, 3], key="dim_viz_clusters")
    n_clusters = st.number_input("Number of Clusters", value=10, key="n_clusters")
    eps = st.number_input("Epsilon", value=0.5, key="eps_value")
    min_samples = st.number_input("Minimum Samples", value=5, key="min_samples_value")
    min_cluster_size = st.number_input("Minimum Cluster Size", value=5, key="min_cluster_size_value")
    n_components_cluster = st.number_input("Number of Components", value=10, key="n_components_cluster_value")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_clusters"
        )
    if st.button("Run Cluster"):
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            embedx = Embedx(combined_embeddings, verbose=False)
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
                embedx.set_labels(labels)
                fig = None
                if visualization_method =="umap":
                    fig = embedx.visualize_umap(dim=dim_viz_clusters, save_path="cluster_umap_plot.png" if save_plot else None)
                    st.pyplot(fig)
                elif visualization_method == "tsne":
                    fig = embedx.visualize_tsne(dim=dim_viz_clusters, save_path="cluster_tsne_plot.png" if save_plot else None)
                    st.pyplot(fig)
                if fig is not None:
                    st.pyplot(fig)
                    if save_plot_option != "None":
                        filename= f"{cluster_method}_plot_{visualization_method}.png"
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", bbox_inches="tight")
                        buffer.seek(0)
                        if save_plot_option == "Download":
                            st.download_button(
                                label="Download Plot",
                                data=buffer,
                                file_name=filename,
                                mime="image/png",
                                key=f"download_button_tsne"
                            )
                        elif save_plot_option == "OneDrive":
                            connect_to_onedrive()
                            upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
                st.write(labels)
                st.session_state.labels = labels
            else:
                st.error("Load embeddings first!")

    st.markdown("### Semantic Coverage")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_semantic"
        )
    if st.button("Run Semantic Coverage"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            if st.session_state.labels != "None":
                if type(st.session_state.labels) is str:
                    labels = st.session_state.df[st.session_state.labels].values
                else:
                    labels = st.session_state.labels
                embedx = Embedx(combined_embeddings, labels=labels, verbose=False)
            else:
                embedx = Embedx(combined_embeddings, verbose=False)
            fig, coverages = embedx.semantic_coverage()
            if save_plot_option != "None":
                filename= f"semantic_plot.png"
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                if save_plot_option == "Download":
                    st.download_button(
                        label="Download Plot",
                        data=buffer,
                        file_name=filename,
                        mime="image/png",
                        key=f"download_button_semantic"
                    )
                elif save_plot_option == "OneDrive":
                    connect_to_onedrive()
                    upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
            st.pyplot(fig)
            df_coverages = pd.DataFrame(list(coverages.items()), columns=["Cluster Label", "Coverage"])
            st.table(df_coverages)
            update_embeddings(embedx, st.session_state.selected_columns)
    

    st.markdown("### Intracluster Variance")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_intra"
        )
    if st.button("Run Intracluster Variance"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            if st.session_state.labels != "None":
                if type(st.session_state.labels) is str:
                    labels = st.session_state.df[st.session_state.labels].values
                else:
                    labels = st.session_state.labels
                embedx = Embedx(combined_embeddings, labels=labels, verbose=False)
            else:
                embedx = Embedx(combined_embeddings, verbose=False)
            fig, variances = embedx.intracluster_variance()
            if save_plot_option != "None":
                filename= f"intra_plot.png"
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                if save_plot_option == "Download":
                    st.download_button(
                        label="Download Plot",
                        data=buffer,
                        file_name=filename,
                        mime="image/png",
                        key=f"download_button_intra"
                    )
                elif save_plot_option == "OneDrive":
                    connect_to_onedrive()
                    upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
            st.pyplot(fig)
            df_variances = pd.DataFrame(list(variances.items()), columns=["Cluster Label", "Intra-cluster Variance"])
            st.table(df_variances)
            update_embeddings(embedx, st.session_state.selected_columns)

    st.markdown("### Intercluster Distance")
    save_plot_option = st.radio(
        "Save plot as:",
            ["None", "Download", "OneDrive"],
            horizontal=True,
            key=f"save_option_inter"
        )
    if st.button("Run Intercluster Distance"):
        if len(st.session_state.selected_columns) == 0:
            st.error("Load an input file, generate embeddings, and select columns first!")
        combined_embeddings = get_all_embeddings(st.session_state.selected_columns, st.session_state.embeddings)
        if combined_embeddings is not None:
            if st.session_state.labels != "None":
                if type(st.session_state.labels) is str:
                    labels = st.session_state.df[st.session_state.labels].values
                else:
                    labels = st.session_state.labels
                embedx = Embedx(combined_embeddings, labels=labels, verbose=False)
            else:
                embedx = Embedx(combined_embeddings, verbose=False)
            fig, dists = embedx.intercluster_distance()
            if save_plot_option != "None":
                    filename= f"inter_plot.png"
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    if save_plot_option == "Download":
                        st.download_button(
                            label="Download Plot",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_button_inter"
                        )
                    elif save_plot_option == "OneDrive":
                        connect_to_onedrive()
                        upload_to_onedrive(filename, buffer.getvalue(), st.session_state.token)
            st.pyplot(fig)
            df_dists = pd.DataFrame(list(dists.items()), columns=["Cluster Pair", "Inter-cluster Distance"])
            st.table(df_dists)
            update_embeddings(embedx, st.session_state.selected_columns)
