# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from embedx.core import Embedx

# ========================
# SETUP: TITLE & SIDEBAR
# ========================
st.set_page_config(page_title="EmbedX Excel Pipeline", layout="wide")
st.title("📊 EmbedX + Excel + Graph API App")

# ========================
# SECTION 1: Upload or Pull Excel
# ========================

st.header("📂 Upload Excel or Pull from OneDrive")

upload_method = st.radio(
    "Choose how to get your data:",
    ("Upload from local", "Automate pull from OneDrive"),
)

if upload_method == "Upload from local":
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("✅ Preview of your data:", df.head())
else:
    st.info("🔒 TODO: Implement Microsoft Graph API pull here.")
    if st.button("Pull Latest File"):
        # Call your Graph API pull function here, then load file
        st.warning("🚧 Not yet implemented: replace with your pull function.")
        df = None  # Replace with your loaded DataFrame

# ========================
# SECTION 2: Generate Embeddings
# ========================

if "df" in locals() and df is not None:
    st.header("🔢 Generate Embeddings")

    text_columns = st.multiselect(
        "Select text column(s) to embed",
        df.columns.tolist(),
    )

    if text_columns:
        # Combine selected columns into single text
        texts = df[text_columns].astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
        st.write(f"ℹ️ You have {len(texts)} text rows to embed.")

        if st.button("Generate Embeddings"):
            embeddings = Embedx.generate_embeddings(texts, model_name="all-MiniLM-L6-v2", output_path="embeddings.npy")
            st.success(f"✅ Embeddings shape: {embeddings.shape}")

# ========================
# SECTION 3: EmbedX Ops
# ========================

    st.header("⚙️ Run EmbedX Functions")

    # Example: Anomaly detection, duplicate removal, etc.
    if st.button("Run Basic Stats"):
        embeddings = np.load("embeddings.npy")
        embedx = Embedx(embeddings, verbose=True)
        stats = embedx.basic_stats()
        st.write(stats)

    if st.button("Remove Duplicates"):
        embeddings = np.load("embeddings.npy")
        embedx = Embedx(embeddings, verbose=True)
        embedx.remove_duplicates()
        np.save("embeddings_cleaned.npy", embedx.embeddings)
        st.success("✅ Duplicates removed. Saved as embeddings_cleaned.npy")

    # Add more buttons for your other EmbedX functions here!

# ========================
# SECTION 4: Export
# ========================

    st.header("⬇️ Export Results")

    if st.button("Export Back to OneDrive"):
        st.info("🚧 TODO: Implement push-back using Graph API here.")

    st.download_button(
        label="Download Cleaned Embeddings (.npy)",
        data=open("embeddings_cleaned.npy", "rb") if "embeddings_cleaned.npy" in locals() else None,
        file_name="embeddings_cleaned.npy",
    )

else:
    st.info("ℹ️ Please upload or pull an Excel file first.")

# ========================
# FOOTER
# ========================
st.caption("🚀 Built with EmbedX | Streamlit + Graph API starter.")
