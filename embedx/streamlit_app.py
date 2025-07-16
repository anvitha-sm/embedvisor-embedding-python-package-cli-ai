import streamlit as st
import pandas as pd
from graph_api import start_device_flow, acquire_token, list_excel_files, download_file

st.set_page_config(page_title="Excipher", layout="wide")
st.title("Excipher: Process and Visualize Excel Data")

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
    uploaded_file = st.file_uploader("Or upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.session_state.show_picker = False
        st.session_state.awaiting_auth = False 
        st.success("File uploaded!")

if st.session_state.df is not None:
    st.success(f"Loaded DataFrame with {len(st.session_state.df)} rows.")
    st.dataframe(st.session_state.df.head())
else:
    st.info("Connect to OneDrive or upload a file to get started.")

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
