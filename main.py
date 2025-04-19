import streamlit as st
from DeteksiImage import detect_eth
from DetechFaceSimilarity import detech_similarity

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Pilih Halaman", ("Deteksi Etnis", "Face Similarity"))

if page == "Deteksi Etnis":
    detect_eth()
elif page == "Face Similarity":
    detech_similarity()
