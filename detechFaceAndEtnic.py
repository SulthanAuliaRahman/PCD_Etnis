import streamlit as st
import pandas as pd
from PIL import Image


st.header("Aplikasi Deteksi Wajah dan Etnis")
st.subheader("Aplikasi ini dapat mendeteksi wajah dan etnis dari gambar yang diupload.")



col1, col2 = st.columns(2)

with col1:
    
    st.header("Gambar Pertama")

    gambar_pertama = st.file_uploader("Upload a file Gambar Pertama", type=["jpg", "png", "jpeg"])
    if gambar_pertama:
        img1 = Image.open(gambar_pertama)
        st.image(img1, caption="Gambar Pertama", use_container_width=True)

with col2:
    st.header("Gambar Kedua")
    gambar_kedua = st.file_uploader("Upload a file Gambar Kedua", type=["jpg", "png", "jpeg"])
    if gambar_kedua:
        img2 = Image.open(gambar_kedua)
        st.image(img2, caption="Gambar Kedua", use_container_width=True)

col3, col4,col5 = st.columns(3)

if gambar_pertama is not None and gambar_kedua is not None:
    with col4:
        st.button("Deteksi Wajah dan Etnis", key="detect_button")

