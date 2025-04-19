import streamlit as st
import pandas as pd
from PIL import Image
import cv2 as cv
import tempfile
from deepface import DeepFace


def verify(img1_path,img2_path,model_name = "VGG-Face"):
  img1 = cv.imread(img1_path)
  img2 = cv.imread(img2_path)

  result = DeepFace.verify(img1_path,img2_path,model_name = model_name) # mendapatkan hasil face similarity
  return result


def detech_similarity():

    st.title("Face Similarity")
    st.subheader("Aplikasi ini dapat mendeteksi kesamaan wajah dari dua gambar yang diupload.")

    col1, col2 = st.columns(2)

    with col1:
        
        st.header("Gambar Pertama")

        gambar_pertama = st.file_uploader("Upload a file Gambar Pertama", type=["jpg", "png", "jpeg"])
        if gambar_pertama:
            Image.open(gambar_pertama)
            img1 = Image.open(gambar_pertama)
            st.image(img1, caption="Gambar Pertama", use_container_width=True)
            

    with col2:
        st.header("Gambar Kedua")
        gambar_kedua = st.file_uploader("Upload a file Gambar Kedua", type=["jpg", "png", "jpeg"])
        if gambar_kedua:
            img2 = Image.open(gambar_kedua)
            st.image(img2, caption="Gambar Kedua", use_container_width=True)

    col3, col4,col5 = st.columns(3)

    if gambar_pertama and gambar_kedua:
        with col4:
            if st.button("Deteksi Wajah dan Etnis", key="detect_button"):
                # Simpan ke file sementara
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1, \
                    tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
                    
                    img1_path = tmp1.name
                    img2_path = tmp2.name
                    img1.save(img1_path)
                    img2.save(img2_path)

                    # Panggil fungsi verifikasi
                    konfirmasi = verify(img1_path, img2_path)
                    st.write("Hasil Verifikasi:")
                    st.write(konfirmasi)

