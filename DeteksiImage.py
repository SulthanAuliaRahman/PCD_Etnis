import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow

from facenet_pytorch import MTCNN
from PIL import Image

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

def detect_eth():
    st.header("Deteksi Etnis")
    st.subheader("Aplikasi ini dapat mendeteksi etnis dari wajah yang diupload.")

    # Inisialisasi MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load model 
    model = load_model("best_model.keras")
    size = 280

    # Label dari class_indices(urutan harus sesuai training)
    class_labels = ['Asian','Jawa', 'Minang', 'Sunda'] 


    st.title("Deteksi Etnis Wajah")

    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca file jadi array numpy (OpenCV style)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)  # 1 untuk warna (BGR)

        if img is None:
            print("Gagal membaca gambar. Pastikan file ada dan formatnya benar.")
                
        else:
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)
            

            # Convert BGR (OpenCV) ke RGB (PIL)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Deteksi wajah dengan MTCNN
            boxes, _ = mtcnn.detect(img_pil)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = img_rgb[y1:y2, x1:x2]  # Crop wajah
                    try:
                        face_pil = Image.fromarray(face).resize((size, size))
                        face_array = img_to_array(face_pil)
                        face_array = preprocess_input(face_array)
                        face_array = np.expand_dims(face_array, axis=0)
                            # Prediksi etnis
                        pred = model.predict(face_array)
                        class_idx = np.argmax(pred)
                        confidence = pred[0][class_idx]
                        label = f"{class_labels[class_idx]} ({confidence*100:.1f}%)"

                        # Gambar bounding box dan label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 150), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
                        
                        print("yang di return kan",pred)

                    except Exception as e:
                        print("Error processing face:", e)
                        continue


        img = cv2.resize(img,(720,1280))
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi")


