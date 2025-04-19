import cv2
import os
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image


#Note: yang ilang suku dayak si zainal kalau pakai teknik crop ini

#MTCNN untuk deteksi 1 wajah saja
mtcnn = MTCNN(keep_all=False)

input_dir = "Dataset"
output_dir = "Cropped"

def crop_faces_from_directory(input_dir, output_dir):
    Total_Crop = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

                # Baca gambar
                img = cv2.imread(input_path)
                if img is None:
                    print(f"[!] Failed to read: {input_path}")
                    continue

                # Deteksi wajah
                face = detect_single_face_mtcnn(img)

                # crop dan simpan
                if face is not None:
                    x, y, w, h = face
                    height, width, _ = img.shape
                    x_end = min(x + w, width)
                    y_end = min(y + h, height)
                    x_start = max(x, 0)
                    y_start = max(y, 0)

                    face_crop = img[y_start:y_end, x_start:x_end]

                    if face_crop.size == 0:
                        print(f"[!] Skipped empty crop at: {input_path}")
                        continue

                    cv2.imwrite(output_path, face_crop)
                    print(f"[✓] Saved: {output_path}")
                    Total_Crop += 1
                else:
                    print(f"[!] No face found: {input_path}")

    print(f"\nTotal wajah berhasil dicrop: {Total_Crop}")

def detect_single_face_mtcnn(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None or len(boxes) == 0:
        return None
    x, y, x2, y2 = boxes[0]
    return int(x), int(y), int(x2 - x), int(y2 - y)

if __name__ == "__main__":
    crop_faces_from_directory(input_dir, output_dir)
