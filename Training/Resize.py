import os
import cv2

# Folder sumber utama (yang punya subfolder seperti 'ambon', dll)
size = 280
source_root = 'Cropped'
output_root = 'resized'+str(size)+'x'+ str(size)

# Ukuran resize
target_size = (size, size)

# Telusuri seluruh subfolder dan file
for root, _, files in os.walk(source_root):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Path lengkap file sumber
        source_path = os.path.join(root, file)

        # Path relatif terhadap root (misal 'ambon/img1.jpg')
        rel_path = os.path.relpath(source_path, source_root)

        # Path tujuan (misal 'resized/ambon/img1.jpg')
        output_path = os.path.join(output_root, rel_path)

        # Buat folder tujuan jika belum ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Baca dan resize gambar
        image = cv2.imread(source_path)
        if image is None:
            print(f"Failed to read {source_path}, skipping...")
            continue

        resized = cv2.resize(image, target_size)
        cv2.imwrite(output_path, resized)

print("Selesai resize dengan struktur folder yang sama.")
