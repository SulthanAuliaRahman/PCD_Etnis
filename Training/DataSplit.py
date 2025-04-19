import os
import shutil
import random
from math import floor

# Path dataset awal
original_dataset_dir = 'resized280x280'

# Path hasil split
base_dir = 'DataSplitTrain'
splits = [ 'test', 'train','val']
split_ratio = {'test': 0.15, 'train': 0.7, 'val': 0.15, }

# Buat folder split (train/val/test)
for split in splits:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

# Iterasi setiap nama
for nama in os.listdir(original_dataset_dir):
    nama_path = os.path.join(original_dataset_dir, nama)
    if not os.path.isdir(nama_path):
        continue

    # Iterasi setiap suku dalam nama
    for suku in os.listdir(nama_path):
        suku_path = os.path.join(nama_path, suku)
        if not os.path.isdir(suku_path):
            continue

        images = os.listdir(suku_path)
        if len(images) < 3:
            print(f"Skipped: {nama}/{suku} (kurang dari 3 gambar)")
            continue

        random.shuffle(images)

        # Ambil minimal 1 val, 1 test
        val_images = [images.pop()]
        test_images = [images.pop()]
        remaining = images

        # Hitung jumlah tambahan berdasarkan rasio
        total = len(remaining)
        train_count = floor(split_ratio['train'] * total)
        val_count = floor(split_ratio['val'] * total)
        test_count = total - train_count - val_count

        train_images = remaining[:train_count]
        val_images += remaining[train_count:train_count + val_count]
        test_images += remaining[train_count + val_count:]

        split_images = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        # Copy ke direktori baru
        for split in splits:
            split_suku_path = os.path.join(base_dir, split, suku)
            os.makedirs(split_suku_path, exist_ok=True)

            for img in split_images[split]:
                src = os.path.join(suku_path, img)
                dst = os.path.join(split_suku_path, f"{nama}_{img}")
                shutil.copy2(src, dst)

print("Dataset berhasil di-split dengan minimal 1 val dan 1 test per subjek.")
