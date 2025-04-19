import os
import cv2
import numpy as np
import random

train_dir = 'DataSplitTrain/train'

def random_rotation(image):
    angle = random.uniform(-35, -15) if random.random() < 0.5 else random.uniform(15, 35)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def random_flip(image):
    flip_type = random.choice([0, 1, -1])
    return cv2.flip(image, flip_type)

def random_brightness_contrast(image):
    def rand_factor():
        return random.uniform(0.8, 0.9) if random.random() < 0.5 else random.uniform(1.1, 1.2)
    brightness = rand_factor()
    contrast = rand_factor()
    img = image.astype(np.float32) * contrast + (brightness - 1) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, std_range=(5, 15)):
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Telusuri hanya folder train
for suku in os.listdir(train_dir):
    suku_path = os.path.join(train_dir, suku)
    if not os.path.isdir(suku_path):
        continue

    for filename in os.listdir(suku_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(suku_path, filename)
            image = cv2.imread(filepath)
            base_filename, ext = os.path.splitext(filename)

            # Simpan hasil augmentasi ke folder yang sama
            cv2.imwrite(os.path.join(suku_path, f"{base_filename}_flip{ext}"), random_flip(image))
            cv2.imwrite(os.path.join(suku_path, f"{base_filename}_brightness{ext}"), random_brightness_contrast(image))
            cv2.imwrite(os.path.join(suku_path, f"{base_filename}_rotate{ext}"), random_rotation(image))
            cv2.imwrite(os.path.join(suku_path, f"{base_filename}_noise{ext}"), add_gaussian_noise(image))

            print(f"[✓] train/{suku}/{filename} -> augmentasi disimpan di folder yang sama")

print("✅ Augmentasi hanya dilakukan di folder train selesai.")
