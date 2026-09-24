#!/usr/bin/env python3
import os
import random

dataset_root = "../v26/dataset/"
output_file = "./calibration/calibration_data.txt"
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

all_images = []
for root, dirs, files in os.walk(dataset_root):
    if "valid" in root or "test" in root:
        continue
        
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            all_images.append(os.path.join(root, file))

if not all_images:
    exit(1)

num_samples = min(30, len(all_images))
calibration_samples = random.sample(all_images, num_samples)

with open(output_file, "w") as f:
    for img_path in calibration_samples:
        f.write(os.path.abspath(img_path) + "\n")
