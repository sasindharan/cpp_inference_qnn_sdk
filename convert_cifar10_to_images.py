import pickle
import os
import numpy as np
import cv2

# Load CIFAR test batch
with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')

data = batch[b'data']       # shape (10000, 3072)
labels = batch[b'labels']   # shape (10000)

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Output folder
output_dir = "data/images_test"
os.makedirs(output_dir, exist_ok=True)

# Convert first N images (e.g., 100)
N = 100

for i in range(N):
    img = data[i].reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC

    label = classes[labels[i]]

    filename = f"{label}_{i}.png"
    path = os.path.join(output_dir, filename)

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img)

print("Done! Images saved to data/images_test/")