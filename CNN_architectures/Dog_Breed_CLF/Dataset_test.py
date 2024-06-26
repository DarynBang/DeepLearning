import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from icecream import ic
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

os.chdir(r'D:\dog_breed_obj_detect')
print(os.listdir())

images_path = os.path.join(os.getcwd(), 'images')

images_list_dir = os.listdir(images_path)



transform = A.Compose([
    A.Resize(128, 128),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0), max_pixel_value=255),
    ToTensorV2()
])


image_dataset = []
labels = []


def preprocess_img(images_dir, label):
    for image_dir in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_dir)
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img, dtype=np.float32)
        if img is not None:
            augmented = transform(image=img_arr)
            augmented_image = augmented['image']

            image_dataset.append(augmented_image)
            labels.append(label)




for label, dog_breed in enumerate(images_list_dir):
    print(dog_breed)
    dog_breed_dir = os.path.join(images_path, dog_breed)
    preprocess_img(dog_breed_dir, label)
    break


# image_dataset = np.stack(image_dataset, dtype=np.float32)
# labels = np.array(labels, dtype=np.float32)

print(image_dataset[0])
print(image_dataset[0].shape)
# Numpy shape is (Height, Width, Dimension or Channel)
# ToTensorV2 converts numpy array into a Torch Tensor with shape (Batch, Channel, Height, Width)
print(labels[0])


def save_dataset(save_dir, images, labels):
    np.save(os.path.join(save_dir, "images.npy"), images)
    np.save(os.path.join(save_dir, "labels.npy"), labels)










