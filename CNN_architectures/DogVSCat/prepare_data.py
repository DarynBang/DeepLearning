import os
import cv2
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


os.chdir(r'C:\Users\steph\Desktop\Excel-CSV files\dogs-vs-cats')
cat_dir = f"{os.getcwd()}\Cats"
dog_dir = f"{os.getcwd()}\Dogs"

cats_list = os.listdir(cat_dir)
dogs_list = os.listdir(dog_dir)

transform = A.Compose([
    A.Resize(128, 128),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),  # Scales image pixel value and normalize them using mean and std
    ## Save it in numpy format because CPU works faster with numpy arrays than Pytorch tensors
    # ToTensorV2()  # converts image to Pytorch tensor
])

images = []
labels = []

def process_img(image_dir, label):
    file_list = os.listdir(image_dir)
    for file in tqdm(file_list):
        file_path = os.path.join(image_dir, file)

        image = cv2.imread(file_path)

        if image is not None:
            augmented = transform(image=image)
            augmented_image = augmented['image']
            images.append(augmented_image)

            labels.append(label)


process_img(cat_dir, [1, 0])
process_img(dog_dir, [0, 1])


images = np.stack(images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

print(len(images))
print(len(labels))

np.save(r'C:\Users\steph\Desktop\Image files\catvsdogimgs.npy', images)
np.save(r'C:\Users\steph\Desktop\Image files\catvsdoglabels.npy', labels)

print("Saving completed")





