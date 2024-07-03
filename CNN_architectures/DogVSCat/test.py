import torch
from torch.utils.data import Dataset
import albumentations as A
import os
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2


class DogVsCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        dog_dir = os.path.join(root_dir, "Dogs")
        cat_dir = os.path.join(root_dir, "Cats")

        self.dog_images = os.listdir(dog_dir)
        self.cat_images = os.listdir(cat_dir)

        self.images = [(os.path.join(dog_dir, img), 1) for img in self.dog_images] + \
                      [(os.path.join(cat_dir, img), 0) for img in self.cat_images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        image_arr = np.asarray(image)

        if self.transform:
            transformed = self.transform(image=image_arr)
            image_arr = transformed['image']


        return image_arr, label


root_file = r'C:\Users\steph\Desktop\Excel-CSV files\dogs-vs-cats'
IMG_HEIGHT, IMG_WIDTH = 224, 224

transforms = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.HorizontalFlip(p=0.65),
    A.Rotate(limit=20, p=0.75),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2()
])


dataset = DogVsCatDataset(root_file, transforms)
print(len(dataset))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


for img, label in data_loader:
    print(img)
    print(label)

    break


