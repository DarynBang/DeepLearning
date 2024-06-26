import os
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from icecream import ic
from torch.utils.data import Dataset


class DogBreedDataset(Dataset):
    def __init__(self,
                 images_folder,
                 train: bool,
                 transform=None):
        super().__init__()
        self.images_folder = images_folder
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self._load_images_and_labels()

    def _load_images_and_labels(self):
        breed_folders = os.listdir(self.images_folder)
        for label_idx, dog_breed_folder in enumerate(breed_folders):
            breed_folder_path = os.path.join(self.images_folder, dog_breed_folder)
            if os.path.isdir(breed_folder_path):
                for image_name in os.listdir(breed_folder_path):
                    image_path = os.path.join(breed_folder_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label_idx)


    def __len__(self):
        # Return length
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Return image and label
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error reading image {image_path}")

        img_arr = np.array(img, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=img_arr)
            img = augmented['image']

        return [img, label]



if __name__ == '__main__':
    images_path = r'D:\dog_breed_clf\train'

    transform = A.Compose([
        A.Resize(128, 128),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0), max_pixel_value=255)
    ])

    dataset = DogBreedDataset(images_folder=images_path, train=True, transform=transform)
    ic(f"Dataset size: {len(dataset)}")

    # Example to check a few samples
    for i in range(5):
        img = dataset[i]
        ic(img)
        ic(img[0].shape)



