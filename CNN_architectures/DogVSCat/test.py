import torch
from torch.utils.data import TensorDataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load tensors from files
images = np.load(r'C:\Users\steph\Desktop\Image files\catvsdogimgs.npy', allow_pickle=True)
labels = np.load(r'C:\Users\steph\Desktop\Image files\catvsdoglabels.npy', allow_pickle=True)

print(len(images))

print(images[0], labels[0])
print(images[0].shape)

plt.imshow(images[0])
plt.show()


