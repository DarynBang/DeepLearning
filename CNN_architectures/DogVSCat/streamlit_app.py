import torch
import numpy as np
import streamlit as st
import albumentations as A
import cv2
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c2(self.c1(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        res_channels = in_channels // 4
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, 1, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.relu(torch.add(f, x))

        return f

## Model combines Residual blocks, batch normalization, dropout regularization
class Network(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(Network, self).__init__()
        self.c1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.resblock1 = ResBlock(in_channels=64, out_channels=64)

        self.c2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.resblock2 = ResBlock(in_channels=128, out_channels=128)

        self.c3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.resblock3 = ResBlock(in_channels=256, out_channels=256)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=256*7*7, out_features=num_classes)

    def forward(self, x):
        output = self.relu(self.c1(x))
        output = self.resblock1(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c2(output))
        output = self.resblock2(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c3(output))
        output = self.resblock3(self.dropout(output))
        output = self.pool(output)

        output = torch.flatten(output, 1)
        logits = self.fc1(output)

        probas = f.softmax(logits, dim=1)
        return logits, probas

model = torch.load(r'C:\Users\steph\Desktop\Excel-CSV files\DogsVsCatsmodel.pth')
model.eval()

IMG_HEIGHT = 65
IMG_WIDTH = 65

# ToTensorV2 converts an image to a 3 channel image by default
preprocessing = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    # A.ToGray(always_apply=True),
    ToTensorV2()
])

def preprocess_image(image):
    image = np.array(image, dtype=np.float32)
    image = preprocessing(image=image)['image']
    return image.unsqueeze(0)  # Add batch dimension

def predict(image):
    image_tensor = preprocess_image(image)
    logits, probas = model(image_tensor)
    cat_prob = probas[0][0].item()
    dog_prob = probas[0][1].item()

    return cat_prob, dog_prob


st.title("Dog vs Cat Image Classification")
uploaded_file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])
st.markdown("----")

if uploaded_file is not None:
    st.subheader("Image Classification")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, caption='Uploaded Image', channels="GRAY")
    cat_prob, dog_prob = predict(image)
    if abs(dog_prob - cat_prob) >= 0.2:
        animal_prob = dog_prob if dog_prob > cat_prob else cat_prob
        prediction = "Dog" if dog_prob > cat_prob else "Cat"

        st.write(f'The model predicts: {prediction} (Probability: {animal_prob * 100:.2f}%)')

    elif abs(dog_prob - cat_prob) < 0.2:
        st.write("The model can't accurately identify the species.")

    extra_info = st.button("Display additional Information", type="primary")
    if extra_info:
        st.write(f'Probability of being a cat: {cat_prob}')
        st.write(f"Probability of being a dog: {dog_prob} ")





