import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from utils import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c2(self.c1(x)))



class SE_Block(nn.Module):
    def __init__(self,
                 n_channels: int,
                 ratio: int) -> None:
        super(SE_Block, self).__init__()
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features = n_channels, out_features = (n_channels // ratio))

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features = (n_channels // ratio), out_features = n_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: Tensor) -> Tensor:
        # Squeeze Operation
        out = self.GlobalAvgPool(x)

        # Excitation operation
        out = self.relu(self.fc1(out))

        out = self.sigmoid(self.fc2(out))

        ## Reshape output to match the input dimensions
        ## out.size(0) -> Batch Size, out.size(1) -> n_channels
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * x


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(ResBlock, self).__init__()
        res_channels = in_channels // 4
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, 1, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.se_block = SE_Block(out_channels, 8)
        self.relu = nn.ReLU()

    def forward(self,
                x: Tensor) -> Tensor:
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        f = self.se_block(f)
        f = self.relu(torch.add(f, x))

        return f

## Model combines Residual blocks, batch normalization, dropout regularization
class Network(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=2) -> None:
        super(Network, self).__init__()
        self.c1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.resblock1 = ResBlock(in_channels=64, out_channels=64)

        self.c2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.resblock2 = ResBlock(in_channels=128, out_channels=128)

        self.c3 = ConvBlock(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.resblock3 = ResBlock(in_channels=192, out_channels=192)

        self.c4 = ConvBlock(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.resblock4 = ResBlock(in_channels=256, out_channels=256)

        self.c5 = ConvBlock(in_channels=256, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.resblock5 = ResBlock(in_channels=320, out_channels=342)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=320 * 8 * 8, out_features=num_classes)

    def forward(self,
                x: Tensor) -> [Tensor, Tensor]:
        output = self.relu(self.c1(x))
        output = self.resblock1(self.dropout(output))

        output = self.relu(self.c2(output))
        output = self.resblock2(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c3(output))
        output = self.resblock3(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c4(output))
        output = self.resblock4(self.dropout(output))
        output = self.pool(output)

        output = self.relu(self.c5(output))
        output = self.resblock5(self.dropout(output))
        output = self.pool(output)

        output = torch.flatten(output, 1)
        logits = self.fc1(output)

        probas = f.softmax(logits, dim=1)
        return logits, probas

## Test Network
# model = Network()
# x = torch.rand(1, 1, 65, 65)
# logits, probas = model(x)
#
# print(logits)
# print(probas)


training_data = np.load(r'C:\Users\steph\Desktop\Excel-CSV files\dogs-vs-cats\training_data_test.npy',
                        allow_pickle=True)

print(training_data[0])
print(training_data[0][0])
plt.imshow(training_data[0][0].reshape(128, 128))
plt.show()

X = [data[0] for data in training_data]
X_arr = np.array(X)

y = [data[1] for data in training_data]
y_arr = np.array(y)

X_tensor = torch.tensor(X_arr, dtype=torch.float32).view(-1, 1, 128, 128) / 255
y_tensor = torch.tensor(y_arr, dtype=torch.long)

lr = 0.001
num_epochs = 10
batch_size = 64

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, stratify=y_tensor, shuffle=True)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Network()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Train network
def train() -> None:
    for epoch in tqdm(range(num_epochs)):
        model = model.train()
        for batch_idx, (features, targets) in tqdm(enumerate(train_loader)):
            features = features.view(-1, 1, 128, 128)
            targets = torch.argmax(targets, dim=1).long()

            # Forward and Backprop
            logits, probas = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # Update weights and biases
            optimizer.step()

            # Logging - Monitor and store relevant information during training process
            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        model = model.eval()

        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
            epoch + 1, num_epochs,
            compute_accuracy(model, train_loader)))


def test() -> None:
    with torch.no_grad():  # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

def save_model() -> None:
    torch.save(model, r'C:\Users\steph\Desktop\Excel-CSV files\BasicCNNsmodel.pth')
    print("Model saved complete")


if __name__ == "__main__":
    # train()
    # test()
    # save_model()
    pass

