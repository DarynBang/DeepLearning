import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
plt.imshow(training_data[0][0].reshape(65, 65))
plt.show()

X = [data[0] for data in training_data]
X_arr = np.array(X)

y = [data[1] for data in training_data]
y_arr = np.array(y)

X_tensor = torch.tensor(X_arr, dtype=torch.float32).view(-1, 1, 65, 65)
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


# Computing accuracy
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features
        targets = torch.argmax(targets, dim=1).long()
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum().item()
    return float(correct_pred)/num_examples * 100


# Train network
for epoch in tqdm(range(num_epochs)):
    model = model.train()
    for batch_idx, (features, targets) in tqdm(enumerate(train_loader)):
        features = features.view(-1, 1, 65, 65)
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

    model = model.eval()
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
        epoch + 1, num_epochs,
        compute_accuracy(model, train_loader)))



with torch.no_grad(): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))


torch.save(model, r'C:\Users\steph\Desktop\Excel-CSV files\DogsVsCatsmodel.pth')
# print("Model saved complete")


