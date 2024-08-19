import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split
from Model import UpdatedNetwork, ConvBlock, SeBlock, ResBlock, Identity
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from Dataset import *

# Configuration and Set up
MODEL_PATH = r''
SAVE_PATH = r''
num_epochs = 40
batch_size = 16
lr = 0.0005
weight_decay = 0.0005
NUM_CLASSES = 120
ROOT_FILE = r''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(MODEL_PATH)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# Freeze weights if necessary
## for param in model.parameters():
##     param.requires_grad = False

freeze_up_to = 10  # Number of layers to freeze up to
for layer in model.layers[:freeze_up_to]:
    for param in layer.parameters():
        param.requires_grad = False


# model.layers[-1] = Identity()
#
# model.layers.append(nn.Sequential(
#     ConvBlock(448, 512, 3, stride=1, padding=1),
#     ResBlock(512, 8),
#     ResBlock(512, 8),
#     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
#
#     ConvBlock(512, 680, 3, stride=1, padding=1),
#     ResBlock(680, 8),
#     ResBlock(680, 8),
#     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
# ))
#
# model.fc = nn.Linear(680*3*3, NUM_CLASSES)

model.to(device)
print(model)

# Test model
x = torch.rand(10, 3, 224, 224).to(device)
with torch.no_grad():
    feature_map = model(x)
    print(feature_map.shape)  # Print the shape of the feature map

def save_checkpoint(state, filename=r""):
    print("-> Saving checkpoint")
    torch.save(state, filename)

def compute_accuracy(model, data_loader, device):
    correct_pred = 0
    num_examples = 0
    with torch.no_grad():  # Disable gradient computation
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            predicted_probs = torch.softmax(logits, 1)
            _, predicted_labels = torch.max(predicted_probs, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum().item()

    accuracy = float(correct_pred) / num_examples * 100
    return accuracy


criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Train network
def train() -> None:
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device).float()
            targets = targets.to(device)

            # Forward and Backprop:
            with autocast():
                scores = model(features)
                loss = criterion(scores, targets)

            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            scaler.scale(loss).backward()

            # Update weights and biases
            scaler.step(optimizer)
            scaler.update()

            # Logging - Monitor and store relevant information during training process
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.5f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))

        # Save model
        save_checkpoint(model, SAVE_PATH)

        if (epoch + 1) % 5 == 0 and (epoch + 1) != 0:
            print("Evaluating Model...")
            model.eval()
            if (epoch + 1) == num_epochs // 2:
                with torch.set_grad_enabled(False):
                    print('Epoch: %03d/%03d Test accuracy: %.4f%%' % (
                        epoch + 1, num_epochs,
                        compute_accuracy(model, test_loader, device)))

            else:
                with torch.set_grad_enabled(False):
                    print('Epoch: %03d/%03d Training accuracy: %.4f%%' % (
                        epoch + 1, num_epochs,
                        compute_accuracy(model, train_loader, device)))


def test() -> None:
    with torch.no_grad():  # save memory during inference
        print('Test accuracy: %.3f%%' % (compute_accuracy(model, test_loader, device)))

def save_model(path) -> None:
    torch.save(model, path)
    print("Model saved complete")


if __name__ == '__main__':
    root_file = ROOT_FILE

    dataset = DogBreedDataset(root_file, train=True, transform=train_transforms)
    print(f"Dataset size: {len(dataset)}")

    train_len = int(0.90 * len(dataset))
    test_len = len(dataset) - train_len

    print(train_len)
    print(test_len)

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    train()
    test()
    save_model(SAVE_PATH)

# Epoch: 080/080 Training accuracy: 44.6334%
# Test accuracy: 27.259%
