import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler
import torch

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 use_batch_norm: bool = True,
                 use_activation: bool = False,
                 **kwargs) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_batch_norm = use_batch_norm
        self.use_activation = use_activation

        self.activation = nn.LeakyReLU(0.1)


    def forward(self, x) -> Tensor:
        f = self.conv(x)
        if self.use_batch_norm:
            f = self.bn(f)

        if self.use_activation:
            f = self.activation(f)


        return f

class ResBlock(nn.Module):
    def __init__(self,
                 channels: int) -> None:
        super(ResBlock, self).__init__()
        res_channels = channels // 2
        self.c1 = ConvBlock(channels, res_channels, 1, stride=1, padding=0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride=1, padding=1)
        self.c3 = ConvBlock(res_channels, channels, 1, stride=1, padding=0)
        self.activation = nn.ReLU(inplace=True)


    def forward(self,
                x: Tensor) -> Tensor:
        f = self.activation(self.c1(x))
        f = self.activation(self.c2(f))
        f = self.c3(f)
        f = self.activation(torch.add(f, x))

        return f

## Model combines Residual blocks, batch normalization, dropout regularization
class Network(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=1) -> None:
        super(Network, self).__init__()
        self.c1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=16, kernel_size=5, use_batch_norm=False, use_activation=True, stride=1, padding=1),
            ConvBlock(in_channels=16, out_channels=64, kernel_size=3, use_activation=True, stride=1, padding=1)
        )

        self.resblock1 = ResBlock(64)

        self.c2 = ConvBlock(in_channels=64, out_channels=128, use_batch_norm=False, kernel_size=3, stride=1, padding=1)
        self.resblock2 = ResBlock(channels=128)

        self.c3 = ConvBlock(in_channels=128, out_channels=192, kernel_size=3, use_activation=True, stride=1, padding=1)
        self.resblock3 = ResBlock(192)

        self.c4 = ConvBlock(in_channels=192, out_channels=256, kernel_size=3, use_activation=True, stride=1, padding=1)
        self.resblock4 = ResBlock(256)

        self.c5 = ConvBlock(in_channels=256, out_channels=320, kernel_size=3, use_batch_norm=False, use_activation=True, stride=1, padding=1)
        self.resblock5 = ResBlock(320)

        self.c6 = ConvBlock(in_channels=320, out_channels=396, kernel_size=3, use_activation=True, stride=1, padding=1)
        self.resblock6 = ResBlock(396)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=396 * 6 * 6, out_features=num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self,
                x: Tensor) -> [Tensor, Tensor]:
        output = self.c1(x)
        output = self.resblock1(output)

        output = self.c2(self.dropout(output))
        output = self.resblock2(output)
        output = self.pool(output)

        output = self.c3(output)
        output = self.resblock3(output)
        output = self.pool(output)

        output = self.c4(output)
        output = self.resblock4(output)
        output = self.pool(output)

        output = self.c5(self.dropout(output))
        output = self.resblock5(output)
        output = self.pool(output)

        output = self.c6(output)
        output = self.resblock6(output)
        output = self.pool(output)

        output = torch.flatten(output, 1)
        logits = self.fc1(output)

        return logits


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
print(device)


training_imgs = np.load(r'C:\Users\Daryn Bang\Desktop\Dataset\dogs-vs-cats\CatvsDogimgs.npy',
                        allow_pickle=True)
training_labels = np.load(r'C:\Users\Daryn Bang\Desktop\Dataset\dogs-vs-cats\CatvsDoglabels.npy',
                          allow_pickle=True)

X_tensor = torch.tensor(training_imgs, dtype=torch.float32).permute(0, 3, 1, 2)
y_tensor = torch.tensor(training_labels, dtype=torch.float32)


lr = 0.001
num_epochs = 10
batch_size = 16
weight_decay = 0.001

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, stratify=y_tensor, shuffle=True)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def save_checkpoint(state, filename=r""):
    print("-> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def compute_accuracy(model, data_loader, device):
    correct_pred = 0
    num_examples = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1).float()  # Ensure targets are float for BCEWithLogitsLoss

            logits = model(features)
            predicted_probs = torch.sigmoid(logits)
            predicted_labels = (predicted_probs > 0.5).float()  # Convert probabilities to binary labels

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum().item()

    accuracy = float(correct_pred) / num_examples * 100
    return accuracy



model = Network().to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# Train network
def train() -> None:
    for num_epoch, epoch in tqdm(enumerate(range(num_epochs))):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1).float()

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
            if not batch_idx % 100:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(train_loader), loss))

        # Save model
        save_checkpoint(model, r'C:\Users\Daryn Bang\Desktop\Dataset\dogs-vs-cats\DogVsCatModel.pth')

        if (num_epoch+1) % 2 == 0:
            model.eval()
            with torch.set_grad_enabled(False):
                print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
                    epoch + 1, num_epochs,
                    compute_accuracy(model, train_loader, device)))


def test() -> None:
    with torch.no_grad():  # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device)))

def save_model(path) -> None:
    torch.save(model, path)
    print("Model saved complete")


if __name__ == "__main__":
    train()
    test()
    save_model(r'C:\Users\Daryn Bang\Desktop\Dataset\dogs-vs-cats\DogVsCatModel.pth')

# Test accuracy: 86.5%
# Train accuracy: 88.6%
