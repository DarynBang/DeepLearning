import torch

def save_checkpoint(state, filename=r""):
    print("-> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def compute_accuracy(model, data_loader):
    correct_pred: int = 0
    num_examples: int = 0
    for features, targets in data_loader:
        features = features
        targets = torch.argmax(targets, dim=1).long()
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum().item()
    return float(correct_pred)/num_examples * 100


