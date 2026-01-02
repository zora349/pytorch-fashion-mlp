# main.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP
from train import train_epoch, test_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    train_set = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Model
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Train
    epochs = 15
    best_acc = 0.0

    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(
            train_loader, model, loss_fn, optimizer, device
        )
        te_loss, te_acc = test_epoch(
            test_loader, model, loss_fn, device
        )

        print(
            f"Epoch {ep+1}/{epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"test_loss={te_loss:.4f} acc={te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ Saved best model (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()

