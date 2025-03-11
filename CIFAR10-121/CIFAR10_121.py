import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import argparse


# **Load CIFAR-10 Dataset**
def load_data(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# **Autoencoder (Encoder Outputs Only 128D)**
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()

        # **Encoder (Conv2d → BatchNorm → ReLU)**
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (3, 32, 32) -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 16, 16) -> (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (128, 8, 8) -> (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),  # (256, 4, 4) -> 4096
            nn.Linear(256 * 4 * 4, latent_dim)  # **Ensures output is 128D**
        )

        # **Decoder (Mirrors the encoder)**
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),  # **128D -> 4096**
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # Reshape to (256, 4, 4)

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256, 4, 4) -> (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 8, 8) -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 16, 16) -> (3, 32, 32)
            nn.Tanh()  # Output normalized to [-1,1]
        )

    def forward(self, x):
        z = self.encoder(x)  # **Ensures 128D output**
        x_recon = self.decoder(z)  # **Reconstruct image**
        return x_recon


# **Classifier (Uses Encoded Features)**
class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),  # (128) -> (64)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # (64) -> (10)
        )

    def forward(self, x):
        return self.fc(x)  # Logits (use CrossEntropyLoss)


# **Train Autoencoder**
def train_autoencoder(model, train_loader, val_loader, args, epochs=30):
    model.train()
    criterion = nn.L1Loss()  # **L1 Loss for sharper images**
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(args.device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(args.device)
                recon = model(images)
                loss = criterion(recon, images)
                val_loss += loss.item()
        model.train()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")


# **Train Classifier**
def train_classifier(autoencoder, classifier, train_loader, val_loader, args, epochs=30):
    autoencoder.eval()  # Freeze encoder
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            with torch.no_grad():
                latent_vectors = autoencoder.encoder(images)  # Get 128D features

            outputs = classifier(latent_vectors)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")


# **Main Function**
def main(args):
    train_loader, val_loader, test_loader = load_data(args)

    # Train Autoencoder
    autoencoder = Autoencoder(args.latent_dim).to(args.device)
    train_autoencoder(autoencoder, train_loader, val_loader, args, epochs=30)

    # Train Classifier with Frozen Encoder
    classifier = Classifier(args.latent_dim).to(args.device)
    train_classifier(autoencoder, classifier, train_loader, val_loader, args, epochs=30)
