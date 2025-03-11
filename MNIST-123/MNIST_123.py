import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


# **Data Augmentation for Contrastive Learning**
transform_simclr = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


# **Load MNIST Dataset**
def load_data(args):
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_simclr)
    test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_simclr)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# **SimCLR Encoder (No Decoder)**
class SimCLR_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(SimCLR_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (1, 28, 28) -> (32, 28, 28)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 28, 28) -> (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 14, 14) -> (128, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (128, 7, 7) -> 6272
            nn.Linear(128 * 7 * 7, latent_dim)  # **Ensure latent_dim = 128**
        )

    def forward(self, x):
        return self.encoder(x)  # 128D latent representation


# **NT-Xent Contrastive Loss**
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute cosine similarity
        similarities = torch.mm(z_i, z_j.T) / self.temperature
        labels = torch.arange(z_i.size(0)).to(z_i.device)

        # Compute contrastive loss
        loss = F.cross_entropy(similarities, labels)
        return loss


# **Train Encoder with Contrastive Learning**
def train_simclr(encoder, train_loader, args, epochs=10):
    encoder.train()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    loss_fn = NTXentLoss(temperature=0.5)

    for epoch in range(epochs):
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(args.device)

            # Generate two augmented versions
            aug1 = images + 0.05 * torch.randn_like(images)  # Add noise
            aug2 = images + 0.05 * torch.randn_like(images)  # Add noise

            # Encode both augmented images
            z_i = encoder(aug1)
            z_j = encoder(aug2)

            # Compute contrastive loss
            loss = loss_fn(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")


# **Classifier Model**
class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# **Train Classifier on Encoded Representations**
def train_classifier(encoder, classifier, train_loader, val_loader, args, epochs=10):
    encoder.eval()  # Freeze encoder
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)

            with torch.no_grad():
                latent_vectors = encoder(images)  # Get 128D representations

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

    # Train Encoder with SimCLR Contrastive Learning
    encoder = SimCLR_Encoder(args.latent_dim).to(args.device)
    train_simclr(encoder, train_loader, args, epochs=10)

    # Train Classifier on Encoded Features
    classifier = Classifier(args.latent_dim).to(args.device)
    train_classifier(encoder, classifier, train_loader, val_loader, args, epochs=10)
