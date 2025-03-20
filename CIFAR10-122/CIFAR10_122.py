import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import argparse

# Load CIFAR-10 dataset
def load_data(args):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1,1]
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

# Encoder Model with Batch Normalization
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (3,32,32) → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (64,16,16) → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=8)              # (128,8,8) → (latent_dim,1,1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.encoder(x).view(batch_size, -1)  # Flatten to (batch_size, latent_dim)

# Classifier Model with Dropout
class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout to prevent overfitting
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Train Encoder + Classifier with Learning Rate Scheduler
def train_encoder_classifier(encoder, classifier, train_loader, val_loader, args, epochs=15):
    encoder.train()
    classifier.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay

    # Learning rate scheduler
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=0.1)
    scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            
            latent_vectors = encoder(images)
            outputs = classifier(latent_vectors)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_encoder.step()
            optimizer_classifier.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)  # Average loss

        # Validation
        encoder.eval()
        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                latent_vectors = encoder(images)
                outputs = classifier(latent_vectors)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)  # Average loss

        encoder.train()
        classifier.train()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler_encoder.step()
        scheduler_classifier.step()

# Main function
def main(args):
    train_loader, val_loader, test_loader = load_data(args)
    
    encoder = Encoder(args.latent_dim).to(args.device)
    classifier = Classifier(args.latent_dim).to(args.device)

    train_encoder_classifier(encoder, classifier, train_loader, val_loader, args, epochs=15)

