import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load MNIST dataset
def load_data(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder: 3 layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, latent_dim),
        )

        # Decoder: 3 layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# Define the Classifier model
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

def train_autoencoder(model, train_loader, val_loader, test_loader, args, epochs=10):
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, _ in train_loader:
            images = images.to(args.device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += ((recon > 0.5) == (images > 0.5)).sum().item()
            train_total += torch.numel(images)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(args.device)
                recon = model(images)
                loss = criterion(recon, images)
                val_loss += loss.item()
                val_correct += ((recon > 0.5) == (images > 0.5)).sum().item()
                val_total += torch.numel(images)

        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(args.device)
                recon = model(images)
                loss = criterion(recon, images)
                test_loss += loss.item()
                test_correct += ((recon > 0.5) == (images > 0.5)).sum().item()
                test_total += torch.numel(images)

        # Averages
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        test_losses.append(test_loss / len(test_loader))

        train_accs.append(100 * train_correct / train_total)
        val_accs.append(100 * val_correct / val_total)
        test_accs.append(100 * test_correct / test_total)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    # === Plot Loss ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # === Plot Accuracy ===
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Reconstruction Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_classifier(autoencoder, classifier, train_loader, val_loader, test_loader, args, epochs=10):
    autoencoder.eval()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(epochs):
        # Train
        classifier.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                latent = autoencoder.encoder(images)
            outputs = classifier(latent)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100. * correct / total)

        # Validation
        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                latent = autoencoder.encoder(images)
                outputs = classifier(latent)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100. * val_correct / val_total)

        # Test
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                latent = autoencoder.encoder(images)
                outputs = classifier(latent)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += preds.eq(labels).sum().item()
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(100. * test_correct / test_total)

        print(f"Epoch [{epoch+1}/{epochs}], Train Acc: {train_accs[-1]:.2f}%, Val Acc: {val_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%")

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accs, label='Val Accuracy')
    plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_reconstructions(autoencoder, dataloader, device, num_images=5):
    # Get a batch of test images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructions = autoencoder(images)

    # Move to CPU
    images = images.cpu().squeeze(1)
    reconstructions = reconstructions.cpu().squeeze(1)

    # Plot results
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title("Original")
        ax.axis('off')

        # Reconstructed images
        ax = plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.title("Reconstructed")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_tsne(model, dataloader, device):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to over over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    '''
    model.eval()
    
    images_list = []
    labels_list = []
    latent_list = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            #approximate the latent space from data
            latent_vector = model.encode(images)
            
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)
    
    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, init='random',random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig('latent_tsne.png')
    plt.show()
    
    #plot image domain tsne
    tsne_image = TSNE(n_components=2, init='random',random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)  
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.savefig('image_tsne.png')
    plt.show()
