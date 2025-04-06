import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# **Load CIFAR-10 Dataset**
def load_data(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 128 * 4 * 4)

        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(z.size(0), 128, 4, 4)
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        return torch.tanh(self.dec_conv3(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class Classifier(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return F.log_softmax(self.fc3(z), dim=1)


def train_autoencoder(autoencoder, train_loader, val_loader, test_loader, device, epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(epochs):
        # === Training ===
        autoencoder.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += ((outputs > 0.5) == (images > 0.5)).sum().item()
            total += torch.numel(images)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # === Validation ===
        autoencoder.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = autoencoder(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
                val_correct += ((outputs > 0.5) == (images > 0.5)).sum().item()
                val_total += torch.numel(images)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * val_correct / val_total)

        # === Test ===
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = autoencoder(images)
                loss = criterion(outputs, images)
                test_loss += loss.item()
                test_correct += ((outputs > 0.5) == (images > 0.5)).sum().item()
                test_total += torch.numel(images)
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(100 * test_correct / test_total)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    # === Plot Loss ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # === Plot Accuracy ===
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return autoencoder


def train_classifier(autoencoder, train_loader, val_loader, test_loader, device, epochs=20):
    # Freeze the autoencoder's parameters
    for param in autoencoder.parameters():
        param.requires_grad = False

    classifier = Classifier(latent_dim=128).to(device)  
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    losses = []
    accuracies = []
    val_accuracies = []  # Track validation accuracy
    test_accuracies = []  # Track test accuracy
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images = images.to(device)  
            optimizer.zero_grad()
            with torch.no_grad():
                latent = autoencoder.encode(images) 
            outputs = classifier(latent)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # Validation loop
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                latent = autoencoder.encode(images)
                outputs = classifier(latent)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        # Test loop (evaluating on test set)
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                latent = autoencoder.encode(images)
                outputs = classifier(latent)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # Plotting training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax1.set_title('Classifier Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NLL Loss')

    ax2.plot(accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy', linestyle='--')
    ax2.plot(test_accuracies, label='Test Accuracy', linestyle='-.')
    ax2.set_title('Classifier Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def visualize_reconstructions(autoencoder, dataloader, device, num_images=5):
    # Get a batch of test images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructions = autoencoder(images)
    
    # Denormalize images
    def denormalize(img):
        img = img.cpu().numpy()
        img = img * 0.5 + 0.5  # Reverse Normalize(-0.5, 0.5)
        return np.transpose(img, (1, 2, 0))
    
    # Plot results
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(denormalize(images[i]))
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Reconstructed images
        ax = plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(denormalize(reconstructions[i]))
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
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
