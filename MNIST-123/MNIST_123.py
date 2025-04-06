import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class MNISTPair(datasets.MNIST):
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform:
            view1 = self.transform(img)
            view2 = self.transform(img)
            
        return view1, view2, label

def load_data(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNISTPair(root=args.data_path, train=True, download=True, transform=transform_train)
    test_dataset = MNISTPair(root=args.data_path, train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    return train_loader, val_loader, test_loader

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x).squeeze()
        projections = self.projector(features)
        return features, projections

def contrastive_loss(z1, z2, temperature=0.1):
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    positives = torch.cat([torch.diag(sim_matrix, z1.size(0)), 
                         torch.diag(sim_matrix, -z1.size(0))])
    negatives = sim_matrix[~torch.eye(2*z1.size(0), dtype=bool, device=z.device)].view(2*z1.size(0), -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*z1.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)

def train_autoencoder(model, train_loader, val_loader, test_loader, args, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for x1, x2, _ in train_loader:
            x1, x2 = x1.to(args.device), x2.to(args.device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x1, x2, _ in val_loader:
                x1, x2 = x1.to(args.device), x2.to(args.device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = contrastive_loss(z1, z2)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # ---- Test ----
        total_test_loss = 0.0
        with torch.no_grad():
            for x1, x2, _ in test_loader:
                x1, x2 = x1.to(args.device), x2.to(args.device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = contrastive_loss(z1, z2)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        scheduler.step()

    # ---- Plotting ----
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.title('Contrastive Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def train_classifier(autoencoder, train_loader, val_loader, test_loader, args, epochs=15):
    classifier = Classifier().to(args.device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        # === Train ===
        classifier.train()
        train_loss, train_correct = 0, 0
        for x, _, labels in train_loader:
            x, labels = x.to(args.device), labels.to(args.device)
            with torch.no_grad():
                _, projections = autoencoder(x)
            logits = classifier(projections)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_acc = 100 * train_correct / len(train_loader.dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # === Validation ===
        classifier.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x, _, labels in val_loader:
                x, labels = x.to(args.device), labels.to(args.device)
                _, projections = autoencoder(x)
                logits = classifier(projections)
                val_loss += criterion(logits, labels).item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
        val_acc = 100 * val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        # === Test ===
        test_loss, test_correct = 0, 0
        with torch.no_grad():
            for x, _, labels in test_loader:
                x, labels = x.to(args.device), labels.to(args.device)
                _, projections = autoencoder(x)
                logits = classifier(projections)
                test_loss += criterion(logits, labels).item()
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
        test_acc = 100 * test_correct / len(test_loader.dataset)
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # === Plot Accuracy ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    # === Plot Loss ===
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_tsne(model, dataloader, device):
    '''
    model - torch.nn.Module subclass (e.g. encoder or projection model)
    dataloader - dataloader that yields (view1, view2, label)
    device - 'cuda' or 'cpu'
    '''
    model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for view1, _, labels in dataloader:
            view1, labels = view1.to(device), labels.to(device)
            _,latent_vector = model(view1)

            images_list.append(view1.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, init='random', random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig('latent_tsne.png')
    plt.show()

    # Plot TSNE for image space
    tsne_image = TSNE(n_components=2, init='random', random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.savefig('image_tsne.png')
    plt.show()
