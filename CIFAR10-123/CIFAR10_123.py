import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
from PIL import Image
from sklearn.manifold import TSNE

class CIFAR10Pair(datasets.CIFAR10):
    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        image = Image.fromarray(image)

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return view1, view2, label


def color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rnd_color_jitter, rnd_gray])


def load_data(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        color_distortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = CIFAR10Pair(root=args.data_path, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10Pair(root=args.data_path, train=False, download=True, transform=transform_test)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,drop_last=True)

    return train_loader, val_loader, test_loader


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = resnet18(pretrained=False)
        self.feature_dim = self.encoder.fc.in_features
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim, bias=True)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return features, projections


def contrastive_loss(z1, z2, temperature=0.1):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)

    mask = (~torch.eye(2 * N, 2 * N, dtype=bool)).to(z.device)
    positives = torch.cat([torch.diag(similarity_matrix, N), torch.diag(similarity_matrix, -N)], dim=0)
    negatives = similarity_matrix[mask].view(2 * N, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * N).long().to(z.device)
    logits = logits / temperature

    return F.cross_entropy(logits, labels)


def train_autoencoder(model, train_loader, val_loader, test_loader, args, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x1, x2, _ in train_loader:
            x1, x2 = x1.to(args.device), x2.to(args.device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, _ in val_loader:
                x1, x2 = x1.to(args.device), x2.to(args.device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = contrastive_loss(z1, z2)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Test
        test_loss = 0.0
        with torch.no_grad():
            for x1, x2, _ in test_loader:
                x1, x2 = x1.to(args.device), x2.to(args.device)
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = contrastive_loss(z1, z2)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}")

        scheduler.step()

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Contrastive Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


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


def train_classifier(autoencoder, train_loader, val_loader, test_loader, args, epochs=10):
    for param in autoencoder.parameters():
        param.requires_grad = False

    classifier = Classifier().to(args.device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, _, labels in train_loader:
            x, labels = x.to(args.device), labels.to(args.device)
            with torch.no_grad():
                _, projections = autoencoder(x)
            outputs = classifier(projections)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, _, labels in val_loader:
                x, labels = x.to(args.device), labels.to(args.device)
                _, projections = autoencoder(x)
                outputs = classifier(projections)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.title('Classifier Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Val Accuracy')
    plt.title('Classifier Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
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
