import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
from MNIST__121 import MNIST_121
from CIFAR10__121 import CIFAR10_121
from MNIST__122 import MNIST_122
from CIFAR10__122 import CIFAR10_122
from MNIST__123 import MNIST_123
from CIFAR10__123 import CIFAR10_123


NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="../data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='Encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')

    # New arguments
    parser.add_argument('--mnist', action='store_true', default=False, help='Use MNIST dataset')
    parser.add_argument('--version', type=int, choices=[121, 122, 123], required=True, help='Model version to run: 121, 122, or 123')
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)

    if args.mnist:

        if args.version == 121:
            train_loader, val_loader, test_loader = MNIST_121.load_data(args)
            autoencoder = MNIST_121.Autoencoder(args.latent_dim).to(args.device)
            classifier = MNIST_121.Classifier(args.latent_dim).to(args.device)
            MNIST_121.train_autoencoder(autoencoder, train_loader, val_loader,test_loader, args, epochs=30)
            MNIST_121.train_classifier(autoencoder, classifier, train_loader, val_loader,test_loader, args, epochs=30)
        
        elif args.version == 122:
            train_loader, val_loader, test_loader = MNIST_122.load_data(args)
            encoder_model = MNIST_122.Encoder(args.latent_dim).to(args.device)
            classifier_model = MNIST_122.Classifier(args.latent_dim).to(args.device)
            MNIST_122.train_encoder_classifier(encoder_model, classifier_model, train_loader, val_loader,test_loader,args, epochs=30)
        
        elif args.version == 123:
            train_loader, val_loader, test_loader = MNIST_123.load_data(args)
            autoencoder = MNIST_123.Autoencoder(latent_dim=args.latent_dim).to(args.device)
            MNIST_123.train_autoencoder(autoencoder, train_loader, val_loader, test_loader,args,epochs=30)
            MNIST_123.train_classifier(autoencoder, train_loader, val_loader, test_loader, args,epochs=30)
        
        else:
            raise ValueError("Invalid MNIST version selected.")
    else:
        
        if args.version == 121:
            train_loader, val_loader, test_loader = CIFAR10_121.load_data(args)
            autoencoder = CIFAR10_121.Autoencoder(latent_dim=args.latent_dim).to(args.device)
            autoencoder = CIFAR10_121.train_autoencoder(autoencoder, train_loader,val_loader,test_loader,args.device, epochs=30)
            CIFAR10_121.train_classifier(autoencoder, train_loader,val_loader,test_loader,args.device, epochs=30)
        
        elif args.version == 122:
            train_loader, val_loader, test_loader = CIFAR10_122.load_data(args)
            encoder_model = CIFAR10_122.Encoder(args.latent_dim).to(args.device)
            classifier_model = CIFAR10_122.Classifier(args.latent_dim).to(args.device)
            CIFAR10_122.train_encoder_classifier(encoder_model, classifier_model, train_loader, val_loader,test_loader, args, epochs=30)
        
        elif args.version == 123:
            train_loader, val_loader, test_loader = CIFAR10_123.load_data(args)
            autoencoder = CIFAR10_123.Autoencoder(latent_dim=args.latent_dim).to(args.device)
            CIFAR10_123.train_autoencoder(autoencoder, train_loader, val_loader, test_loader,args,epochs=30)
            classifier = CIFAR10_123.train_classifier(autoencoder, train_loader, val_loader, test_loader, args,epochs=30)
        else:
            raise ValueError("Invalid CIFAR-10 version selected.")



