import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
import MNIST-121/MNIST_121
import CIFAR10-121/CIFAR10_121
import MNIST-122/MNIST_122
import CIFAR10-122/CIFAR10_122

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
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)

    if args.mnist and args.self_supervised:
        MNIST_121.main(args)
    elif (not args.mnist) and args.self_supervised:
        CIFAR10_121.main(args)
    elif args.mnist and (not args.self-supervised):
        MNIST_122.main(args)
    else:
        CIFAR10_122.main(args)

