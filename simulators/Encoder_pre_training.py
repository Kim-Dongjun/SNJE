# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-var.ipynb
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from simulators.Encoder_network import ConvVariationalAutoencoder
import matplotlib.pyplot as plt
import random
import os

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################



parser = argparse.ArgumentParser(description="ConvVAE")
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=50) # 50
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--mode', type=str, default="all", choices=["all", "train", "test"])
parser.add_argument('--dataset', type=str, default="mnist")#, choices=["all", "train", "test"])
parser.add_argument('--device', type=str, default="cuda:0")#, choices=["all", "train", "test"])
parser.add_argument('--input_size', type=int, default=28, help='The size of input image')

# train: train only
# test: test only (load the pretrained model)
parser.add_argument('--num_latent', type=int, default=50, choices=[10,30,50])
parser.add_argument('--num_features', type=int, default=784)
args = parser.parse_args()

# Device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print('Device:', device)

# path
model_name = "ConvVAE_" + str(args.num_latent) + "_" + str(args.num_epochs)
image_dir = "images_"+str(args.dataset)+"/"+model_name
param_dir = "parameters_"+str(args.dataset)+"/"+model_name
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
if not os.path.exists(param_dir):
    os.mkdir(param_dir)

##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
in_channels = 1
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=transforms.ToTensor())

elif args.dataset == 'fasion-mnist':
    train_dataset = datasets.FashionMNIST(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.FashionMNIST(root='data',
                                  train=False,
                                  transform=transforms.ToTensor())

elif args.dataset == 'svhn':
    train_dataset = datasets.SVHN(root='data', split='train',
                                          transform=transforms.ToTensor(),
                                          download=True)

    test_dataset = datasets.SVHN(root='data', split='test',
                                         transform=transforms.ToTensor(), download=True)
    in_channels = 3


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

'''set random seed'''
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

##########################
### Model, COST AND OPTIMIZER
##########################

model = ConvVariationalAutoencoder(args.device, in_channels, args.num_features, args.num_latent)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if args.mode in ["train", "all"]:
    start_time = time.time()
    for epoch in range(args.num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):
            # don't need labels, only the images (features)
            features = features.to(device)
            ### FORWARD AND BACK PROP
            z_mean, z_log_var, encoded, decoded = model(features)

            # cost = reconstruction loss + Kullback-Leibler divergence
            kl_divergence = (0.5 * (z_mean ** 2 +
                                    torch.exp(z_log_var) - z_log_var - 1)).sum()
            pixelwise_bce = F.binary_cross_entropy(decoded, features, reduction='sum')
            cost = kl_divergence + pixelwise_bce

            optimizer.zero_grad()
            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (epoch + 1, args.num_epochs, batch_idx,
                         len(train_loader), cost))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    torch.save(model.state_dict(), param_dir+"/model.pt")

    ##########################
    ### VISUALIZATION
    ##########################
    n_images = 15
    image_width = 28
    fig, axes = plt.subplots(nrows=2, ncols=n_images,
                             sharex=True, sharey=True, figsize=(20, 2.5))
    orig_images = features[:n_images]
    decoded_images = decoded[:n_images]
    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            ax[i].imshow(img[i].detach().to(torch.device('cpu')).reshape((image_width, image_width)), cmap='binary')
        plt.savefig(image_dir+"/reconstructed"+str(i)+".png")
        plt.show()


if args.mode in ["test", "all"]:
    model.load_state_dict(torch.load(param_dir+"/model.pt"))
    model.eval()

    for i in range(10):
        ##########################
        ### RANDOM SAMPLE
        ##########################

        n_images = 10
        rand_features = torch.randn(n_images, args.num_latent).to(device)
        new_images = model.decoder(rand_features)

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:n_images]

        for ax, img in zip(axes, decoded_images):
            ax.imshow(img.detach().to(torch.device('cpu')).reshape((image_width, image_width)), cmap='binary')
        plt.savefig(image_dir+"/generated_"+str(i)+".png")
        plt.show()