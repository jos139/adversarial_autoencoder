from tqdm import tqdm

from keras.datasets import mnist
import numpy as np
import math

import torch
from torch import tensor
import torch.optim as optim
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return self.transform(x).float(), y

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Reshape(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 28, 28)
    
    
# Encoder
class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
        )
        self.layer4 = nn.Conv2d(64, 2, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h
    
# Decoder
class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(2, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            Flatten(),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h
    
# Discriminator
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.flatten = Flatten()
        self.layer1 = nn.Linear(2, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        return F.sigmoid(self.layer3(h))
    

def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, num_labels), num_labels)
    return z

def swiss_roll(batchsize, ndim, num_labels):
    def sample(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.0 * np.sqrt(uni)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, num_labels), num_labels)
    return z

def inv(t, n=1):
    if t > 1/2:
        return (2*t - 1)**(1/(2*n + 1))
    else:
        return -(1 - 2*t)**(1/(2*n + 1))
    
def linear(t):
    if t < 1/2:
        return -np.sqrt(1 - 2*t)
    else:
        return np.sqrt(2*t - 1)
    
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255
    x_train = x_train.astype(np.float64)
    
    train_dataset = Dataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
    
    #torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    encoder, decoder, discriminator = Encoder(), Decoder(), Discriminator()

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    # Set optimizators
    optim_enc = optim.Adam(encoder.parameters(), lr=0.0006)
    optim_dec = optim.Adam(decoder.parameters(), lr = 0.0006)
    optim_dis = optim.Adam(discriminator.parameters(), lr= 0.0008)
    optim_gen = optim.Adam(encoder.parameters(), lr = 0.0008)
    
    encoder.train()
    decoder.train()

    for i in range(10):
        for j, batch in enumerate(trainloader):
            inputs, targets = batch
            inputs = inputs.to(device)

            # train encoder-decoder
            encoder.zero_grad()
            decoder.zero_grad()
            z_sample = encoder(inputs)
            X_sample = decoder(z_sample)
            recon_loss = F.binary_cross_entropy(X_sample, inputs.view(-1, 784))
            recon_loss.backward()
            optim_enc.step()
            optim_dec.step()

        print("[{:d}, recon_loss : {:.3f}]".format(i, recon_loss.data))
        
        
    discriminator.train()

    for i in range(100):
        for j, batch in enumerate(trainloader):
            inputs, targets = batch
            inputs = inputs.to(device)

            # train encoder-decoder
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()
            z_sample = encoder(inputs)

            for param in encoder.parameters():
                print("----------------")
                print(param.max())
                print(param.min())

            X_sample = decoder(z_sample)
            recon_loss = F.binary_cross_entropy(X_sample, inputs.view(-1, 784))
            recon_loss.backward()
            optim_enc.step()
            optim_dec.step()

            # train discriminator
            encoder.eval() 
            discriminator.zero_grad()

            #z_real = torch.randn(1000).view(500,2) * 5
            u = np.random.rand(1000)
            sample = np.vectorize(linear)(u)
            z_real = torch.from_numpy(sample.reshape(-1,2)).float()
            z_real = z_real.to(device)
            z_fake = encoder(inputs)

            D_real, D_fake = discriminator(z_real), discriminator(z_fake)
            D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss.backward()
            optim_dis.step()

            # train generator
            encoder.train()
            encoder.zero_grad()
            z_fake = encoder(inputs)
            D_fake = discriminator(z_fake)

            G_loss = -torch.mean(torch.log(D_fake))
            G_loss.backward()
            optim_gen.step()
        print("[{:d}, recon_loss : {:.3f}, D_loss : {:.3f}, G_loss : {:.3f}]".format(i, recon_loss.data, D_loss.data, G_loss.data))
        encoder.eval()
        colors = ["#880000", "#0000FF", "#88FF88", "#FFFF00", "#00FF00", "#0088FF", "#FF00FF", "#880088", "#FF8800", "#FF0000", "#008800"]
        z = encoder(inputs).cpu().detach().numpy().reshape(-1,2)
        lists = []
        for k in range(10):
            lists.append([])
        for k in range(500):
            lists[targets[k]].append(z[k])
        plt.clf()
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        for k in range(10):
            plt.scatter(np.array(lists[k])[:,0], np.array(lists[k])[:,1], c=colors[k])
        plt.savefig("images1/img{}.png".format(str(i).zfill(3)))