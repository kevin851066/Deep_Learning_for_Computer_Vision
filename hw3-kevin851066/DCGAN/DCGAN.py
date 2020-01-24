import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms
import torchvision.utils as vutils
import data

import matplotlib.pyplot as plt

seed = 999
torch.manual_seed(seed)

data_root = "../hw3_data/face/train/"
label_root = "../hw3_data/face/train.csv"
save_dir = 'log/'
n_workers = 4
batch_size = 128
image_size = 64
num_channels = 3
z_size = 100  # Size of z latent vector (i.e. size of generator input)
gf_size = 64  # Size of feature maps in generator
df_size = 64  # Size of feature maps in discriminator
num_epochs = 50
lr = 0.0002
beta1 = 0.5
num_gpus = 1

train_loader = torch.utils.data.DataLoader(data.Data(data_root, label_root),
                                           batch_size=batch_size,
                                           num_workers=n_workers,
                                           shuffle=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_gpus):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_size, gf_size*8, 4, 1, 0, bias=False), # (in_channels, out_channels, kernel_size, stride, padding)
            nn.BatchNorm2d(gf_size*8),
            nn.ReLU(inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(gf_size*8, gf_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size*4),
            nn.ReLU(inplace=True) 
            )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(gf_size*4, gf_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size*2),
            nn.ReLU(inplace=True) 
            )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(gf_size*2, gf_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_size),
            nn.ReLU(inplace=True)
            )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d( gf_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            )

    def forward(self, z):
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)

        return z

class Discriminator(nn.Module):
    def __init__(self, num_gpus):
        super(Discriminator, self).__init__()
        self.num_gpus = num_gpus
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, df_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(df_size, df_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 2),
            nn.LeakyReLU(0.2, inplace=True)
            )   
        self.layer3 = nn.Sequential(
            nn.Conv2d(df_size * 2, df_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 4),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(df_size * 4, df_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_size * 8),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(df_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, img):
        ft = self.layer1(img)
        ft = self.layer2(ft)
        ft = self.layer3(ft)
        ft = self.layer4(ft)
        prob = self.layer5(ft)

        return prob

netG, netD = Generator(num_gpus).to(device), Discriminator(num_gpus).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# print("G: ", netG)
# print("D: ", netD)

criterion = nn.BCELoss()
fixed_noise = torch.randn(32, z_size, 1, 1, device=device) # 64: image_size     # size:(64*z_size*1*1)

real_label, fake_label = 1, 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # betas ??
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_loss, D_loss = [], []
iters = 0

print('Start training: ')

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0): # data: (128,3,64,64)
        # Disçriminator
        netD.zero_grad()
        b_size = data.size()[0] # b_size = 128
        label = torch.full((b_size,), real_label, device=device)
        output = netD(data.to(device)).view(-1) # output: (128)

        Dloss4real = criterion(output, label) # label: all 1
        Dloss4real.backward()
        # print("output: ", output)
        # print("output: ", output.shape)
        D_x = output.mean().item()

        noise = torch.randn(b_size, z_size, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        Dloss4fake = criterion(output, label) # label: all 0
        Dloss4fake.backward()
        D_G_z1 = output.mean().item()

        Dloss = Dloss4real + Dloss4fake
        optimizerD.step()
        # Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        Gloss = criterion(output, label) # label: all 1
        Gloss.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     Dloss.item(), Gloss.item(), D_x, D_G_z1, D_G_z2))

        G_loss.append(Gloss.item())
        D_loss.append(Dloss.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu() # fake:(32,3,64,64)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

save_model(netG, os.path.join(save_dir, 'DCGAN.pth.tar'))


# Plot the fake images from the last epoch
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig("fig1_2.png")
# plt.show()





