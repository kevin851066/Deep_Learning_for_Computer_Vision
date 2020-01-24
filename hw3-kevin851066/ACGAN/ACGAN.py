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
batch_size = 100
image_size = 64
num_channels = 3
num_classes = 2
z_size = 100  # Size of z latent vector (i.e. size of generator input)
num_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999  # (beta1, beta2) are the hyper-parameters of Adam optim 
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
        self.layer0 = nn.Sequential(
            nn.Linear(z_size+1, 384),
            # nn.ReLU(inplace=True)
            )
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False), # (in_channels, out_channels, kernel_size, stride, padding)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True) 
            )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True) 
            )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True) 
            )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(24, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, z_size+1)
        fc = self.layer0(input)
        fc = fc.view(-1, 384, 1, 1)
        z = self.layer1(fc)
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
            nn.Conv2d(num_channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )   
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5)
            )
        self.dis_fc = nn.Linear(512*5*5, 1)
        self.ac_fc = nn.Linear(512*5*5, num_classes)
        # self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, img): # img: (batch_size, 3, 64, 64)
        ft = self.layer1(img) # (batch_size, 16, 32, 32)
        ft = self.layer2(ft) # (batch_size, 32, 30, 30)

        ft = self.layer3(ft) # (batch_size, 64, 15, 15)
        ft = self.layer4(ft) # (batch_size, 128, 13, 13)
        ft = self.layer5(ft) # (batch_size, 256, 7, 7)
        ft = self.layer6(ft) # (batch_size, 512, 5, 5)

        flat = ft.view(-1, 512*5*5)
        dis_prob = self.sigmoid( self.dis_fc(flat) ) # (batch_size, 1)
        ac_prob = self.ac_fc(flat) # (batch_size, 1)
        
        return dis_prob, ac_prob

netG, netD = Generator(num_gpus).to(device), Discriminator(num_gpus).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# print("G: ", netG)
# print("D: ", netD)

dis_criterion = nn.BCELoss()
ac_criterion = nn.CrossEntropyLoss()

fixed_noise = torch.randn(10, z_size, 1, 1, device=device) 
smile_label = torch.full((10, 1, 1, 1), 1, device=device)
not_smile_label = torch.full((10, 1, 1, 1), 0, device=device)
fixed_noise_smile = torch.cat((fixed_noise, smile_label), dim=1)
fixed_noise_not_smile = torch.cat((fixed_noise, not_smile_label), dim=1)
compare_set = torch.cat((fixed_noise_smile, fixed_noise_not_smile), dim=0)

real_label, fake_label = 1, 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # betas ??
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_loss, D_loss = [], []
iters = 0

print('Start training: ')

for epoch in range(num_epochs):
    for i, (data, smile) in enumerate(train_loader, 0): # data: (100,3,64,64), smile: (100, 1)
        # Disçriminator: maximize log(D(x)) + log(1 - D(G(z)))
        ac_label = smile.squeeze().to(device)

        netD.zero_grad()
        b_size = data.size()[0] 
        dis_label = torch.full((b_size,), real_label, device=device)
        dis_output, ac_output = netD(data.to(device)) # output:
        Dloss4real = dis_criterion(dis_output, dis_label) # real_label: all 1
        
        Acloss4real = ac_criterion(ac_output, ac_label) # ac_output: (100, 2) ac_label: (100)
        TotalLoss4real = Dloss4real + Acloss4real
        TotalLoss4real.backward()
        D_x = dis_output.mean().item()

        noise = torch.randn(b_size, z_size, 1, 1, device=device)
        fake_smile_label = torch.randint(low=0, high=2, size=(b_size, 1, 1, 1)).to(device)
        noise_conditioned_on_label = torch.cat((noise, fake_smile_label.float()), dim=1)

        fake = netG(noise_conditioned_on_label) # (100, 3, 64, 64)

        dis_label = torch.full((b_size,), fake_label, device=device)
        dis_output, ac_output = netD(fake.detach()) 

        fake_smile_label = fake_smile_label.view(b_size)
        Dloss4fake = dis_criterion(dis_output, dis_label) # label: all 0
        Acloss4fake = ac_criterion(ac_output, fake_smile_label)
        TotalLoss4fake = Dloss4fake + Acloss4fake
        TotalLoss4fake.backward()
        
        D_G_z1 = dis_output.mean().item()

        Dloss = TotalLoss4real + TotalLoss4fake
        optimizerD.step()

        # Generator: maximize log(D(G(z)))
        netG.zero_grad()
        dis_label = torch.full((b_size,), real_label, device=device)
        dis_output, ac_output = netD(fake) 
        Gloss = dis_criterion(dis_output, dis_label) + ac_criterion(ac_output, fake_smile_label) 
        Gloss.backward()
        D_G_z2 = dis_output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     Dloss.item(), Gloss.item(), D_x, D_G_z1, D_G_z2))

        G_loss.append(Gloss.item())
        D_loss.append(Dloss.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                result = netG(compare_set).detach().cpu() 
                # not_smile_fake = netG(fixed_noise_not_smile).detach().cpu() # fake:(64,3,64,64)

            img_list.append(vutils.make_grid(result, padding=2, normalize=True, nrow=10))
            plt.figure(figsize=(15,15))
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            plt.savefig("fig2_2_{}.png".format(epoch))
            plt.close()
            # print("test: ", img_list[-1].shape)
        iters += 1

save_model(netG, os.path.join(save_dir, 'ACGAN.pth.tar'))

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig("fig2_2.png")