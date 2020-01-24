import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import parser4acgan
# import data_testing

num_channels = 3
z_size = 100
num_gpus = 1

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

args = parser4acgan.arg_parse()

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

netG = Generator(num_gpus).to(device)

resume = 'ACGAN.pth.tar'
netG.load_state_dict(torch.load(resume))

# seed = 999
# torch.manual_seed(seed)

# fixed_noise = torch.randn(10, z_size, 1, 1, device=device) 
# torch.save(fixed_noise, "noise_ac.pt")
fixed_noise = torch.load('noise_ac.pt')

smile_label = torch.full((10, 1, 1, 1), 1, device=device)
not_smile_label = torch.full((10, 1, 1, 1), 0, device=device)
fixed_noise_smile = torch.cat((fixed_noise, smile_label), dim=1)
fixed_noise_not_smile = torch.cat((fixed_noise, not_smile_label), dim=1)
compare_set = torch.cat((fixed_noise_smile, fixed_noise_not_smile), dim=0)

fake = netG(compare_set).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True, nrow=10) ,(1,2,0)))
save_img_dir = args.save_img_dir + '/fig2_2.png'
plt.savefig(save_img_dir)


