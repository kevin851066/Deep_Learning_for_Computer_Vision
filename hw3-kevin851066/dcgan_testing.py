import parser4dcgan
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

gf_size = 64  # Size of feature maps in generator
df_size = 64
num_channels = 3
z_size = 100
num_gpus = 1

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

args = parser4dcgan.arg_parse()


device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

netG = Generator(num_gpus).to(device)

resume = 'DCGAN.pth.tar'
netG.load_state_dict(torch.load(resume))

# seed = 999
# torch.manual_seed(seed)
# fixed_noise = torch.randn(32, z_size, 1, 1, device=device) # 64: image_size     # size:(64*z_size*1*1)
# torch.save(fixed_noise, "noise_dc.pt")
fixed_noise = torch.load('noise_dc.pt')
fake = netG(fixed_noise).detach().cpu()

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True) ,(1,2,0)))
save_img_dir = args.save_img_dir + '/fig1_2.png'
plt.savefig(save_img_dir)


