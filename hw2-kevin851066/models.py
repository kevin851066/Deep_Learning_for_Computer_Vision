import torch
import torch.nn as nn
import torchvision.models as models

import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from resnet import resnet101, resnet152


class Baseline_model(nn.Module):

    def __init__(self, args):
        super(Baseline_model, self).__init__()  

        ''' declare layers used in this network'''
        # preprocess
        self.resnet18 = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(self.resnet18.children())[:-2])

        # first block
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False) 
        self.relu1 = nn.ReLU()
        
        # second block
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False) 
        self.relu2 = nn.ReLU()
        
        # third block
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False) 
        self.relu3 = nn.ReLU()

        # forth block
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False) 
        self.relu4 = nn.ReLU()

        # fifth block
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False) 
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):
        x = self.base(img)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x)
        return x


model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class resnet_backbone(nn.Module):
    def __init__(self, num_class = 9, input_channel = 3, output_stride=16, layer=101):
        super(resnet_backbone, self).__init__()

        if layer == 101:
            self.resnet = resnet101(pretrained=True, output_stride=output_stride)
        elif layer == 152:
            self.resnet = resnet152(pretrained=True, output_stride=output_stride)

        self.conv1 = self.resnet.conv1

    def forward(self, x):

        x = self.conv1(x) #1, 320*320
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) #4, 80*80

        x = self.resnet.layer1(x)
        low_feat = x
        x = self.resnet.layer2(x) #8, 40*40
        x = self.resnet.layer3(x) #16, 20*20
        x = self.resnet.layer4(x) #32, 10*10

        return x, low_feat

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x

#ResNet101 as backbone 
class deeplabv3p(nn.Module):
    def __init__(self, input_channel=3, num_class=9, output_stride=16, layer=101):

        super(deeplabv3p, self).__init__()
        self.feature_extractor = resnet_backbone(num_class=num_class, input_channel=input_channel, output_stride=output_stride, layer=layer)
        self.output_stride = output_stride

        #ASPP
        rates = [6, 12, 18]
        self.aspp1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.aspp1_bn = nn.BatchNorm2d(256)

        self.aspp2 = ASPP(2048, 256, rate=rates[0])
        self.aspp3 = ASPP(2048, 256, rate=rates[1])
        self.aspp4 = ASPP(2048, 256, rate=rates[2])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # (H, W) = (1, 1)
                                        nn.Conv2d(2048, 256, 1, stride=1),
                                        nn.BatchNorm2d(256)
                                        )

        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)

        #channel reduction to 48.
        self.conv2 = nn.Conv2d(256, 48, 1)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, num_class, kernel_size=1, stride=1))


    def forward(self, x):

        x, low_level_features = self.feature_extractor(x)
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)

        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)

        x = F.interpolate(x, scale_factor= self.output_stride//4, mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, scale_factor= self.output_stride//(self.output_stride//4), mode='bilinear', align_corners=True)

        return x


