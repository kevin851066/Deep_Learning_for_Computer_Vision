import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from .utils import denormalize, normalize


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # self.backbone = models.resnet50(pretrained=True)
        # self.backbone = models.resnet34(pretrained=True)
        self.backbone = models.resnet18(pretrained=True)
        # self.model = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x_pool = self.backbone.avgpool(x).view(x.shape[0], -1)

        return x_pool, x
        # return self.model(input).view(input.shape[0], -1)


class Classifier(nn.Module):
    def __init__(self, feature_size, class_num):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=feature_size//8),
            nn.BatchNorm1d(num_features=feature_size//8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_size//8, out_features=feature_size//16),
            nn.BatchNorm1d(num_features=feature_size//16),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_size//16, out_features=class_num),
        )

    def forward(self, input):
        return self.classifier(input)


class ReID_Model(nn.Module):
    def __init__(self, class_num=71, feature_size=512, seg_n_classes=4, seg_extractor=None):
        super(ReID_Model, self).__init__()
        if seg_extractor is not None:
            self.extractor = seg_extractor
        else:
            self.extractor = Extractor()
        self.extractor_trunk = Extractor()
        # self.extractor_leg = Extractor()
        self.extractor_f_leg = Extractor()
        self.extractor_h_leg = Extractor()

        self.classifier = Classifier(feature_size=feature_size*seg_n_classes, class_num=class_num)
        self.softmax2d = nn.Softmax2d()

    def forward(self, img, seg, global_feat, denorm):
        feat_list = []
        seg = self.softmax2d(seg)  # convert to probability
        img_den = denormalize(img)
        # TODO: ignore background

        # global_feat = self.fc(global_feat)

        trunk_map = seg[:, 1, :, :].unsqueeze(1)
        if denorm:
            trunk_feat, _ = self.extractor_trunk(normalize(img_den * trunk_map))
        else:
            trunk_feat, _ = self.extractor_trunk(img * trunk_map)

        # leg_map = seg[:, 2, :, :].unsqueeze(1)
        # leg_feat, _ = self.extractor_leg(normalize(img_den * leg_map))

        f_leg_map = seg[:, 2, :, :].unsqueeze(1)
        if denorm:
            f_leg_feat, _ = self.extractor_f_leg(normalize(img_den * f_leg_map))
        else:
            f_leg_feat, _ = self.extractor_f_leg(img * f_leg_map)

        h_leg_map = seg[:, 3, :, :].unsqueeze(1)
        if denorm:
            h_leg_feat, _ = self.extractor_h_leg(normalize(img_den * h_leg_map))
        else:
            h_leg_feat, _ = self.extractor_h_leg(img * h_leg_map)

        # concat
        # f = torch.cat([global_feat, trunk_feat, leg_feat], dim=1)
        f = torch.cat([global_feat, trunk_feat, f_leg_feat, h_leg_feat], dim=1)

        x = self.classifier(f)
        return f, x


class Seg_Model(nn.Module):
    def __init__(self, class_num=4):
        super(Seg_Model, self).__init__()
        ''' feature extractor '''
        # self.extractor = models.resnet18(pretrained=True)  # (B, C, H//32, W//32)
        self.extractor = Extractor()
        ''' ASPP module'''
        self.aspp = ASPP(input_c=512, output_c=64)
        # self.aspp = ASPP(input_c=2048, output_c=64)
        ''' decoder '''
        self.transpose4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1,
                                             bias=False)
        self.transpose5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1,
                                             bias=False)
        self.conv = nn.Conv2d(in_channels=16, out_channels=class_num, kernel_size=1, stride=1, padding=0, bias=True)
        ''' 1x1 conv '''
        self.bypassConv_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.bypassConv_2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        self.bypassConv_3 = nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=1, stride=1, padding=0,
                                      bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        f_H, f_W = img.shape[2]//32, img.shape[3]//32
        # x = self.extractor.conv1(img)
        # x = self.extractor.bn1(x)
        # x = self.extractor.relu(x)
        # x = self.extractor.maxpool(x)
        #
        # x = self.extractor.layer1(x)
        # x = self.extractor.layer2(x)
        # x = self.extractor.layer3(x)
        # x = self.extractor.layer4(x)

        x_pool, x = self.extractor(img)

        x = self.aspp(x)
        x_1 = F.interpolate(x, size=(f_H * 8, f_W * 8), mode='bilinear', align_corners=True)
        x_2 = self.bypassConv_1(F.interpolate(x, size=(f_H * 16, f_W * 16), mode='bilinear', align_corners=True))
        x_3 = self.bypassConv_2(F.interpolate(x, size=(f_H * 32, f_W * 32), mode='bilinear', align_corners=True))
        x_4 = self.bypassConv_3(F.interpolate(x, size=(f_H * 32, f_W * 32), mode='bilinear', align_corners=True))
        x = self.relu(self.transpose4(x_1))
        x = self.relu(self.transpose5(x+x_2))
        x = self.conv(x+x_3)
        return x_pool, x + x_4  # (B, cls, 224, 224)


class ASPP(nn.Module):
    def __init__(self, input_c, output_c):
        super(ASPP, self).__init__()
        self.atrous1 = nn.Sequential(nn.Conv2d(input_c, output_c, 3, stride=1, padding=4, dilation=4, bias=True),
                                     nn.ReLU(inplace=True))
        self.atrous2 = nn.Sequential(nn.Conv2d(input_c, output_c, 3, stride=1, padding=6, dilation=6, bias=True),
                                     nn.ReLU(inplace=True))
        self.atrous3 = nn.Sequential(nn.Conv2d(input_c, output_c, 3, stride=1, padding=8, dilation=8, bias=True),
                                     nn.ReLU(inplace=True))
        self.atrous4 = nn.Sequential(nn.Conv2d(input_c, output_c, 3, stride=1, padding=11, dilation=11, bias=True),
                                     nn.ReLU(inplace=True))

    def forward(self, feature_map):
        x1 = self.atrous1(feature_map)
        x2 = self.atrous2(feature_map)
        x3 = self.atrous3(feature_map)
        x4 = self.atrous4(feature_map)
        x = x1 + x2 + x3 + x4
        return x
