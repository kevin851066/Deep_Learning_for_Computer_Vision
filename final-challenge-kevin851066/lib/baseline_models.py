import torch
import torch.nn as nn
import torchvision.models as models


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))

    def forward(self, input):
        return self.model(input).view(input.shape[0], -1)


class Classifier(nn.Module):
    def __init__(self, feature_size, class_num):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_size, out_features=feature_size//2),
            nn.BatchNorm1d(num_features=feature_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_size//2, out_features=feature_size//4),
            nn.BatchNorm1d(num_features=feature_size//4),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_size//4, out_features=class_num),
        )

    def forward(self, input):
        return self.classifier(input)


class Model(nn.Module):
    def __init__(self, class_num=72, feature_size=2048):
        super(Model, self).__init__()
        self.extractor = Extractor()
        self.classifier = Classifier(feature_size=feature_size, class_num=class_num)

    def forward(self, input):
        x1 = self.extractor(input)
        x2 = self.classifier(x1)
        return x1, x2
