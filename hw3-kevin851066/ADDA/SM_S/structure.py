# original

class Encoder(nn.Module):   # modified LeNet
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature_extrator = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(800, 500)

    def forward(self, img):
        ft = self.feature_extrator(img)
        ft = ft.view(-1, 800)
        ft = self.fc1(ft)
        return ft

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 10)
        )
    
    def forward(self, ft):
        prob = self.clf(ft)
        return prob


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2),
            # nn.LogSoftmax()
        )
        
    def forward(self, img):
        prob = self.dis(img)

        return prob



class Encoder(nn.Module):   # modified LeNet
    def __init__(self):
        super(Encoder, self).__init__()
        self.feature_extrator = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(800, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, img):
        ft = self.feature_extrator(img)
        # print('ft: ', ft.shape)
        ft = ft.view(-1, 800)
        ft = self.leakyrelu( self.bn3( self.fc1(ft) ) )
        return ft

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(500, 10)
        )
    
    def forward(self, ft):
        prob = self.clf(ft)
        return prob

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Linear(500, 2),
        )
        
    def forward(self, img):
        prob = self.dis(img)

        return prob