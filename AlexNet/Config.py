import math

import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, k=2):
        super(LRN, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.average = nn.AvgPool2d(kernel_size=local_size, stride=1)

    def forward(self, x):
        div = x.pow(2)
        div = self.average(div)
        div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
                                    nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),
                                    LRN(local_size=5))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2,
                                              padding=2, stride=1), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2),
                                    LRN(local_size=5))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer6 = nn.Sequential(nn.Linear(in_features=6 * 6 * 256, out_features=4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout())
        self.layer7 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout())
        self.layer8 = nn.Linear(in_features=4096, out_features=self.num_classes)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(-1, 6 * 6 * 256)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


# Local Response Normalization
batch_size = 256
use_gpu = torch.cuda.is_available()
# 此标准将LogSoftMax和NLLLoss集成到一个类中,torch.nn.NLLLoss()负的log likelihood loss损失.
criterion = nn.CrossEntropyLoss(size_average=False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder('/DataDisk/data/ILSVRC12_image_train/',
                                     transforms.Compose([transforms.RandomSizedCrop(227),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         normalize
                                                         ]
                                                        )
                                     )
weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
model = AlexNet(len(train_dataset.classes))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=2,
                                           pin_memory=True,
                                           sampler=sampler
                                           )
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))