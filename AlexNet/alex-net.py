import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import Config


if Config.use_gpu:
    model = Config.model.cuda()
    print('USE GPU')
else:
    print('USE CPU')
# Adam: A Method for Stochastic Optimization

for epoch in range(1, 5):
    Config.train(epoch)