from torch import utils
import torch
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 4

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10('../data', train=True, transform=transforms,
                                        download=True)
# why cannot utils be detected by IDE?
# I should find the answer from __init__ file of torch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)