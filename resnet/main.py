from torch.autograd import Variable
from torch import utils
from model import ResNet
import torch.nn as nn
from torch.optim import Adam
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

VERBOSE = True
BATCH_SIZE = 4

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10('../data', train=True, transform=transformer,
                                        download=True)
# why cannot utils be detected by IDE?
# I should find the answer from __init__ file of torch
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE)

testset = torchvision.datasets.CIFAR10('../data', train=False, transform=transformer, download=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print trainloader[0]
def imshow(img):
    # input Tensor
    img = 0.5*img + 0.5
    img = img.numpy()
    print img.shape
    plt.imshow(img.transpose((1,2,0)))
    plt.show()

def train(nb_epoch):
    model = ResNet()
    optimizer = Adam(ResNet.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    for epoch in xrange(nb_epoch):
        running_loss = 0.0
        for i,data in enumerate(trainloader):
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if not VERBOSE:
                if i%2000 == 1999:
                    print 'batch %d/%d of epoch %d: loss=%.3f'%(i, len(trainloader), epoch, running_loss)
            else:
                 print 'batch %d/%d of epoch %d: loss=%.3f'%(i,len(trainloader), epoch, running_loss)
    return model

net = train(2)

correct_samples = np.zeros([len(classes), 1]).ravel()
total_samples = np.zeros_like(correct_samples)
for i,data in enumerate(testloader):
    if i > 20:
        break
    inputs, targets = data
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    _, predictions = torch.max(outputs.data, 1)
    predict_classes = [classes[predictions[i][0]] for i in xrange(BATCH_SIZE)]
    target_classes = [classes[targets.data[i]] for i in xrange(BATCH_SIZE)]
    for i in xrange(BATCH_SIZE):
        if predictions[i][0] == targets.data[i]:
            correct_samples[targets.data[i]] += 1
        total_samples[targets.data[i]] += 1
corrct_rate = [correct/total for correct, total in zip(correct_samples, total_samples)]
print 'correct rate is ', corrct_rate