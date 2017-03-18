from model import vgg
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

VERBOSE = True
BATCH_SIZE = 4

transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10('../data', train = True, download=True, transform=transformer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)

testset = torchvision.datasets.CIFAR10('../data', train = False, download=True, transform = transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(img)


def train(nb_epoch):
    net = vgg()
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr= 0.0001, momentum=0.9)
    for epoch in range(nb_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if VERBOSE:
                if i%2000 == 1999:
                    print 'batch %d of epoch %d: loss=%.3f'%(i, epoch, running_loss)
            else:
                 print 'batch %d/%d of epoch %d: loss=%.3f'%(i, epoch, running_loss)
    print 'Finished training'
    return net

def test(net):
    predictions = list()
    for i,data in enumerate(testloader):
        inputs, targets = data
        outputs = net(inputs)
        predict_classes = [classes[output] for output in outputs]
        target_classes = [classes[target] for target in targets]

net = train(2)