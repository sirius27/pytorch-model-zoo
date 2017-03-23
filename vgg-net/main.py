from model import vgg
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
VERBOSE = True
BATCH_SIZE = 4

NB_EPOCH = 500
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
    plt.imshow(npimg.transpose((1,2,0)))
    plt.show()


def train(nb_epoch):
    net = vgg()
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr= 0.0001)
    for epoch in range(nb_epoch):
        start = time.clock()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            if i>10:
                break
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if not VERBOSE:
                if i%2000 == 1999:
                    print 'batch %d/%d of epoch %d: loss=%.3f'%(i, len(trainloader), epoch, running_loss)
            else:
                 print 'batch %d/%d of epoch %d: loss=%.3f'%(i,len(trainloader), epoch, running_loss)
        end = time.clock()
        print "epoch{0} time:{1}".format(epoch,end-start)
    print 'Finished training'
    return net

def test(net):
    predictions = list()
    for i,data in enumerate(testloader):
        inputs, targets = data
        outputs = net(inputs)
        predict_classes = [classes[output] for output in outputs]
        target_classes = [classes[target] for target in targets]

##redirect output to file
import datetime
import sys
report_file = "../report/report_file_{0}.txt".format(str(datetime.datetime.now()))
ini_stdout = sys.stdout
with open(report_file,'w') as file:
    sys.stdout = file
    net = train(NB_EPOCH)

    correct_samples = np.zeros([len(classes),1]).ravel()
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