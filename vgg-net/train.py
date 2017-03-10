from model import vgg
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

def get_batch(batch_id, data):
    pass

def train(data, nb_epoch):
    model = vgg()
    running_loss = 0.0
    criterion = CrossEntropyLoss()
    optimizer = SGD(vgg.parameters(), lr= 0.0001, momentum=0.9)
    for i in range(nb_epoch):
        input, target = get_batch(i, data)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        # what is the shape of loss.data?
        running_loss += loss.data[0]
    print 'Finished training'
    return model