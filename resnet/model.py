from torch.nn import Module
class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, *x):
        return x