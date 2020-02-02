import numpy as np

import torch
import torch.nn as nn

class BinocularNetwork(nn.Module):
    def __init__(self, n_filters=28, n_latent=100, relu_latent=False, k_size=19, input_size=30):
        super(BinocularNetwork, self).__init__()
        assert k_size % 2 == 1, "Kernel/filter size must be odd!"

        self.n_filters = n_filters
        self.k_size = k_size
        self.input_size = input_size
        self.num_in_channels = 2

        self.simple_unit = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                n_filters,
                kernel_size=k_size,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        )
        n_units = n_filters*(self.input_size-k_size+1)*(self.input_size-k_size+1) / 4
        if relu_latent:
            self.complex_unit = nn.Sequential(
                nn.Linear(n_units, n_latent, bias=True),
                nn.ReLU(inplace=False)
            )
        else:
            self.complex_unit = nn.Linear(n_units, n_latent, bias=True)
        self.classify = nn.Linear(n_latent, 6, bias=True)

        print "Initialize simple units with Xavier initialization."
        self.simple_unit.apply(self._init_weights)

        print "Initialize complex units with Xavier initialization."
        self.complex_unit.apply(self._init_weights)
        self.classify.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            print m
            nn.init.xavier_uniform_(m.weight)

    def _init_zeros(self, m):
        if type(m) == nn.Linear:
            print m
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    def forward(self, x):
        x = self.simple_unit(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.complex_unit(x)
        x = self.classify(x)
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
    else:
        device = torch.device("cpu")
    print "Device:", device

    m = BinocularNetwork(n_filters=40, input_size=108).to(device)
    t = torch.rand(29,2,108,108)
    q = m(t.to(device))
    print q.size()
#    print q

#    print m.simple_unit[0].weight
#
#    from torch.autograd import Variable
#    a = torch.rand(5,2, requires_grad=True).to(device)
#    print a
#    a = Variable(torch.rand(5,2).to(device), requires_grad=True)
#    print a
#    a = torch.rand(5,2, device=device, requires_grad=True)
#    print a
    
