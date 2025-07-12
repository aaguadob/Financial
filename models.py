import torch
import torch.nn.functional as F
import torch.nn as nn

class DNN(torch.nn.Module):
    def __init__(self, architecture, input_size):
        super(DNN, self).__init__()
        self.architecture = architecture
        self.input = input_size
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input, self.architecture[0]))
        self.batchnorms= torch.nn.ModuleList()
        self.batchnorms.append(torch.nn.LayerNorm(self.architecture[0]))
        for in_dim, out_dim in zip(self.architecture[:-1], self.architecture[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            self.batchnorms.append(torch.nn.LayerNorm(out_dim))
        self.layers.append(torch.nn.Linear(self.architecture[-1], 1))
    def forward(self, x):
        for i,layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.batchnorms[i](x)
            x = F.relu(x)
            
        x = self.layers[-1](x).squeeze(1)
        return x

        