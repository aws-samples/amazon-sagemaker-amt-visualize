
import torch
import torch.nn as nn
import torch.nn.functional as F

                
class CNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 d, 
                 kernel_sizes,
                 num_filters,
                 strides,
                 dropout,
                 **kwargs):

        super().__init__()
        print(f'superfluous model parameters not used: {kwargs}')
        assert len(num_filters) == len(kernel_sizes) == len(strides), 'Same number of values for kernel_sizes, num_filters and strides must be specified.'
       
        self.embeddings = nn.Embedding(vocab_size, d, padding_idx=1)
        
        layers = []
        layers.append(Multihead([self._create_conv(
            d, 
            num_filters[i], 
            kernel_sizes[i], 
            dropout, 
            strides[i]
        ) for i in range(len(kernel_sizes))]))
        
        ff_in = sum(num_filters)
        
        self.nn = nn.Sequential(
            *layers,
            nn.BatchNorm1d(ff_in),
            MaxOverTimePooling(),
            nn.Linear(ff_in, num_classes))

    def forward(self, X):
        emb = self.embeddings(X).transpose(2, 1)
        
        return self.nn(emb)

    def _create_conv(self,
                     in_channels,
                     out_channels,
                     kernel_size,
                     dropout,
                     stride):
        
        padding = int((kernel_size-1)//2)
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout))

class MaxOverTimePooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.max(x, -1)[0]

class Cat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.cat(x, dim=self.dim)
    def extra_repr(self):
        return str(f'dim={self.dim}')

class Multihead(nn.Module):
    def __init__(self, heads, dim=1):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.cat = Cat(dim)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return self.cat(outputs)
