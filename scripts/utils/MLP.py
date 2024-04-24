import torch.nn as nn
import torch
import torch.nn.init as init
from typing import Any, Callable, List, Optional, Type, Union


torch.manual_seed(42)


class MLP_Block(nn.Module):
    def __init__(self, 
                 inplanes : int,
                outplanes: int, 
                downsample : Optional[nn.Module]):
        super().__init__()
        self.mlp_layer = nn.Linear(inplanes, outplanes)
        self.ReLU = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(outplanes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.mlp_layer(x)
        out = self.ReLU(out)
        out = self.bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return out


class MLP_EXP(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.in_layer = MLP_Block(3072, 4608, nn.Sequential(nn.Linear(3072, 4608), nn.BatchNorm1d(4608), nn.ReLU()))
        self.hidden_layers = self._make_layer(4608, 4608, num_blocks=num_layers)
        self.out_layer1 = MLP_Block(4608, 1536, nn.Sequential(nn.Linear(4608, 1536), nn.BatchNorm1d(1536), nn.ReLU()))
        self.out_layer2 = MLP_Block(1536, 10, nn.Sequential(nn.Linear(1536, 10)))
        self.apply(self._init_weights)


    def _make_layer(self, 
                    inplanes: int,
                    outplanes : int,
                    num_blocks: int
                    ):
        layers = []
        for i in range(num_blocks):
            layers.append(MLP_Block(inplanes, outplanes, None))
        
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = x.flatten(1)
        out = self.in_layer(out)
        out = self.hidden_layers(out)
        out = self.out_layer1(out)
        out = self.out_layer2(out)

        return out



"""
class MLP_EXP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Sequential(nn.Linear(3072, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 4608),
                                    nn.BatchNorm1d(4608),
                                    nn.ReLU(),
                                    nn.Linear(4608, 1536),
                                    nn.BatchNorm1d(1536),
                                    nn.ReLU(),
                                    nn.Linear(1536,10)
                                    )
        self.apply(self._init_weights)
        self.ReLU = nn.ReLU()    
    def forward(self, x):
        x = x.flatten(1)
        x = self.hidden(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)

"""
            

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3072, 6144)
        init.kaiming_normal_(self.l1.weight)
        self.hidden = nn.Sequential(
                                    nn.Linear(6144, 3072),
                                    nn.ReLU(),
                                    nn.Linear(3072, 1536),
                                    nn.ReLU(),
                                    )
        self.apply(self._init_weights)
        self.l2 = nn.Linear(1536,10)
        init.kaiming_normal_(self.l2.weight)
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.hidden(x)
        x = self.l2(x)
        return x
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)




"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3072, 6144)
        init.kaiming_normal_(self.l1.weight)
        self.hidden = nn.Sequential(
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 6144),
                                    nn.ReLU(),
                                    nn.Linear(6144, 3072),
                                    nn.ReLU(),
                                    nn.Linear(3072, 1536),
                                    nn.ReLU(),


                                    )
        self.apply(self._init_weights)
        self.l2 = nn.Linear(1536,10)
        init.kaiming_normal_(self.l2.weight)
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.hidden(x)
        x = self.l2(x)
        return x
    


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)

"""
