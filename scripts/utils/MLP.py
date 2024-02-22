import torch.nn as nn
import torch
import torch.nn.init as init

torch.manual_seed(42)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 1024)
        init.kaiming_normal_(self.l1.weight)
        self.hidden = nn.Sequential(
                                    nn.Linear(1024, 1024),
                                    nn.ReLU(),
                                    )
        self.apply(self._init_weights)
        self.l2 = nn.Linear(1024,2)
        init.kaiming_normal_(self.l2.weight)
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.hidden(x)
        x = self.l2(x)
        return x
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            