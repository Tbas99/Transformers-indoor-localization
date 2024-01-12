import torch
from torch import nn







class Classic(nn.Module):
    def __init__(self):
        super().__init__()

        self.base1 = nn.Linear(620, 620)
        self.base2 = nn.Linear(620, 512)
        self.fcc = nn.Linear(512, 3)

    def forward(self, x):
        out = x.float()
        out = self.base1(out)
        out = self.base2(out)

        #out = torch.flatten(out, 1)
        out = self.fcc(out)

        return out