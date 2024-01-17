import torch
from torch import nn


class CNNRSS(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fcc = nn.Linear(18400, 3)

    def forward(self, x):
        out = x.float()
        out = out.unsqueeze(1)

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = torch.flatten(out, 1)
        out = self.fcc(out)

        return out


class MLPRSS(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.dense1 = nn.Linear(input_dim, 960)
        self.dense2 = nn.Linear(960, 860)
        self.dense3 = nn.Linear(860, 560)
        self.dense4 = nn.Linear(560, 460)
        self.dense5 = nn.Linear(460, 360)
        self.dense6 = nn.Linear(360, 16)
        self.fcc = nn.Linear(16, 3)


    def forward(self, x):
        out = x.float()

        out = self.dense1(out)
        out = self.relu(out)

        out = self.dense2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.dense3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.dense4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.dense5(out)
        out = self.relu(out)

        out = self.dense6(out)
        out = self.fcc(out)

        return out



