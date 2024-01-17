from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.hidden1 = nn.LSTM(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden2 = nn.LSTM(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.fcc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out = x.float()

        out, (_, _) = self.hidden1(out)
        out = self.dropout1(out)
        out, (_, _) = self.hidden2(out)
        out = self.dropout2(out)
        out = self.fcc(out)

        return out