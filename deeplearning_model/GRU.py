import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        self.lstm1 = nn.GRU(input_size=args.input_size, hidden_size=args.hidden_size, batch_first=True)
        self.lstm2 = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(in_features=args.hidden_size, out_features=args.predict_len)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        y = self.linear(x)
        return y
