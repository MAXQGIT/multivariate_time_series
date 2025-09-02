import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.linear = nn.Linear(args.history_len, args.predict_len)

    def forward(self, x):
        seq_last = x[:, -1:, :]
        x = x - seq_last
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        x= x[:,:,-1:].squeeze(2)
        return x
