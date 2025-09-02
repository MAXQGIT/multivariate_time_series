import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.conv1d = nn.Conv1d(in_channels=self.args.input_size, out_channels=self.args.input_size,
                                kernel_size=self.args.kernel_size,
                                stride=1, padding=0)
        self.gru = nn.GRU(input_size=self.args.input_size, hidden_size=args.hidden_size, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(args.hidden_size, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, self.args.predict_len))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x
