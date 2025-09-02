import torch.nn as nn
import torch


class RevIN(nn.Module):
    def __init__(self, args):
        super(RevIN, self).__init__()
        self.args = args
        self.eps = 1e-5
        self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.args.input_size))
        self.affine_bias = nn.Parameter(torch.ones(self.args.input_size))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))  # (batch_size,time,hidden),确定数据维度，取到时间维度
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.args.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.args.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x * self.mean
        return x

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.seg_len = args.seg_len
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.history_len = args.history_len
        self.predict_len = args.predict_len
        self.dec_way = args.dec_way
        self.input_size = args.input_size

        self.revinLayer = RevIN(args)
        self.valueEmbeding = nn.Sequential(nn.Linear(in_features=self.seg_len, out_features=self.hidden_size),
                                           nn.ReLU())
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

        self.predict = nn.Sequential(nn.Dropout(self.dropout_rate),
                                     nn.Linear(self.hidden_size, self.seg_len))
        self.seg_num_x = self.history_len // self.seg_len
        self.seg_num_y = self.predict_len // self.seg_len

        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.hidden_size))

    def forward(self, x):
        x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        x = self.valueEmbeding(x.reshape(-1, self.seg_num_x, self.seg_len))
        _, hn = self.gru(x)
        if self.dec_way == 'rmf':
            y = []
            for i in range(self.seg_num_y):
                yy = self.predict(hn)
                yy = yy.permute(1, 0, 2)
                y.append(yy)
                yy = self.valueEmbeding(yy)
                _, hn = self.gru(yy, hn)
            y = (torch.stack(y, dim=1).squeeze(2).reshape(-1, self.input_size, self.predict_len))
        elif self.dec_way == 'pmf':
            pos_emb = self.pos_emb.repeat(x.size(0) * self.input_size, 1).unsqueeze(1)
            _, hy = self.gru(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.hidden_size))
            y = self.predict(hy).view(-1, self.input_size, self.predict_len)
        y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        y = y[:, :, -1:].squeeze(2)
        return y
