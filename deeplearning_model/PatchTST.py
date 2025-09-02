import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class series_decomp(nn.Module):
    def __init__(self, kersize):
        super(series_decomp, self).__init__()
        self.kersize = kersize
        self.avg = nn.AvgPool1d(kernel_size=kersize, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kersize - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kersize - 1) // 2, 1)
        y = torch.cat([front, x, end], dim=1)
        moving_mean = self.avg(y.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean

class RevIN(nn.Module):
    def __init__(self, c_in):
        super(RevIN, self).__init__()
        self.input_size = c_in
        self.eps = 1e-5
        self._init_parms()

    def _init_parms(self):
        self.affine_weight = nn.Parameter(torch.ones(self.input_size))
        self.affine_bias = nn.Parameter(torch.zeros(self.input_size))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stedv = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stedv
        x = x * self.affine_weight
        x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        x = x - self.affine_bias
        x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stedv
        x = x + self.mean
        return x

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise 'Please select "norm" or "denorm"'
        return x


def positional_encoding(q_len, d_model):
    W_pos = torch.empty((q_len, d_model))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=True)


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None: attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=True)
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return output, attn_weights


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, attn_dropout=0, dropout=0.):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.self_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=True))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, src):
        src2, attn = self.self_attn(src, src, src)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)
        return src


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 attn_dropout=0., dropout=0., n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout) for _ in
             range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, d_ff=256, attn_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len
        self.W_pos = positional_encoding(q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                                  n_layers=n_layers)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        return z


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride, n_layers=3, d_model=128, n_heads=8,
                 d_ff=256, attn_dropout=0., dropout=0., fc_dropout=0., head_dropout=0):
        super().__init__()
        self.revin_layer = RevIN(c_in)
        self.patch_len = patch_len
        self.stride = stride
        patch_num = int((context_window - patch_len) / stride + 1) + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.backbone = TSTiEncoder(patch_num=patch_num, patch_len=patch_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout)

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        self.head = Flatten_Head(self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z):
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)
        z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1))


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.stride = args.stride
        context_window = args.history_len
        patch_len = args.patch_len
        stride = args.stride
        n_layers = args.n_layers
        d_model = args.hidden_size
        n_heads = args.n_heads
        c_in = args.input_size
        d_ff = args.d_ff
        dropout = args.dropout_rate
        target_window = args.predict_len

        self.decomp_module = series_decomp(args.kernel_size)
        self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                             patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
                                             n_heads=n_heads, d_ff=d_ff, attn_dropout=0, dropout=dropout,
                                             fc_dropout=dropout, head_dropout=0)
        self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_ff=d_ff,
                                           attn_dropout=0, dropout=dropout, fc_dropout=dropout, head_dropout=0)

    def forward(self, x):
        res_init, trend_init = self.decomp_module(x)
        res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        x = x.permute(0, 2, 1)
        x = x[:, :, -1]
        return x
