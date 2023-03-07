
import torch
import torch.nn as nn
import sys
import numpy as np
from functools import partial
from src.model.apple_wraping import get_mean_wrapper
sys.path.append("../../../")


class ExtActFixed(nn.Module):
    def __init__(self,
                 d,
                 K,
                 silence,
                 not_doing=False,
                 var_coef=1,
                 data_init=False):
        super().__init__()
        self.d = d
        self.data_init = data_init
        self.K = K
        self.silence = silence
        self.var_coef = var_coef
        self.not_doing = not_doing
        self.compute_mean_var()

        self.fixed_embedding = nn.Embedding.from_pretrained(
            self.bias_scale_matrix, freeze=True)

    def compute_mean_var(self):
        mean, min_dist = get_mean_wrapper(K=self.K, d=self.d, not_doing=self.not_doing)
        var = min_dist/(self.var_coef*2*self.K*3**(1/self.d))
        if not self.silence:
            print('min_dist : ', min_dist, ' var val : ', var)
        
        var = np.full((self.K, self.d), var)
        scale = np.log(var)
        bias = mean * np.exp(-scale)
        bias_scale_matrix = np.concatenate((bias, scale), 1)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("bias_scale_matrix", to_torch(bias_scale_matrix))

    def get_bias_scales(self, x):
        out = self.fixed_embedding(x)
        bias, scales = out.chunk(2, dim=2)
        return bias, scales

    def get_mean_var(self, device):
        input = torch.LongTensor([i for i in range(self.K)]).to(device)
        out = self.fixed_embedding(input)
        bias, scales = out.chunk(2, dim=1)

        var = torch.exp(scales)
        means = bias * var
        return means, var

    def forward(self, z, x, ldj=None, reverse=False, channel_padding_mask=None, layer_share_dict=None, **kwargs):
        if ldj is None:
            ldj = z.new_zeros(z.size(0), )
        if channel_padding_mask is None:
            channel_padding_mask = 1.0

        bias, scales = self.get_bias_scales(x)
        if not reverse:
            z = (z + bias) * torch.exp(scales)
            ldj += (scales * channel_padding_mask).sum(dim=[1, 2])
            if layer_share_dict is not None:
                layer_share_dict["t"] = (layer_share_dict["t"] +
                                         bias) * torch.exp(scales)
                layer_share_dict["log_s"] = layer_share_dict["log_s"] + scales
        else:
            z = z * torch.exp(-scales) - bias
            ldj += -(scales * channel_padding_mask).sum(dim=[1, 2])

        assert torch.isnan(z).sum() == 0, "[!] ERROR: z contains NaN values."
        assert torch.isnan(
            ldj).sum() == 0, "[!] ERROR: ldj contains NaN values."
        # print(self.bias_scale_matrix.numpy())
        return z, ldj

    def need_data_init(self):
        return self.data_init

    def data_init_forward(self,
                          input_data,
                          channel_padding_mask=None,
                          **kwargs):
        if channel_padding_mask is None:
            channel_padding_mask = input_data.new_ones(input_data.shape)
        else:
            channel_padding_mask = channel_padding_mask.view(
                input_data.shape[:-1] + channel_padding_mask.shape[-1:])
        mask = channel_padding_mask
        num_exp = mask.sum(dim=[0, 1], keepdims=True)
        masked_input = input_data

        bias_init = -masked_input.sum(dim=[0, 1], keepdims=True) / num_exp

        var_data = (((input_data + bias_init)**2) * mask).sum(
            dim=[0, 1], keepdims=True) / num_exp
        scaling_init = -0.5 * var_data.log()

        bias = torch.cat([bias_init, scaling_init], dim=-1).squeeze()

        out = (masked_input + bias_init) * torch.exp(scaling_init)
        out_mean = (out * mask).sum(dim=[0, 1]) / num_exp.squeeze()
        out_var = torch.sqrt(
            (((out - out_mean)**2) * mask).sum(dim=[0, 1]) / num_exp)
        print("[INFO - External ActNorm] New mean", out_mean)
        print("[INFO - External ActNorm] New variance", out_var)

    def info(self):
        return "External Activation Fixed Log (d=%i)" % (self.d)
