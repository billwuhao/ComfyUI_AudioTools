import inspect
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel
from ..layers import activations, normalizations


def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class ATTConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = 0 if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs)
            if self.norm_type == "LayerNormalization4D"
            else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")
        # sig_act = self.act(global_act)

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out


class InjectionMulti(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")
        # sig_act = self.act(global_act)

        out = local_feat * sig_act
        return out


class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(
        self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=5,
                    stride=2,
                    groups=in_channels,
                    d=1,
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(InjectionMultiSum(in_channels, in_channels))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)

        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(InjectionMultiSum(in_channels, in_channels, 5))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
            # global_f = global_f + fea
        global_f = self.globalatt(global_f)  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        return self.res_conv(expanded) + residual


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "prelu",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = ATTConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view(
            [self.n_head, batch_size, emb_dim, time, freq]
        )  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


class Recurrent(nn.Module):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        nband=8,
        upsampling_depth=3,
        n_head=4,
        att_hid_chan=4,
        kernel_size: int = 8,
        stride: int = 1,
        _iter=4,
    ):
        super().__init__()
        self.nband = nband

        self.freq_path = nn.ModuleList(
            [
                UConvBlock(out_channels, in_channels, upsampling_depth),
                MultiHeadSelfAttention2D(
                    out_channels,
                    1,
                    n_head=n_head,
                    hid_chan=att_hid_chan,
                    act_type="prelu",
                    norm_type="LayerNormalization4D",
                    dim=4,
                ),
                normalizations.get("LayerNormalization4D")((out_channels, 1)),
            ]
        )

        self.frame_path = nn.ModuleList(
            [
                UConvBlock(out_channels, in_channels, upsampling_depth),
                MultiHeadSelfAttention2D(
                    out_channels,
                    1,
                    n_head=n_head,
                    hid_chan=att_hid_chan,
                    act_type="prelu",
                    norm_type="LayerNormalization4D",
                    dim=4,
                ),
                normalizations.get("LayerNormalization4D")((out_channels, 1)),
            ]
        )

        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        # B, nband, N, T
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # B, N, nband, T
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.freq_time_process(x, B, nband, N, T)  # B, N, nband, T
            else:
                x = self.freq_time_process(
                    self.concat_block(mixture + x), B, nband, N, T
                )  # B, N, nband, T

        return x.permute(0, 2, 1, 3).contiguous()  # B, nband, N, T

    def freq_time_process(self, x, B, nband, N, T):
        # Process Frequency Path
        residual_1 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()  # B, T, N, nband
        freq_fea = self.freq_path[0](x.view(B * T, N, nband))  # B*T, N, nband
        freq_fea = (
            freq_fea.view(B, T, N, nband).permute(0, 2, 1, 3).contiguous()
        )  # B, N, T, nband
        freq_fea = self.freq_path[1](freq_fea)  # B, N, T, nband
        freq_fea = self.freq_path[2](freq_fea)  # B, N, T, nband
        freq_fea = freq_fea.permute(0, 1, 3, 2).contiguous()
        x = freq_fea + residual_1  # B, N, nband, T
        # Process Frame Path
        residual_2 = x.clone()
        x2 = x.permute(0, 2, 1, 3).contiguous()
        frame_fea = self.frame_path[0](x2.view(B * nband, N, T))  # B*nband, N, T
        frame_fea = frame_fea.view(B, nband, N, T).permute(0, 2, 1, 3).contiguous()
        frame_fea = self.frame_path[1](frame_fea)  # B, N, nband, T
        frame_fea = self.frame_path[2](frame_fea)  # B, N, nband, T
        x = frame_fea + residual_2  # B, N, nband, T
        return x


class TIGER(nn.Module):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        att_n_head=4,
        att_hid_chan=4,
        att_kernel_size=8,
        att_stride=1,
        win=2048,
        stride=512,
        num_sources=2,
        sample_rate=44100,
    ):
        super(TIGER, self).__init__()

        self.sample_rate = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = out_channels
        self.num_output = num_sources
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (50 hop), 1k-2k (100 hop), 2k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sample_rate / 2.0) * self.enc_dim))
        self.band_width = [bandwidth_50] * 20
        self.band_width += [bandwidth_100] * 10
        self.band_width += [bandwidth_250] * 8
        self.band_width += [bandwidth_500] * 8
        self.band_width += [bandwidth_1k] * 8
        self.band_width += [bandwidth_2k] * 2
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        self.separator = Recurrent(
            self.feature_dim,
            in_channels,
            self.nband,
            upsampling_depth,
            att_n_head,
            att_hid_chan,
            att_kernel_size,
            att_stride,
            num_blocks,
        )

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.PReLU(),
                    nn.Conv1d(
                        self.feature_dim,
                        self.band_width[i] * 4 * num_sources,
                        1,
                        groups=num_sources,
                    ),
                )
            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        # input shape: (B, C, T)
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            was_one_d = True
            input = input.unsqueeze(1)
        if input.ndim == 3:
            input = input
        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        # frequency-domain separation
        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).type(input.dtype).to(input.device),
            return_complex=True,
        )

        # print(spec.shape)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(
                spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            subband_spec.append(
                spec[:, band_idx : band_idx + self.band_width[i]]
            )  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec_RI[i].view(
                        batch_size * nch, self.band_width[i] * 2, -1
                    )
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T
        # import pdb; pdb.set_trace()
        # separator
        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband, self.feature_dim, -1)
        )  # B, nband, N, T
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(self.nband):
            this_output = self.mask[i](sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.num_output, self.band_width[i], -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1]
            )  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            # force mask sum to 1
            this_mask_real_sum = this_mask_real.sum(1).unsqueeze(1)  # B*nch, 1, BW, T
            this_mask_imag_sum = this_mask_imag.sum(1).unsqueeze(1)  # B*nch, 1, BW, T
            this_mask_real = this_mask_real - (this_mask_real_sum - 1) / self.num_output
            this_mask_imag = this_mask_imag - this_mask_imag_sum / self.num_output
            est_spec_real = (
                subband_spec[i].real.unsqueeze(1) * this_mask_real
                - subband_spec[i].imag.unsqueeze(1) * this_mask_imag
            )  # B*nch, K, BW, T
            est_spec_imag = (
                subband_spec[i].real.unsqueeze(1) * this_mask_imag
                + subband_spec[i].imag.unsqueeze(1) * this_mask_real
            )  # B*nch, K, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        sep_subband_spec = torch.cat(sep_subband_spec, 2)

        output = torch.istft(
            sep_subband_spec.view(batch_size * nch * self.num_output, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).type(input.dtype).to(input.device),
            length=nsample,
        )
        output = output.view(batch_size * nch, self.num_output, -1)
        # if was_one_d:
        #     return output.squeeze(0)
        return output

    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args


class TIGERDNR(BaseModel):
    def __init__(
        self,
        out_channels=132,
        in_channels=256,
        num_blocks=8,
        upsampling_depth=5,
        att_n_head=4,
        att_hid_chan=4,
        att_kernel_size=8, 
        att_stride=1,
        win=2048, 
        stride=512,
        num_sources=3,
        sample_rate=44100,
    ):
        super(TIGERDNR, self).__init__(sample_rate=sample_rate)
        self.sr = sample_rate

        self.dialog = TIGER(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            att_kernel_size=att_kernel_size,
            att_stride=att_stride,
            win=win,
            stride=stride,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )
        self.effect = TIGER(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            att_kernel_size=att_kernel_size,
            att_stride=att_stride,
            win=win,
            stride=stride,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )
        self.music = TIGER(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            att_kernel_size=att_kernel_size,
            att_stride=att_stride,
            win=win,
            stride=stride,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )
        
    def wav_chunk_inference(self, model, mixture_tensor, target_length=12.0, hop_length=4.0, batch_size=1, n_tracks=3):
        """
        Input:
            mixture_tensor: Tensor, [nch, input_length]
            
        Output:
            all_target_tensor: Tensor, [nch, n_track, input_length]    
        """

        batch_mixture = mixture_tensor # [1, nch, T]
        # print(batch_mixture.shape, [:,:int(self.sr*24)])

        # split data into segments
        batch_length = batch_mixture.shape[-1]

        session = int(self.sr * target_length)
        target = int(self.sr * target_length)
        ignore = (session - target) // 2
        hop = int(self.sr * hop_length)
        tr_ratio = target_length / hop_length
        if ignore > 0:
            zero_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], ignore).type(batch_mixture.dtype).to(batch_mixture.device)
            batch_mixture_pad = torch.cat([zero_pad, batch_mixture, zero_pad], -1)
        else:
            batch_mixture_pad = batch_mixture
        if target - hop > 0:
            hop_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], target-hop).type(batch_mixture.dtype).to(batch_mixture.device)
            batch_mixture_pad = torch.cat([hop_pad, batch_mixture_pad, hop_pad], -1)

        skip_idx = ignore + target - hop
        zero_pad = torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], session).type(batch_mixture.dtype).to(batch_mixture.device)
        num_session = (batch_mixture_pad.shape[-1] - session) // hop + 2
        all_target = torch.zeros(batch_mixture_pad.shape[0], n_tracks, batch_mixture_pad.shape[1], batch_mixture_pad.shape[2]).to(batch_mixture_pad.device)
        all_input = []
        all_segment_length = []

        for i in range(num_session):
            this_input = batch_mixture_pad[:,:,i*hop:i*hop+session]
            segment_length = this_input.shape[-1]
            if segment_length < session:
                this_input = torch.cat([this_input, zero_pad[:,:,:session-segment_length]], -1)
            all_input.append(this_input)
            all_segment_length.append(segment_length)

        all_input = torch.cat(all_input, 0)
        num_batch = num_session // batch_size
        if num_session % batch_size > 0:
            num_batch += 1
        
        for i in range(num_batch):

            this_input = all_input[i*batch_size:(i+1)*batch_size]
            actual_batch_size = this_input.shape[0]
            with torch.no_grad():
                est_target = model(this_input)
                # batch, ntrack, nch, T = est_target.shape
                # est_target = est_target.transpose(1, 2).view(batch*nch, ntrack, T)
                est_target = est_target.unsqueeze(2)
                
            for j in range(actual_batch_size):
                this_est_target = est_target[j,:,:,:all_segment_length[i*batch_size+j]][:,:,ignore:ignore+target].unsqueeze(0)
                all_target[:,:,:,ignore+(i*batch_size+j)*hop:ignore+(i*batch_size+j)*hop+target] += this_est_target # [batch, ntrack, nch, T]

        all_target = all_target[:,:,:,skip_idx:skip_idx+batch_length].contiguous() / tr_ratio

        return all_target.squeeze(0)
    
    def forward(self, mixture_tensor):
        all_target_dialog = self.wav_chunk_inference(self.dialog, mixture_tensor)[2]
        all_target_effect = self.wav_chunk_inference(self.effect, mixture_tensor)[1]
        all_target_music = self.wav_chunk_inference(self.music, mixture_tensor)[0]
        return all_target_dialog, all_target_effect, all_target_music
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args