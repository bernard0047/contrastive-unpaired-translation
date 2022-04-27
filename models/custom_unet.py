import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
# from torchvision.models import vgg16_bn
from torchvision.models.feature_extraction import create_feature_extractor


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class ConvNorm(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, groups=1, bias=None, bn=True, bn_zero=False,
                 act_cls=nn.ReLU, norm_lyr=nn.BatchNorm2d, spectral=False, icnr=False):
        super().__init__()
        if padding is None:
            padding = 'same' if stride == 1 else int(np.ceil((ks-1)/2))
        if bias is None:
            bias = not bn
        while ni % groups:
            groups //= 2
        while nf % groups:
            groups //= 2
        self.conv = nn.Conv2d(ni, nf, ks, stride, padding, groups=groups, bias=bias)
        if icnr:
            self.conv.weight.data.copy_(icnr_init(self.conv.weight.data))
            self.conv.bias.data.zero_()
        if spectral:
            self.conv = spectral_norm(self.conv)
        if bn:
            self.bn = norm_lyr(nf)
            if bn_zero and norm_lyr is nn.BatchNorm2d:
                self.bn.weight.data.fill_(0.)
        else:
            self.bn = nn.Identity()
        if act_cls is None:
            self.act = nn.Identity()
        else:
            self.act = act_cls(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=1, groups=1, reduction=0, spectral=False,
                 act_cls=nn.ReLU, self_attn=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = ConvNorm(ni, nf, ks, stride, groups=groups, act_cls=act_cls, spectral=spectral,
                              norm_lyr=norm_lyr)
        self.conv2 = ConvNorm(nf, nf, ks, groups=1, act_cls=None, spectral=spectral, bn_zero=True,
                              norm_lyr=norm_lyr)
        self.act = act_cls(inplace=True) if act_cls is nn.ReLU else act_cls()

        shortcut = []
        if ni != nf:
            shortcut.append(ConvNorm(ni, nf, 1, act_cls=nn.Identity, norm_lyr=norm_lyr))
        if stride > 1:
            shortcut.append(nn.MaxPool2d(stride))
        self.shortcut = nn.Sequential(*shortcut)

        if self_attn:
            self.atn = SelfAttention(nf)
        elif reduction:
            self.atn = SqueezeExcite(nf, reduction)
        else:
            self.atn = nn.Identity()

    def forward(self, x):
        inp = x
        x = self.conv2(self.conv1(x))
        x = self.atn(x)
        return self.act(x.add_(self.shortcut(inp)))


class UnetBlock(nn.Module):
    def __init__(self, ni, nf, skip_in, blur=True, act_cls=nn.ReLU, groups=1,
                 self_attn=False, reduction=0, spectral=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        self.pix_shuf = PixelShuffle_ICNR(ni, ni//2, blur=blur, act_cls=act_cls,
                                          spectral=spectral, norm_lyr=norm_lyr)
        rin = ni//2 + skip_in
        self.resb = ResBlock(rin, nf, groups=groups, reduction=reduction, spectral=spectral,
                             act_cls=act_cls, self_attn=self_attn, norm_lyr=norm_lyr)

    def forward(self, x, skip=None):
        x = self.pix_shuf(x)
        # x = F.interpolate(x, skip.shape[-2:], mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.resb(x)


class CustomUnet(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=32, num_scale=3, groups=32, reduction=16, spectral=True,
                 self_attn=False, norm_lyr=nn.InstanceNorm2d):
        super().__init__()
        self.conv_in = ConvNorm(in_c, ngf, ks=3, norm_lyr=norm_lyr, act_cls=nn.ReLU)
        kwargs = dict(groups=groups, reduction=reduction, spectral=spectral, norm_lyr=norm_lyr)
        self.down = self.get_block(ngf,  64, num=1, **kwargs)
        self.down0 = self.get_block(64,  96, num=1, **kwargs)
        self.down1 = self.get_block(96, 128, num=1, self_attn=self_attn, **kwargs)
        self.down2 = self.get_block(128, 256, num=1, **kwargs)
        self.down3 = self.get_block(256, 512, num=1, **kwargs)

        self.middle_conv = nn.Sequential()  # ConvNorm(512, 1024, spectral=spectral, norm_lyr=norm_lyr,
        #                                           act_cls=nn.ReLU),
        #                                  ConvNorm(1024, 512, spectral=spectral, norm_lyr=norm_lyr,
        #                                           act_cls=nn.ReLU),
        #                                 )

        self.up3 = UnetBlock(512, 256, 256, **kwargs)
        self.up2 = UnetBlock(256, 128, 128, **kwargs)
        self.up1 = UnetBlock(128,  96,  96, **kwargs)
        self.up0 = UnetBlock(96,  64,  64, **kwargs)
        self.up = UnetBlock(64, ngf, ngf, **kwargs)

        n_up = (ngf, 64, 96, 128, 256, 512)
        self.deep_convs = nn.ModuleList([nn.Conv2d(n_up[i], out_c, kernel_size=3 if i == 0 else 1,
                                                   padding='same') for i in range(num_scale)])

    def forward(self, x,encode_only=False):  # 3, 768
        x = self.conv_in(x)        # 32, 768
        d = self.down(x)           # 64, 384
        d0 = self.down0(d)          # 96, 192
        d1 = self.down1(d0)         # 128, 96
        d2 = self.down2(d1)         # 256, 48
        d3 = self.down3(d2)         # 512, 24

        u3 = self.middle_conv(d3)   # 512, 24

        u2 = self.up3(u3, d2)       # 256, 48
        u1 = self.up2(u2, d1)       # 128, 96
        u0 = self.up1(u1, d0)       # 96, 192
        u = self.up0(u0, d)        # 64, 384
        o = self.up(u, x)        # 32, 768

        if encode_only:
            return[x,d0,d1,d2,d3]
            # return list(torch.tanh_(conv_out(feat)) for conv_out, feat in zip(self.deep_convs, features))

        out = torch.tanh(self.deep_convs[0](o))
        return out
        # features = (o, u, u0, u1, u2, u3)
        # return tuple(torch.tanh_(conv_out(feat)) for conv_out, feat in zip(self.deep_convs, features))

    def get_block(self, ni, nf, num=2, self_attn=False, **kwargs):
        return nn.Sequential(*[ResBlock(ni if i == 0 else nf, nf, stride=2 if i == 0 else 1,
                                        self_attn=self_attn if i == 0 else False, **kwargs)
                               for i in range(num)])


class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf, scale=2, blur=True, act_cls=nn.ReLU, spectral=False, norm_lyr=nn.BatchNorm2d):
        super().__init__()
        layers = [ConvNorm(ni, nf*(scale**2), ks=1, bn=False, act_cls=act_cls, spectral=spectral,
                           icnr=True, norm_lyr=norm_lyr),
                  nn.PixelShuffle(scale)]
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction, act_cls=nn.ReLU) -> None:
        super().__init__()
        nf = ch//reduction
        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNorm(ch, nf, ks=1, bn=False, act_cls=act_cls),
            ConvNorm(nf, ch, ks=1, bn=False, act_cls=nn.Sigmoid)
        )

    def forward(self, x):
        return x * self.sq(x)


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.qkv_c = (n_channels//8, n_channels//8, n_channels)
        self.to_qkv = spectral_norm(nn.Conv2d(n_channels, sum(self.qkv_c), kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):       # [B, C, H, W]
        size = x.size()
        qkv = self.to_qkv(x)
        q, k, v = qkv.flatten(2).split(self.qkv_c, dim=1)   # [B, (dq,dk,dv), H*W]
        attn = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)  # [B, lq, lk]
        o = torch.bmm(v, attn)
        del attn, q, k, v, qkv
        o = o.view(*size)  # .contiguous()
        o = o.mul_(self.gamma) + x
        return o

