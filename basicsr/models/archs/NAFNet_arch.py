# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


##########################################################################
# PSF Kernel Encoder  (from Restormer)
##########################################################################

class PSFKernelEncoder(nn.Module):
    """Encode a 2D PSF kernel (e.g. 31x31) into a compact feature vector via CNN.

    Input:  (B, 1, psf_size, psf_size)  or  (B, psf_size, psf_size)
    Output: (B, feat_dim)
    """
    def __init__(self, psf_size=31, feat_dim=16):
        super().__init__()
        self.psf_size = psf_size
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feat_dim),
        )

    def forward(self, psf):
        if psf.dim() == 3:
            psf = psf.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        return self.encoder(psf)


def tile_psf_feature(psf_batch, psf_encoder, patch_size):
    """Encode batch PSF kernels and tile into patch-sized feature maps.

    Args:
        psf_batch:  (B, 3, S, S)  RGB PSF kernels
        psf_encoder: PSFKernelEncoder
        patch_size: int, target spatial size

    Returns:
        feat_map: (B, feat_dim, patch_size, patch_size)
    """
    B = psf_batch.shape[0]
    feat_dim = psf_encoder.feat_dim
    psf_gray = psf_batch.mean(dim=1, keepdim=True)  # (B, 1, S, S)
    psf_gray = psf_gray / (psf_gray.sum(dim=[2, 3], keepdim=True) + 1e-10)
    feats = psf_encoder(psf_gray)  # (B, feat_dim)
    return feats[:, :, None, None].expand(-1, -1, patch_size, patch_size)


##########################################################################
# NAFNet
##########################################################################

class NAFNet(nn.Module):

<<<<<<< HEAD
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],
                 out_channel=None):
=======
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[],
                 out_channel=None, kernel_channels=0):
>>>>>>> 10271d7 (add_psfencoder)
        super().__init__()

        if out_channel is None:
            out_channel = img_channel
        self.out_channel = out_channel
<<<<<<< HEAD

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
=======
        self.kernel_channels = kernel_channels

        # PSF encoder
        if kernel_channels > 0:
            self.psf_encoder = PSFKernelEncoder(psf_size=31, feat_dim=kernel_channels)

        total_in = img_channel + kernel_channels
        self.intro = nn.Conv2d(in_channels=total_in, out_channels=width,
                               kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel,
                                kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
>>>>>>> 10271d7 (add_psfencoder)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, kernel=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # Encode PSF kernel and concatenate
        if self.kernel_channels > 0 and kernel is not None:
            _, _, Hp, Wp = inp.shape
            kfeat = tile_psf_feature(kernel, self.psf_encoder, Wp)
            inp = torch.cat([inp, kfeat], dim=1)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:, :self.out_channel, :, :]

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=None, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        if train_size is None:
            img_ch = kwargs.get('img_channel', 3)
<<<<<<< HEAD
            train_size = (1, img_ch, 256, 256)
=======
            k_ch = kwargs.get('kernel_channels', 0)
            total_ch = img_ch + k_ch
            train_size = (1, total_ch, 256, 256)
>>>>>>> 10271d7 (add_psfencoder)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
