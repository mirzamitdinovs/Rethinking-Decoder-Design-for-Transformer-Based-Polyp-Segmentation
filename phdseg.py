import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class FFEModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        hp = self._high_pass_filter(H, W, x.device)
        high_freq = torch.fft.irfft2(freq * hp, s=(H, W), norm='ortho')
        high_freq = high_freq.to(x.dtype)
        return x + self.alpha * high_freq

    @staticmethod
    def _high_pass_filter(H, W, device):
        h_freq = torch.fft.fftfreq(H, device=device)
        w_freq = torch.fft.rfftfreq(W, device=device)
        fh, fw = torch.meshgrid(h_freq, w_freq, indexing='ij')
        dist = torch.sqrt(fh ** 2 + fw ** 2)
        sigma = 0.1
        hp = 1.0 - torch.exp(-dist ** 2 / (2 * sigma ** 2))
        return hp.unsqueeze(0).unsqueeze(0)


class DecoderBlock(nn.Module):

    def __init__(self, dec_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_up, skip):
        skip = self.skip_conv(skip)
        return self.fuse(torch.cat([x_up, skip], dim=1))


class PHDSeg(nn.Module):

    def __init__(self, pretrained=True, use_ffe=False):
        super().__init__()
        self.use_ffe = use_ffe

        _tmp = smp.Unet(
            encoder_name='mit_b2',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3, classes=1,
        )
        self.encoder = _tmp.encoder
        del _tmp

        enc_ch = [64, 128, 320, 512]
        dec_ch = [64, 128, 256, 512]

        if use_ffe:
            self.ffe = nn.ModuleList([FFEModule(c) for c in enc_ch])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_ch[3], dec_ch[3], 1, bias=False),
            nn.BatchNorm2d(dec_ch[3]),
            nn.ReLU(inplace=True),
        )

        self.up4_proj = nn.Sequential(
            nn.Conv2d(dec_ch[3], dec_ch[2], 1, bias=False),
            nn.BatchNorm2d(dec_ch[2]),
            nn.ReLU(inplace=True),
        )
        self.dec3 = DecoderBlock(dec_ch[2], enc_ch[2], dec_ch[2])

        self.up3_proj = nn.Sequential(
            nn.Conv2d(dec_ch[2], dec_ch[1], 1, bias=False),
            nn.BatchNorm2d(dec_ch[1]),
            nn.ReLU(inplace=True),
        )
        self.dec2 = DecoderBlock(dec_ch[1], enc_ch[1], dec_ch[1])

        self.up2_proj = nn.Sequential(
            nn.Conv2d(dec_ch[1], dec_ch[0], 1, bias=False),
            nn.BatchNorm2d(dec_ch[0]),
            nn.ReLU(inplace=True),
        )
        self.dec1 = DecoderBlock(dec_ch[0], enc_ch[0], dec_ch[0])

        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_ch[0], dec_ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch[0], 1, 1),
        )

    def forward(self, x):
        B, _, H, W = x.shape

        enc_feats = self.encoder(x)
        f1 = enc_feats[2]
        f2 = enc_feats[3]
        f3 = enc_feats[4]
        f4 = enc_feats[5]

        if self.use_ffe:
            f1 = self.ffe[0](f1)
            f2 = self.ffe[1](f2)
            f3 = self.ffe[2](f3)
            f4 = self.ffe[3](f4)

        d4 = self.bottleneck(f4)

        d4_proj = self.up4_proj(d4)
        d4_up = F.interpolate(d4_proj, size=f3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(d4_up, f3)

        d3_proj = self.up3_proj(d3)
        d3_up = F.interpolate(d3_proj, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d3_up, f2)

        d2_proj = self.up2_proj(d2)
        d2_up = F.interpolate(d2_proj, size=f1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(d2_up, f1)

        seg = self.seg_head(d1)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)

        return seg
