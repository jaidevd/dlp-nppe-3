"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, et al
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttentionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ChannelAttentionBlock, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = ChannelAttentionLayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class SupervisedAttention(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SupervisedAttention, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class Encoder(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff
    ):
        super(Encoder, self).__init__()

        self.encoder_level1 = [
            ChannelAttentionBlock(n_feat, kernel_size, reduction,
                                  bias=bias, act=act) for _ in range(2)
        ]
        self.encoder_level2 = [
            ChannelAttentionBlock(n_feat + scale_unetfeats, kernel_size,
                                  reduction, bias=bias, act=act)
            for _ in range(2)
        ]
        self.encoder_level3 = [
            ChannelAttentionBlock(
                n_feat + (scale_unetfeats * 2),
                kernel_size,
                reduction,
                bias=bias,
                act=act,
            )
            for _ in range(2)
        ]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(
                n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(
                n_feat + scale_unetfeats,
                n_feat + scale_unetfeats,
                kernel_size=1,
                bias=bias,
            )
            self.csff_enc3 = nn.Conv2d(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )

            self.csff_dec1 = nn.Conv2d(
                n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(
                n_feat + scale_unetfeats,
                n_feat + scale_unetfeats,
                kernel_size=1,
                bias=bias,
            )
            self.csff_dec3 = nn.Conv2d(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 += self.csff_enc1(
                encoder_outs[0]
            ) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 += self.csff_enc2(
                encoder_outs[1]
            ) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 += self.csff_enc3(
                encoder_outs[2]
            ) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias,
                 scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [
            ChannelAttentionBlock(n_feat, kernel_size, reduction,
                                  bias=bias, act=act) for _ in range(2)
        ]
        self.decoder_level2 = [
            ChannelAttentionBlock(n_feat + scale_unetfeats, kernel_size,
                                  reduction, bias=bias, act=act)
            for _ in range(2)
        ]
        self.decoder_level3 = [
            ChannelAttentionBlock(
                n_feat + (scale_unetfeats * 2),
                kernel_size,
                reduction,
                bias=bias,
                act=act,
            )
            for _ in range(2)
        ]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = ChannelAttentionBlock(n_feat, kernel_size,
                                                reduction, bias=bias, act=act)
        self.skip_attn2 = ChannelAttentionBlock(
            n_feat + scale_unetfeats, kernel_size,
            reduction, bias=bias, act=act
        )

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(
                scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels, in_channels + s_factor, 1, stride=1,
                padding=0, bias=False
            ),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels + s_factor, in_channels, 1, stride=1, padding=0,
                bias=False
            ),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels + s_factor, in_channels, 1, stride=1, padding=0,
                bias=False
            ),
        )

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class OrgResolutionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(OrgResolutionBlock, self).__init__()
        modules_body = []
        modules_body = [
            ChannelAttentionBlock(n_feat, kernel_size, reduction,
                                  bias=bias, act=act)
            for _ in range(num_cab)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(
        self,
        n_feat,
        scale_orsnetfeats,
        kernel_size,
        reduction,
        act,
        bias,
        scale_unetfeats,
        num_cab,
    ):
        super(ORSNet, self).__init__()

        self.orb1 = OrgResolutionBlock(
            n_feat + scale_orsnetfeats, kernel_size, reduction,
            act, bias, num_cab
        )
        self.orb2 = OrgResolutionBlock(
            n_feat + scale_orsnetfeats, kernel_size, reduction,
            act, bias, num_cab
        )
        self.orb3 = OrgResolutionBlock(
            n_feat + scale_orsnetfeats, kernel_size, reduction,
            act, bias, num_cab
        )

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats),
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats),
        )

        self.conv_enc1 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_enc2 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_enc3 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )

        self.conv_dec1 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_dec2 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_dec3 = nn.Conv2d(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x += self.conv_enc1(
            encoder_outs[0]
        ) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = (
            x
            + self.conv_enc2(self.up_enc1(encoder_outs[1]))
            + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        )

        x = self.orb3(x)
        x = (
            x
            + self.conv_enc3(self.up_enc2(encoder_outs[2]))
            + self.conv_dec3(self.up_dec2(decoder_outs[2]))
        )

        return x


class MPRNet(nn.Module):
    def __init__(
        self,
        in_c=3,
        out_c=3,
        n_feat=80,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
        kernel_size=3,
        reduction=4,
        bias=False,
    ):
        super(MPRNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            ChannelAttentionBlock(n_feat, kernel_size, reduction, bias=bias,
                                  act=act),
        )
        self.shallow_feat2 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            ChannelAttentionBlock(n_feat, kernel_size, reduction, bias=bias,
                                  act=act),
        )
        self.shallow_feat3 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            ChannelAttentionBlock(n_feat, kernel_size, reduction, bias=bias,
                                  act=act),
        )

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats,
            csff=False
        )
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats
        )

        self.stage2_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats,
            csff=True
        )
        self.stage2_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats
        )

        self.stage3_orsnet = ORSNet(
            n_feat,
            scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            num_cab,
        )

        self.sam12 = SupervisedAttention(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SupervisedAttention(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(
            n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias
        )
        self.tail = conv(n_feat + scale_orsnetfeats, out_c,
                         kernel_size, bias=bias)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0: int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2): H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0: int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2): W]
        x1lbot_img = x2bot_img[:, :, :, 0: int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2): W]

        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        feat1_top = [
            torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [
            torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        res2 = self.stage2_decoder(feat2)

        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        x3 = self.shallow_feat3(x3_img)

        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return [stage3_img + x3_img, stage2_img, stage1_img]
