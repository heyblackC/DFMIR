import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from . import layers
from .modelio import LoadableModel, store_config_args
from torch.nn import functional as F

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
        '''

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class ConvBlock2(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

from util.trans_model import *
class Unet_Transformer(nn.Module):
    def __init__(self, inshape, config, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        '''

        # build feature list automatically
        self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        prev_nf2 = 1
        self.downarm2 = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm2.append(ConvBlock2(ndims, prev_nf2, nf, stride=2))
            prev_nf2 = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf*2
            self.uparm.append(ConvBlock2(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))

        # self.image_encoder = LidarEncoder(512, in_channels=1)
        # self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=1)

        self.transformer1 = GPT(n_embd=16,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer2 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer5 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

        self.trans_list = \
            [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]
        #self.fuse_list = nn.ModuleList()
        #self.fuse_list.append(nn.Conv2d(16*2,16,1,1))
        #self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        #self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        #self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        #self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))



    def forward(self, x, y):
        import math
        # get encoder activations
        x_enc = [x]
        y_enc = [y]
        xy_fuse = [torch.cat([x,y],dim=1)]
        for i,layer in enumerate(self.downarm):
            tmp = layer(x_enc[-1])
            tmp2 = self.downarm2[i](y_enc[-1])
            #print(tmp.size())
            #print(tmp2.size())

            image_embd_layer1 = self.avgpool(tmp)
            lidar_embd_layer1 = self.avgpool(tmp2)
            image_features_layer1, lidar_features_layer1 = self.trans_list[i](image_embd_layer1, lidar_embd_layer1, None)
            image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            tmp = tmp + image_features_layer1
            tmp2 = tmp2 + lidar_features_layer1

            #torch.cat([tmp,tmp2],dim=1)
            x_enc.append(tmp)
            y_enc.append(tmp2)
            #xy_fuse.append(self.fuse_list[i](torch.cat([tmp,tmp2],dim=1)))
            xy_fuse.append(torch.cat([tmp,tmp2],dim=1))


        # conv, upsample, concatenate series
        x = xy_fuse.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, xy_fuse.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class Whole_Transformer(nn.Module):
    def __init__(self, inshape, config, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        '''

        # build feature list automatically
        self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        prev_nf2 = 1
        self.downarm2 = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm2.append(ConvBlock2(ndims, prev_nf2, nf, stride=2))
            prev_nf2 = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock2(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))

        self.transformer1 = GPT(n_embd=16,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer2 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer5 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

        self.transformer6 = GPT(n_embd=64,
                                n_head=config.n_head,#4
                                block_exp=config.block_exp,#4
                                n_layer=config.n_layer,#8
                                vert_anchors=config.vert_anchors,#8
                                horz_anchors=config.horz_anchors,#8
                                seq_len=config.seq_len,#1
                                embd_pdrop=config.embd_pdrop,#0.1
                                attn_pdrop=config.attn_pdrop,#0.1
                                resid_pdrop=config.resid_pdrop,#0.1
                                config=config)
        self.transformer7 = GPT(n_embd=64,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer8 = GPT(n_embd=32,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer9 = GPT(n_embd=16,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        # self.transformer10 = GPT(n_embd=16,
        #                         n_head=config.n_head,
        #                         block_exp=config.block_exp,
        #                         n_layer=config.n_layer,
        #                         vert_anchors=config.vert_anchors,
        #                         horz_anchors=config.horz_anchors,
        #                         seq_len=config.seq_len,
        #                         embd_pdrop=config.embd_pdrop,
        #                         attn_pdrop=config.attn_pdrop,
        #                         resid_pdrop=config.resid_pdrop,
        #                         config=config)

        self.trans_list = \
            [self.transformer1, self.transformer2, self.transformer3, self.transformer4, self.transformer5]
        self.trans_list_skip = \
            [self.transformer6, self.transformer7, self.transformer8, self.transformer9]

        self.fuse_list = nn.ModuleList()
        self.fuse_list.append(nn.Conv2d(16*2,16,1,1))
        self.fuse_list.append(nn.Conv2d(32 * 2, 32, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))
        self.fuse_list.append(nn.Conv2d(64 * 2, 64, 1, 1))


    def forward(self, x, y):
        import math
        # get encoder activations
        x_enc = [x]
        y_enc = [y]
        xy_fuse = [torch.cat([x,y],dim=1)]
        for i,layer in enumerate(self.downarm):
            tmp = layer(x_enc[-1])
            tmp2 = self.downarm2[i](y_enc[-1])
            #print(tmp.size())
            #print(tmp2.size())

            image_embd_layer1 = self.avgpool(tmp)
            lidar_embd_layer1 = self.avgpool(tmp2)
            image_features_layer1, lidar_features_layer1 = self.trans_list[i](image_embd_layer1, lidar_embd_layer1, None)
            image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=16/int(math.pow(2,i)), mode='bilinear')
            tmp = tmp + image_features_layer1
            tmp2 = tmp2 + lidar_features_layer1

            #torch.cat([tmp,tmp2],dim=1)
            x_enc.append(tmp)
            y_enc.append(tmp2)
            xy_fuse.append(self.fuse_list[i](torch.cat([tmp,tmp2],dim=1)))


        # conv, upsample, concatenate series
        x = xy_fuse.pop()
        for i,layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            skip_feat = xy_fuse.pop()
            # print(x.size())
            # print(skip_feat.size())
            if i<len(self.trans_list_skip):
                image_embd_layer1 = self.avgpool(x)
                lidar_embd_layer1 = self.avgpool(skip_feat)
                image_features_layer1, lidar_features_layer1 = self.trans_list_skip[i](image_embd_layer1, lidar_embd_layer1, None)
                image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=2* int(math.pow(2, i)),
                                                      mode='bilinear')
                lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=2* int(math.pow(2, i)),
                                                      mode='bilinear')
                x = x + image_features_layer1
                skip_feat = skip_feat + lidar_features_layer1
            x = torch.cat([x, skip_feat], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class DualUnet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
        '''
        nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
        '''

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.fusion = nn.ModuleList()
        self.fusion.append(nn.Conv2d(256*2,256,1,1))
        self.fusion.append(nn.Conv2d(128*2,128,1,1))
        self.fusion.append(nn.Conv2d(64*2,64,1,1))

    def forward_(self, x, x_enc2):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 输出结果的排列顺序：以 [16, 32, 64, 128, 256],  # encoder为例
        # 2 -> 16 -> 32 -> 64 ->128 -> 256 共5次卷积
        # 224*224*2 -> 112*112*16 -> 56*56*32 -> 28*28*64 -> 14*14*128 -> 7*7*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256,

        # [128, 256, 256]  # encoder
        # [256, 128, 64, 16, 8]  # decoder
        # conv, upsample, concatenate series
        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop(), x_enc2.pop()], dim=1)

        # decoder   [128, 64, 32, 16, 16, 8]  # decoder
        # 这里的变化为：从7*7*256变为14*14* 128（叠加第1次，）->64(叠加1次) -> 32(叠加一次) 16 (叠加一次)， 最后一次叠加是224*224*2和224*224*16的叠加，没有卷积了
        # 1. 7*7*256 -> 14*14*128, 而且 把concat操作完成了 14*14*256
        # 2. 14*14*64 -> 28*28*64
        # 224*224*16
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8

        return x

    def forward(self, x, x_enc2, x_enc3):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 出来的是 224*224*2, 112*112*128, 56*56*256, 28*28*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256

        # [128, 256, 256]  # encoder
        # [256, 128, 64, 16, 8]  # decoder
        # conv, upsample, concatenate series
        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            x_enc_fused = self.fusion[i](torch.cat([x_enc2.pop(),x_enc3.pop()],dim=1))
            x = torch.cat([x, x_enc.pop(), x_enc_fused], dim=1)

        # decoder   [128, 64, 32, 16, 16, 8]  # decoder
        # 这里的变化为：从7*7*256变为14*14* 128（叠加第1次，）->64(叠加1次) -> 32(叠加一次) 16 (叠加一次)， 最后一次叠加是224*224*2和224*224*16的叠加，没有卷积了
        # 1. 7*7*256 -> 14*14*128, 而且 把concat操作完成了 14*14*256
        # 2. 14*14*64 -> 28*28*64
        # 224*224*16
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8


class AttentionNet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            pass
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        self.fusion = nn.ModuleList()
        # self.fusion.append(nn.Conv2d(256*2,256,1,1))
        # self.fusion.append(nn.Conv2d(128*2,128,1,1))
        # self.fusion.append(nn.Conv2d(64*2,64,1,1))
        self.fusion.append(NLBlockND_cross(256, dimension=2))
        #self.fusion.append(nn.Conv2d(128 * 2, 128, 1, 1))
        self.fusion.append(NLBlockND_cross(128, dimension=2))
        self.fusion.append(nn.Conv2d(64*2,64,1,1))
        #self.fusion.append(NLBlockND_cross(128, dimension=2))
        #self.fusion.append(NLBlockND_cross(64, dimension=2))
        self.activation_atten = nn.LeakyReLU(0.2)

    def forward(self, x, x_enc2, x_enc3):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # 出来的是 224*224*2, 112*112*128, 56*56*256, 28*28*256
        # 能给到的feature map为 224*224*2, 224*224*64, 112*112*128, 56*56*256


        x = x_enc.pop()
        # x2 = x_enc2.pop()
        # x = self.conv_combine(torch.cat([x,x2],dim=1))
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)
            tmp1 = x_enc2.pop()
            tmp2 = x_enc3.pop()
            if i ==0 or i == 1:
                x_enc_atten1 = self.activation_atten(self.fusion[i](tmp1,tmp2))
                x_enc_atten2 = self.activation_atten(self.fusion[i](tmp2,tmp1))
                x_enc_fused = torch.cat([x_enc_atten1, x_enc_atten2],dim=1)
            else:
                x_enc_fused = self.fusion[i](torch.cat([tmp1, tmp2], dim=1))
            x = torch.cat([x, x_enc.pop(), x_enc_fused], dim=1)
        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8


class DecoderNet(nn.Module):
    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()


        # build feature list automatically
        if isinstance(nb_features, int):
            pass
            # if nb_levels is None:
            #     raise ValueError('must provide unet nb_levels if nb_features is an integer')
            # feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            # self.enc_nf = feats[:-1]
            # self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            # 通常情况下，enc_nf取第一行，dec_nf取第二行
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        # 输入的xy图像叠加到一起，因此是两个channel的
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        # 最后，prev_nf就变成了最终的encoder输出的channel

        # self.conv_combine = ConvBlock(ndims, prev_nf*2,prev_nf, stride=1)

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # 解码器和编码器保持同样的深度
            channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf*2 + 64
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        # self.fusion = nn.ModuleList()
        # self.fusion.append(nn.Conv2d(256*2,256,1,1))
        # self.fusion.append(nn.Conv2d(128*2,128,1,1))
        # self.fusion.append(nn.Conv2d(64*2,64,1,1))
        self.conv1 = ConvBlock(ndims, 256*2,256,stride=1)
        self.conv2 = ConvBlock(ndims, 256,256,stride=2)

    def forward(self, x_enc1, x_enc2):

        out1 = self.conv1(torch.cat([x_enc1[-1],x_enc2[-1]], dim=1))
        x = self.conv2(out1)

        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc1.pop(), x_enc2.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        # 最后多一个8，就是224*224*16+2进去，输出224*224*8

        return x



class VxmAttentionNet(LoadableModel):

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):

        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = AttentionNet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc, enc3, registration=False):

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = self.unet_model(x, enc)
        x = self.unet_model(x, enc, enc3)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class VxmDecoderDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = DecoderNet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc1, enc2, registration=False):

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(enc1,enc2)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field

class VxmDenseTransformer(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        config = GlobalConfig()

        # configure core unet model
        self.unet_model = Unet_Transformer(
            inshape,
            config,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        ).to('cuda')

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow





class VxmDenseTransformerWhole(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        config = GlobalConfig()

        # configure core unet model
        self.unet_model = Whole_Transformer(
            inshape,
            config,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        ).to('cuda')

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x = self.unet_model(source,target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

class VxmDenseDual(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = DualUnet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, enc, enc3, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        # x = self.unet_model(x, enc)
        x = self.unet_model(x, enc, enc3)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
