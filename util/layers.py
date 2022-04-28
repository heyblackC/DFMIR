# py
import functools

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_filter(filt_size=3):

    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_filter3D(filt_size=3):

    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None, None] * a[None, :, None] * a[None, None,:])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

def get_pad_layer3D(pad_type, size):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ConstantPad3d(size, -1.)
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad3d
    elif(pad_type == 'zero'):
        PadLayer = nn.ConstantPad3d(size, 0)
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

########################################
##   Basic Blocks
########################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ConstantPad3d(1, -1.0)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block +=[nn.ConstantPad3d(1, -1.0)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

class Upsample3D(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample3D, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter3D(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose3d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1, :-1]

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Downsample3D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample3D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter3D(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

        self.pad = get_pad_layer3D(pad_type, self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class BaseConvBlock2D(nn.Module):

    def initialize_padding(self, pad_type, padding):
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn':
            self.norm_layer = functools.partial(nn.BatchNorm2d(norm_dim), affine=True, track_running_stats=True)
        elif norm_layer == 'bn_raw':
            self.norm_layer = nn.BatchNorm2d(norm_dim)
        elif norm_layer == 'in':
            self.norm_layer = functools.partial(nn.InstanceNorm2d(norm_dim, affine=False, track_running_stats=False))
        elif norm_layer == 'in_raw':
            self.norm_layer = nn.InstanceNorm2d(norm_dim, affine=False, track_running_stats=False)
        elif norm_layer == 'ln':
            self.norm_layer = LayerNorm(norm_dim)
        elif norm_layer == 'adain':
            self.norm_layer = AdaptiveInstanceNorm2d(norm_dim)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

class BaseConvBlock3D(nn.Module):

    def initialize_padding(self, pad_type, padding):
        # initialize padding
        if pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ConstantPad3d(padding, 0.0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn':
            self.norm_layer = functools.partial(nn.BatchNorm3d(norm_dim), affine=True, track_running_stats=True)
        elif norm_layer == 'bn_raw':
            self.norm_layer = nn.BatchNorm3d(norm_dim)
        elif norm_layer == 'in':
            self.norm_layer = functools.partial(nn.InstanceNorm3d(norm_dim, affine=False, track_running_stats=False))
        elif norm_layer == 'in_raw':
            self.norm_layer = nn.InstanceNorm3d(norm_dim, affine=False, track_running_stats=False)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


class ConvBlock2D(BaseConvBlock2D):
    '''
    2D ConvlutionBlock performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param Conv2D input parameters: see nn.Conv2D
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_size=3, padding=0, stride=1, bias=True,
                 norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        # initialize padding
        self.initialize_padding(pad_type, padding)
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)
        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,  stride=stride, bias=bias)


    def forward(self, inputs):
        outputs = self.conv_layer(self.pad(inputs))
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ConvBlock3D(BaseConvBlock3D):
    '''
    2D ConvlutionBlock performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param Conv2D input parameters: see nn.Conv2D
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_size=3, padding=0, stride=1, bias=True,
                 norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        # initialize padding
        self.initialize_padding(pad_type, padding)
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)
        self.conv_layer = nn.Conv3d(input_filters, output_filters, kernel_size=kernel_size,  stride=stride, bias=bias)


    def forward(self, inputs):
        outputs = self.conv_layer(self.pad(inputs))
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class ConvTransposeBlock2D(BaseConvBlock2D):
    '''
    2D ConvTransposeBlock2D performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param ConvTranspose2D input parameters: see nn.ConvTranspose2d
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_sizeT=4, kernel_size=3, output_padding=0, padding=0,
                 stride=2, bias=True, norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        self.initialize_padding(pad_type, padding, int(np.floor((kernel_size-1)/2)))
        self.initialize_normalization(norm_layer,norm_dim=output_filters)
        self.initialize_activation(activation)

        self.convT_layer = nn.ConvTranspose2d(input_filters, input_filters, kernel_size=kernel_sizeT,
                                              output_padding=output_padding, stride=stride, bias=bias)

        self.conv_layer = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size,
                                    stride=1, bias=bias)


    def initialize_padding(self, pad_type, padding1, padding2):
        # initialize padding
        if pad_type == 'reflect':
            self.pad1 = nn.ReflectionPad2d(padding1)
            self.pad2 = nn.ReflectionPad2d(padding2)

        elif pad_type == 'replicate':
            self.pad1 = nn.ReplicationPad2d(padding1)
            self.pad2 = nn.ReplicationPad2d(padding2)

        elif pad_type == 'zeros':
            self.pad1 = nn.ZeroPad2d(padding1)
            self.pad2 = nn.ZeroPad2d(padding2)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)


    def forward(self, inputs):
        outputs = self.convT_layer(self.pad1(inputs))
        outputs = self.conv_layer(self.pad2(outputs))

        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class UpConv(nn.Module):

    def __init__(self, input_filters, output_filters, kernel_size=3, stride=2,
                 bias=True, norm_layer='bn', activation='relu', mode='bilinear', pad_type='zeros'):

        super().__init__()
        self.up_layer = nn.Upsample(scale_factor=stride, mode=mode)

        self.conv_layer = ConvBlock2D(input_filters, output_filters, kernel_size=kernel_size, padding=1, stride=1,
                                      norm_layer=norm_layer, activation=activation, bias=bias, pad_type=pad_type)


    def forward(self, inputs):
        outputs = self.up_layer(inputs)
        outputs =  self.conv_layer(outputs)

        return outputs

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm_layer == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm_layer == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode='border'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, flow, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if 'padding_mode' in kwargs:
            self.padding_mode = kwargs['padding_mode']
        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        new_locs = self.grid + flow
        shape = flow.shape[2:]


        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)

class SpatialTransformerAffine(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample

    This is copied from voxelmorph code, so for more information and credit
    visit https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """

    def __init__(self, size, mode='bilinear', padding_mode='border', torch_dtype = torch.float):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        ndims = len(size)

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = grid.type(torch_dtype)


        flat_mesh = torch.reshape(grid, (ndims,-1))
        ones_vec = torch.ones((1, np.prod(size))).type(torch_dtype)
        mesh_matrix = torch.cat((flat_mesh, ones_vec), dim=0)

        # grid = torch.unsqueeze(grid, 0)  # add batch
        # grid = grid.type(torch_dtype)
        # self.register_buffer('grid', grid)

        mesh_matrix = mesh_matrix.type(torch_dtype)
        self.register_buffer('mesh_matrix', mesh_matrix)

        self.size = size
        self.mode = mode
        self.padding_mode = padding_mode
        self.torch_dtype = torch_dtype

    def _get_locations(self, affine_matrix):
        batch_size = affine_matrix.shape[0]
        ndims = len(self.size)
        vol_shape = self.size


        # compute locations
        loc_matrix = torch.matmul(affine_matrix, self.mesh_matrix)  # N x nb_voxels
        loc = torch.reshape(loc_matrix[:,:ndims], [batch_size, ndims] + list(vol_shape))  # *volshape x N

        return loc.float()

    def forward(self, src, affine_matrix, **kwargs):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image of size [batch_size, n_dims, *volshape]
            :param flow: the output from the U-Net [batch_size, n_dims, *volshape]
        """

        if 'mode' in kwargs:
            self.mode = kwargs['mode']

        affine_matrix = affine_matrix.type(self.torch_dtype)

        new_locs = self._get_locations(affine_matrix)
        new_locs = new_locs.type(self.torch_dtype)

        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
        else:
            shape = new_locs.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]


        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode=self.padding_mode)

class RescaleTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, factor=None, target_size=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor:
                :param latent_size: it only applies if factor is None

        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag

        if factor is None:
            assert target_size is not None
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])
        elif isinstance(factor, list) or isinstance(factor, tuple):
            self.factor = list(factor)
        else:
            self.factor = [factor for _ in range(self.ndims)]

        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if self.factor[0] < 1 and self.gaussian_filter_flag:
            kernel_sigma = [0.44 * 1 / f for f in self.factor]

            if self.ndims == 2:
                kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)
            elif self.ndims == 3:
                kernel = self.gaussian_filter_3d(kernel_sigma=kernel_sigma)
            else:
                raise ValueError('[RESCALE TF] No valid kernel found.')
            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def gaussian_filter_3d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX, ZZ = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
        xyz_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis], ZZ[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1, 1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)
        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xyz_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1],kernel_size[2])

        # Total kernel

        total_kernel = np.zeros((3,3) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel
        total_kernel[2, 2] = kernel


        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x):

        # x = x.clone()
        if self.factor[0] < 1:
            if self.gaussian_filter_flag:
                padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
                if self.ndims == 2:
                    x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)
                else:
                    x = F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=padding)

            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]

        elif self.factor[0] > 1:
            # multiply first to save memory
            for i in range(self.ndims):
                x[:, i] = x[:, i] * self.factor[i]
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    Credit to voxelmorph: https://github.com/voxelmorph/voxelmorph/blob/redesign/voxelmorph/torch/layers.py
    """

    def __init__(self, inshape, target_size=None, factor=None, gaussian_filter_flag=True):
        '''

        :param vol_size:
        :param factor: if factor<1 the shape is reduced and viceversa.
        :param latent_size: it only applies if factor is None
        '''
        super().__init__()

        self.ndims = len(inshape)
        self.mode = 'linear'
        self.inshape = inshape
        self.gaussian_filter_flag = gaussian_filter_flag
        if self.ndims == 2:
            self.mode = 'bi' + self.mode
        elif self.ndims == 3:
            self.mode = 'tri' + self.mode

        if target_size is None:
            self.factor = factor
            if isinstance(factor, float) or isinstance(factor, int):
                self.factor = [factor for _ in range(self.ndims)]
        else:
            self.factor = tuple([b/a for a, b in zip(inshape, target_size)])

        if self.factor[0] < 1 and self.gaussian_filter_flag:

            kernel_sigma = [0.44 / f for f in self.factor]
            if self.ndims == 2:
                kernel = self.gaussian_filter_2d(kernel_sigma=kernel_sigma)
            elif self.ndims == 3:
                kernel = self.gaussian_filter_3d(kernel_sigma=kernel_sigma)
            else:
                raise ValueError('[RESCALE TF] No valid kernel found.')
            self.register_buffer('kernel', kernel)

    def gaussian_filter_2d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))


        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX = np.meshgrid(coord[0], coord[1], indexing='ij')
        xy_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1,1,1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1,1,1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)

        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xy_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1])

        # Total kernel
        total_kernel = np.zeros((2, 2) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel

        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def gaussian_filter_3d(self, kernel_sigma):

        if isinstance(kernel_sigma, list):
            kernel_size = [int(np.ceil(ks*3) + np.mod(np.ceil(ks*3) + 1, 2)) for ks in kernel_sigma]

        else:
            kernel_size = int(np.ceil(kernel_sigma*3) + np.mod(np.ceil(kernel_sigma*3) + 1, 2))

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        coord = [np.arange(ks) for ks in kernel_size]

        YY, XX, ZZ = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
        xyz_grid = np.concatenate((YY[np.newaxis], XX[np.newaxis], ZZ[np.newaxis]), axis=0)  # 2, y, x

        mean = np.asarray([(ks - 1) / 2. for ks in kernel_size])
        mean = mean.reshape(-1, 1, 1, 1)
        variance = np.asarray([ks ** 2. for ks in kernel_sigma])
        variance = variance.reshape(-1, 1, 1, 1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        # 2.506628274631 = sqrt(2 * pi)
        norm_kernel = (1. / (np.sqrt(2 * np.pi) ** 2 + np.prod(kernel_sigma)))
        kernel = norm_kernel * np.exp(-np.sum((xyz_grid - mean) ** 2. / (2 * variance), axis=0))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / np.sum(kernel)

        # Reshape
        kernel = kernel.reshape(1, 1, kernel_size[0], kernel_size[1],kernel_size[2])

        # Total kernel

        total_kernel = np.zeros((3,3) + tuple(kernel_size))
        total_kernel[0, 0] = kernel
        total_kernel[1, 1] = kernel
        total_kernel[2, 2] = kernel


        total_kernel = torch.from_numpy(total_kernel).float()

        return total_kernel

    def forward(self, x, *args, **kwargs):

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode=self.mode

        if self.gaussian_filter_flag and self.factor[0] < 1:
            padding = [int((s - 1) // 2) for s in self.kernel.shape[2:]]
            if self.ndims == 2:
                x = F.conv2d(x, self.kernel, stride=(1, 1), padding=padding)
            else:
                x = F.conv3d(x, self.kernel, stride=(1, 1, 1), padding=padding)

        x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=mode)

        return x

class VecInt(nn.Module):
    """
    Vector Integration Layer

    Enables vector integration via several methods
    (ode or quadrature for time-dependent vector fields,
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, field_shape, int_steps=7, **kwargs):
        """
        Parameters:
            int_steps is the number of integration steps
        """
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1 / (2 ** self.int_steps)
        self.transformer = SpatialTransformer(field_shape)

    def forward(self, field, **kwargs):

        output = field
        output = output * self.scale

        for _ in range(self.int_steps):
            output = output + self.transformer(output, output)

        return output

class AffineTransformer(nn.Module):
    def __init__(self, vol_shape, input_channels, enc_features):
        super(AffineTransformer, self).__init__()

        # Spatial transformer localization-network
        out_shape = [v for v in vol_shape]
        nf_list = [input_channels] + enc_features
        localization_layers = []
        for in_nf, out_nf in zip(nf_list[:-1], nf_list[1:]):
            localization_layers.append(nn.Conv2d(in_nf, out_nf, kernel_size=3, stride=2, padding=1))
            localization_layers.append(nn.LeakyReLU(0.2))
            out_shape = [o/2 for o in out_shape]

        self.localization = nn.Sequential(*localization_layers)
        self.out_shape = int(enc_features[-1]*np.prod(out_shape))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.out_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 3 * 2)
        )

    # Spatial transformer network forward function
    def forward(self, x):
        x_floating = x[:,0:1]
        xs = self.localization(x)
        xs = xs.view(-1, self.out_shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x_floating.size())

        return F.grid_sample(x_floating, grid), theta


########################################
##   Normalization layers
########################################
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
