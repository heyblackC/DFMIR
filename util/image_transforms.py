import warnings
import copy
import random

from scipy.interpolate import griddata
import numpy as np
from cv2 import resize, INTER_LINEAR, INTER_NEAREST

from util.image_utils import bilinear_interpolate, deform2D
from util.image_utils import get_affine_from_rotation_2d, get_affine_from_rotation_3d


def get_deformation_field(affine_params, nonlinear_params, _image_shape, num=2):
    affine_list = []
    nonlinear_field_list = []

    for it_i in range(num):
        affine = affine_params.get_affine(_image_shape)
        nlf_x, nlf_y = nonlinear_params.get_lowres_strength(ndim=2)
        nonlinear_field = np.zeros((2,) + nlf_x.shape)
        nonlinear_field[0] = nlf_y
        nonlinear_field[1] = nlf_x

        affine_list.append(affine)
        nonlinear_field_list.append(nonlinear_field)

    return affine_list, nonlinear_field_list

##############################################################################################
###################################  PARAMETERS  #############################################
##############################################################################################
class ResizeParams(object):
    def __init__(self, resize_shape):
        if isinstance(resize_shape, int):
            resize_shape = (resize_shape, resize_shape)
        self.resize_shape = resize_shape

class CropParams(object):
    def __init__(self, crop_shape, init_coordinates = None):
        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape)
        self.crop_shape = crop_shape
        self.init_coordinates = init_coordinates

class FlipParams(object):
    pass

class PadParams(object):
    def __init__(self, psize, pfill=0, pmode='constant', dim=2):
        if isinstance(psize, int):
            psize = (psize, psize)
        self.psize = psize
        self.pmode = pmode
        self.pfill = pfill
        self.dim = dim

class NonLinearParams(object):
    def __init__(self, lowres_size, lowres_strength=1, distribution = 'normal', nstep=5):
        self.lowres_size = lowres_size
        self.lowres_strength = lowres_strength
        self.distribution = distribution
        self.nstep = nstep

    def get_lowres_strength(self, ndim=2):

        size = 1
        if self.distribution == 'normal':
            mean, std = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean

        elif self.distribution == 'uniform':
            high, low = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.rand(size) * (high - low) + low

        elif self.distribution == 'lognormal':
            mean, std = self.lowres_strength[1], self.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean
            lowres_strength = np.exp(lowres_strength)

        elif self.distribution is None:
            lowres_strength = [self.lowres_strength] * size

        else:
            raise ValueError("[src/utils/transformations: NonLinearDeformation]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        if ndim ==2:
            field_lowres_x = lowres_strength * np.random.randn(self.lowres_size[0],
                                                               self.lowres_size[1])  # generate random noise.

            field_lowres_y = lowres_strength * np.random.randn(self.lowres_size[0],
                                                               self.lowres_size[1])  # generate random noise.

            return field_lowres_x, field_lowres_y

        else:
            field_lowres_x = lowres_strength * np.random.randn(self.lowres_size[0],
                                                               self.lowres_size[1],
                                                               self.lowres_size[2])  # generate random noise.

            field_lowres_y = lowres_strength * np.random.randn(self.lowres_size[0],
                                                               self.lowres_size[1],
                                                               self.lowres_size[2])  # generate random noise.

            field_lowres_z = lowres_strength * np.random.randn(self.lowres_size[0],
                                                               self.lowres_size[1],
                                                               self.lowres_size[2])  # generate random noise.
            return field_lowres_x, field_lowres_y, field_lowres_z

class RotationParams(object):

    def __init__(self, value_range, distribution='uniform'):
        self.value_range = value_range
        self.distribution = distribution

    def get_affine(self, ndim=2):

        assert ndim in [2,3]
        size = 1 if ndim == 2 else 3

        if self.distribution == 'normal':
            mean, std = self.value_range[1], self.value_range[0]
            angle = np.random.randn(size) * std + mean

        elif self.distribution == 'uniform':
            high, low = self.value_range[1], self.value_range[0]
            angle = np.random.rand(size) * (high - low) + low

        elif self.distribution is None:
            angle = self.value_range[1] * np.ones(size)

        else:
            raise ValueError("[src/utils/transformations: MultiplicativeNoise]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        if ndim == 2:
            return get_affine_from_rotation_2d(angle)
        elif ndim == 3:
            return get_affine_from_rotation_3d(angle)


class AffineParams(object):

    def __init__(self, rotation, scaling, translation):
        self.rotation = rotation
        self.scaling = scaling
        self.translation = translation

    def get_angles(self):
        angles = []
        for r in self.rotation:
            angles.append((2 * np.random.rand(1) - 1) * r/180*np.pi)

        return angles

    def get_scaling(self):
        scales = []
        for s in self.scaling:
            scales.append(1 + (2 * np.random.rand(1) - 1)*s)

        return scales

    def get_translation(self):
        tranlsation = []
        for t in self.translation:
            tranlsation.append((2 * np.random.rand(1) - 1) * t)

        return tranlsation

    def get_affine(self, image_shape):
        if len(image_shape) == 2:
            return self._get_affine_2d(image_shape)
        else:
            return self._get_affine_3d(image_shape)

    def _get_affine_2d(self, image_shape):
        T1 = np.eye(3)
        T2 = np.eye(3)
        T3 = np.eye(3)
        T4 = np.eye(3)
        T5 = np.eye(3)

        cr = [i/2 for i in image_shape]
        scaling = self.get_scaling()
        angles = self.get_angles()
        translation = self.get_translation()

        T1[0, 2] = -cr[0]
        T1[1, 2] = -cr[1]

        T2[0, 0] = scaling[0]
        T2[1, 1] = scaling[1]

        T3[0, 0] = np.cos(angles[0])
        T3[0, 1] = -np.sin(angles[0])
        T3[1, 0] = np.sin(angles[0])
        T3[1, 1] = np.cos(angles[0])

        T4[0, 2] = cr[0]
        T4[1, 2] = cr[1]

        T5[0, 2] = translation[0]
        T5[1, 2] = translation[1]

        return T5 @ T4 @ T3 @ T2 @ T1

    def _get_affine_3d(self, image_shape):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T3 = np.eye(4)
        T4 = np.eye(4)
        T5 = np.eye(4)
        T6 = np.eye(4)
        T7 = np.eye(4)

        cr = [i/2 for i in image_shape]
        scaling = self.get_scaling()
        angles = self.get_angles()
        translation = self.get_translation()

        T1[0, 3] = -cr[0]
        T1[1, 3] = -cr[1]
        T1[2, 3] = -cr[2]

        T2[0, 0] += scaling[0]
        T2[1, 1] += scaling[1]
        T2[2, 2] += scaling[2]

        T3[1, 1] = np.cos(angles[0])
        T3[1, 2] = -np.sin(angles[0])
        T3[2, 1] = np.sin(angles[0])
        T3[2, 2] = np.cos(angles[0])

        T4[0, 0] = np.cos(angles[1])
        T4[0, 2] = np.sin(angles[1])
        T4[2, 0] = -np.sin(angles[1])
        T4[2, 2] = np.cos(angles[1])

        T5[0, 0] = np.cos(angles[2])
        T5[0, 1] = -np.sin(angles[2])
        T5[1, 0] = np.sin(angles[2])
        T5[1, 1] = np.cos(angles[2])

        T6[0, 3] = cr[0]
        T6[1, 3] = cr[1]
        T6[2, 3] = cr[2]

        T7[0, 3] = translation[0]
        T7[1, 3] = translation[1]
        T7[2, 3] = translation[2]

        return T7 @ T6 @ T5 @ T4 @ T3 @ T2 @ T1

#############################################################################################
##################################  COMPOSITIONS  ###########################################
#############################################################################################
class Compose(object):
    def __init__(self, transform_parameters):

        self.transform_parameters = transform_parameters if transform_parameters is not None else []

    def _compute_data_shape(self, init_shape):

        self.img_shape = init_shape
        if not self.transform_parameters:
            return init_shape

        final_shape = init_shape
        if isinstance(init_shape, list):
            n_shape = len(init_shape)
        else:
            n_shape = 1

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                final_shape = [t.crop_shape] * n_shape

            elif isinstance(t, PadParams):
                if t.psize is None:
                    final_shape = init_shape
                    #     psize = max([max([di.size for di in d]) for d in self.data])
                    #     t.psize = (1 << (psize[0] - 1).bit_length(), 1 << (psize[1] - 1).bit_length())
                else:
                    final_shape = [t.psize] * n_shape

            elif isinstance(t, ResizeParams):
                final_shape = [t.resize_shape] * n_shape

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        if isinstance(init_shape, list):
            return final_shape
        else:
            return final_shape[0]

    def __call__(self, img):

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf(img)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape)(i) for i in img]

            elif isinstance(t, ResizeParams):
                img = [Resize(t)(i) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')
        return img

    def inverse(self, img, img_shape=None):

        if img_shape is None:
            if self.img_shape is None:
                raise ValueError("You need to provide the initial image shape or call the forward transform function"
                                 "before calling the inverse")
            else:
                img_shape = self.img_shape

        for t in self.transform_parameters:
            if isinstance(t, CropParams):
                tf = RandomCropManyImages(t)
                img = tf.inverse(img, img_shape)

            elif isinstance(t, PadParams):
                img = [Padding(t, i.shape).inverse(i, img_shape) for i in img]

            else:
                raise ValueError(
                    str(type(t)) + 'is not a valid type for transformation. Please, specify a valid one')

        return img

class Compose_DA(object):
    def __init__(self, data_augmentation_parameters):
        self.data_augmentation_parameters = data_augmentation_parameters if data_augmentation_parameters is not None else []

    def __call__(self, img, mask_flag = None, **kwargs):
        '''
        Mask flag is used to indicate which elements of the list are not used in intensity-based transformations, and
        only in deformation-based transformations.
        '''

        islist = True
        if not isinstance(img, list):
            img = [img]
            islist = False

        if mask_flag is None:
            mask_flag = [False] * len(img)


        for da in self.data_augmentation_parameters:

            if isinstance(da, NonLinearParams):
                tf = NonLinearDifferomorphismManyImages(da)
                img = tf(img, mask_flag)

            elif isinstance(da, RotationParams):
                tf = Rotation(da)
                img = tf(img, mask_flag)

            else:
                raise ValueError(str(type(da)) + 'is not a valid type for data augmentation. Please, specify a valid one')

        if not islist:
            img = img[0]

        return img

class Normalization(object):
    def __init__(self, normalization_list):
        self.normalization_list = normalization_list if normalization_list is not None else []
        if not isinstance(self.normalization_list, list):
            self.normalization_list = [self.normalization_list]

    def __call__(self, data, *args, **kwargs):
        for normalization in self.normalization_list:
            data = normalization(data)

        return data


#############################################################################################
###################################  FUNCTIONS  #############################################
#############################################################################################
class Resize(object):
    def __init__(self, parameters):
        self.params = parameters
        self.resize_shape = parameters.resize_shape


    def __call__(self, data):
        output = resize(data, dsize=self.resize_shape, interpolation=INTER_LINEAR)
        return output

class NormalNormalization(object):
    def __init__(self, mean = 0, std = 1, dim = None, inplace = False):

        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.dim = None

    def __call__(self, data, *args, **kwargs):
        if not self.inplace:
            data = copy.deepcopy(data)

        mean_d = np.mean(data, axis = self.dim)
        std_d = np.std(data, axis = self.dim)

        assert len(mean_d) == self.mean

        d_norm = (data - mean_d) / std_d
        out_data = (d_norm + self.mean) * self.std

        return out_data

class DeMean(object):
    def __init__(self):
        pass

    def __call__(self, data, *args, **kwargs):
        mean = np.mean(data)

        return data-mean

class ScaleNormalization(object):
    def __init__(self, scale=1.0, dtype='float', range=None, quantile=False, contrast=[0.99, 0.01]):

        self.scale = scale
        self.range = range
        self.quantile = quantile
        self.dtype = dtype
        self.contrast = contrast

    def get_mask_value(self, data):
        if self.range is not None:
            return self.range[0]
        else:
            return np.min(data)*self.scale

    def __call__(self, data, mask=None, *args, **kwargs):
        if mask is None:
            mask = np.ones_like(data, dtype='bool')
        else:
            mask = mask > 0

        if self.range is not None:
            if self.quantile:
                dmax = np.quantile(data[mask], self.contrast[0])
                dmin = np.quantile(data[mask], self.contrast[1])
            else:
                dmax = np.max(data[mask])
                dmin = np.min(data[mask])

            data = data.astype(self.dtype)
            data = (data - dmin) / (dmax-dmin) * (self.range[1] - self.range[0]) + self.range[0]
            data = np.clip(data, self.range[0], self.range[1])

        else:
            data[mask] = data[mask] * self.scale

        return data

class Padding(object):
    def __init__(self, parameters, isize, dim=2):


        if len(isize) > dim+1:
            raise ValueError("Please, specify a valid dimension and size")

        osize = parameters.psize
        assert len(osize) == dim

        pfill = parameters.pfill
        pmode = parameters.pmode

        psize = []
        for i, o in zip(isize, osize):
            if o - i > 0:
                pfloor = int(np.floor((o - i) / 2))
                pceil = pfloor if np.mod(o - i, 2) == 0 else pfloor + 1
            else:
                pfloor = 0
                pceil = 0

            psize.append((pfloor, pceil))

        pad_tuple = psize

        self.padding = pad_tuple
        self.fill = pfill
        self.padding_mode = pmode
        self.dim = dim
        self.osize = osize

    def __call__(self, data):

        if len(data.shape) == self.dim+1:
            nchannels = data.shape[-1]
            output_data = np.zeros(self.osize + (nchannels, ))
            for idim in range(nchannels):
                output_data[..., idim] = np.pad(data[..., idim], pad_width=self.padding, mode=self.padding_mode,
                                                constant_values=self.fill)
            return output_data
        else:
            return np.pad(data, pad_width=self.padding, mode=self.padding_mode, constant_values=self.fill)

class RandomCropManyImages(object):
    """Crop the given numpy array at a random location.
    Images are cropped at from the center as follows:


    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (d1, d2, ... , dN), a square crop (size, size, ..., size) is
            made.

        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding.
    """

    def __init__(self, parameters, pad_if_needed=True, fill=0, padding_mode='constant'):

        self.parameters = parameters
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params(self, data_shape, output_shape):

        if all([a==b for a,b in zip(data_shape, output_shape)]):
            return [0]*len(data_shape), data_shape

        if self.parameters.init_coordinates is None:
            init_coordinates = []
            for a, b in zip(data_shape, output_shape):
                init_coordinates.append(int((a-b)/2))
        else:
            init_coordinates = self.parameters.init_coordinates

        return init_coordinates, output_shape

    def __call__(self, data_list):
        """
        Args:
            data_list : list of numpy arrays. Each numpy array has the following size: (num_channels, d1, ..., dN)

        Returns:
            output_list: list of cropped numpy arrays.
        """
        size = self.parameters.crop_shape
        n_dims = len(size)
        padded_data_list = []
        for i in range(len(data_list)):
            data = data_list[i]
            # pad the width if needed
            pad_width = []
            for it_dim in range(n_dims):
                if self.pad_if_needed and data.shape[it_dim] < size[it_dim]:
                    pad_width.append((size[it_dim] - data.shape[it_dim],0))
                else:
                    pad_width.append((0,0))

            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            padded_data_list.append(data)


        init_coord, output_shape = self.get_params(padded_data_list[0].shape, size)

        self.init_coord = init_coord
        self.output_shape = output_shape

        output = []
        for i in range(len(padded_data_list)):
            padded_data = padded_data_list[i]
            for it_dim in range(n_dims):
                idx = (slice(None),) * (it_dim) + \
                      (slice(init_coord[it_dim], init_coord[it_dim] + output_shape[it_dim], 1), )
                padded_data = padded_data[idx]
            output.append(padded_data)

        return output

    def inverse(self, data_list, data_shape):
        size = self.parameters.crop_shape
        n_dims = len(size)

        cropped_data_list = []
        for data, dshape in zip(data_list, data_shape):
            cropped_data = data
            for it_dim in range(n_dims):
                init_coord = size[it_dim] - dshape[it_dim]
                if init_coord < 0:
                    init_coord = 0

                idx = (slice(None),) * (it_dim) + (slice(init_coord, init_coord + dshape[it_dim], 1),)
                cropped_data = cropped_data[idx]
            cropped_data_list.append(cropped_data)

        init_coord, _ = self.get_params(data_shape[0], size)

        output = []
        for data, dshape in zip(cropped_data_list, data_shape):
            pad_width = []
            for it_dim in range(n_dims):
                if size[it_dim] < dshape[it_dim]:
                    pad_width.append((int(init_coord[it_dim]), int(dshape[it_dim] - size[it_dim] - init_coord[it_dim])))
                else:
                    pad_width.append((0, 0))

            data = np.pad(data, pad_width=pad_width, mode=self.padding_mode, constant_values=self.fill)
            output.append(data)

        return output

class NonLinearDeformationManyImages(object):

    def __init__(self, params, output_flow=False, reverse_field=False):
        self.params = params
        self.output_flow = output_flow
        self.reverse_field = reverse_field

    def _get_lowres_strength(self):

        size = 1
        if self.params.distribution == 'normal':
            mean, std = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean

        elif self.params.distribution == 'uniform':
            high, low = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.rand(size) * (high - low) + low

        elif self.params.distribution == 'lognormal':
            mean, std = self.params.lowres_strength[1], self.params.lowres_strength[0]
            lowres_strength = np.random.randn(size) * std + mean
            lowres_strength = np.exp(lowres_strength)

        elif self.params.distribution is None:
            lowres_strength = [self.params.lowres_strength] * size

        else:
            raise ValueError("[src/utils/transformations: NonLinearDeformation]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        field_lowres_x = lowres_strength * np.random.randn(self.params.lowres_size[0],
                                                           self.params.lowres_size[1])  # generate random noise.

        field_lowres_y = lowres_strength * np.random.randn(self.params.lowres_size[0],
                                                           self.params.lowres_size[1])  # generate random noise.

        return field_lowres_x, field_lowres_y

    def __call__(self, data, mask_flag, XX, YY, flow_x, flow_y, *args, **kwargs):

        x, y = XX + flow_x, YY + flow_y

        data_tf = []
        for it_image, (image, m) in enumerate(zip(data, mask_flag)):
            if m:
                data_tf.append(griddata((YY.flatten(), XX.flatten()), image.flatten(), (y, x), method='nearest'))
            else:
                # data_tf.append(np.double(bilinear_interpolate(image, y, x)>0.5))
                data_tf.append(bilinear_interpolate(image, x, y))
        return data_tf

class NonLinearDifferomorphismManyImages(NonLinearDeformationManyImages):


    def get_diffeomorphism(self, field_lowres_x, field_lowres_y, image_shape, reverse=False):

        field_highres_x = resize(field_lowres_x, dsize=(image_shape[1], image_shape[0]), interpolation=INTER_LINEAR)
        field_highres_y = resize(field_lowres_y, dsize=(image_shape[1], image_shape[0]), interpolation=INTER_LINEAR)

        # integrate
        YY, XX = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]), indexing='ij')

        flow_x = field_highres_x / (2 ** self.params.nstep)
        flow_y = field_highres_y / (2 ** self.params.nstep)

        if reverse:
            flow_x = -flow_x
            flow_y = -flow_y

        for it_step in range(self.params.nstep):
            x = XX + flow_x
            y = YY + flow_y
            incx = bilinear_interpolate(flow_x, x, y)
            incy = bilinear_interpolate(flow_y, x, y)
            flow_x = flow_x + incx.reshape(image_shape)
            flow_y = flow_y + incy.reshape(image_shape)

        return XX, YY, flow_x, flow_y

    def __call__(self, data, mask_flag, *args, **kwargs):

        image_shape = data[0].shape
        field_lowres_x, field_lowres_y = self._get_lowres_strength()

        XX, YY, flow_x, flow_y = self.get_diffeomorphism(field_lowres_x, field_lowres_y, image_shape)
        data_tf = super().__call__(data, mask_flag, XX, YY, flow_x, flow_y)

        if self.output_flow:
            if self.reverse_field:
                XX, YY, flow_x, flow_y = self.get_diffeomorphism(field_lowres_x, field_lowres_y, image_shape, reverse=True)

            return data_tf, np.stack([flow_x, flow_y], axis=0)
        else:
            return data_tf

class Rotation(object):
    def __init__(self, params, dense_field=False, reverse=True):
        '''

        :param params: instance of RotationParams
        :param dense_field: affine matrix transformation to dense field.
        :param reverse: it outputs the reverse transformation
        '''
        self.params = params
        self.dense_field = dense_field
        self.reverse = reverse

    def _get_angle(self, size):

        if self.params.distribution == 'normal':
            mean, std = self.params.value_range[1], self.params.value_range[0]
            angle = np.random.randn(size) * std + mean

        elif self.params.distribution == 'uniform':
            high, low = self.params.value_range[1], self.params.value_range[0]
            angle = np.random.rand(size) * (high - low) + low

        elif self.params.distribution is None:
            angle = self.params.lowres_strength * np.ones(size)

        else:
            raise ValueError("[src/utils/transformations: MultiplicativeNoise]. Please, specify a valid distribution "
                             "for the low-res nonlinear distribution")

        return angle

    def _get_affine_matrix(self, angle):
        angle_rad = angle * np.pi / 180
        affine_matrix = np.array([
            [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
            [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
            [0, 0, 1]
        ])
        return affine_matrix

    def _get_dense_field(self, affine_matrix, volshape):

        ndims = len(volshape)

        vectors = [np.arange(0, s) for s in volshape]
        mesh = np.meshgrid(*vectors, indexing=('ij')) #grid of vectors
        mesh = [f.astype('float32') for f in mesh]
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(ndims)] #shift center

        # mesh = volshape_to_meshgrid(volshape, indexing=indexing)
        # mesh = [tf.cast(f, 'float32') for f in mesh]

        # add an all-ones entry and transform into a large matrix
        flat_mesh = [np.reshape(f, (-1,)) for f in mesh]
        flat_mesh.append(np.ones(flat_mesh[0].shape, dtype='float32'))
        mesh_matrix = np.transpose(np.stack(flat_mesh, axis=1))  # ndims+1 x nb_voxels

        # compute locations
        loc_matrix = np.matmul(affine_matrix, mesh_matrix)  # ndims+1 x nb_voxels
        loc = np.reshape(loc_matrix[:ndims, :], [ndims] + list(volshape))  # ndims x *volshape

        # get shifts and return
        shift = loc - np.stack(mesh, axis=0)
        shift = np.stack([shift[1], shift[0]], axis=0)
        return shift.astype('float32')

    def __call__(self, data, mask_flag, *args, **kwargs):
        '''
        :param data: 2D data
        :param mask_flag: True = nearest interpolation, False = bilinear interpolation
        :return:
        '''

        angle = self._get_angle(1)
        affine_matrix = self._get_affine_matrix(angle)
        flow = self._get_dense_field(affine_matrix, data[0].shape)
        data_tf = []
        for image, m in zip(data, mask_flag):
            o = 'nearest' if m else 'bilinear'
            data_tf.append(deform2D(image,flow,o))

        if self.dense_field:
            if self.reverse:
                affine_matrix = self._get_affine_matrix(-angle)
                flow = self._get_dense_field(affine_matrix, data[0].shape)
            return data_tf, flow
        else:
            if self.reverse:
                affine_matrix = self._get_affine_matrix(-angle)
            return data_tf, affine_matrix
