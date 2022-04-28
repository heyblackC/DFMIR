import numpy as np
from scipy.interpolate import interpn



def tanh2im(data_list, mask_list=None):
    output_list = []
    if mask_list is None:
        for data in data_list:
            mask = data == 0
            data[mask] = (data[mask] + 1) / 2
            output_list.append(data)
    else:
        for data, mask in zip(data_list, mask_list):
            mask = mask > 0.5
            data[mask] = (data[mask] + 1) / 2
            data[mask==0] = 0
            output_list.append(data)

    return output_list


def normalize_target_tensor(labels, class_labels=None, num_classes=None):

    if class_labels is None:
        if num_classes is None:
            raise ValueError('Need to specify class_labels or num_classes')
        else:
            class_labels = list(range(num_classes))

    for it_cl, cl in enumerate(class_labels):
        labels[labels == cl] = it_cl

    return labels

def one_hot_encoding(target, num_classes, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None:
        categories = list(range(num_classes))

    labels = np.zeros((num_classes,) + target.shape)
    for it_cl, c in enumerate(categories):
        idx_class = np.where(target == c)
        idx = (it_cl,) + idx_class
        labels[idx] = 1

    return labels.astype(int)

def get_affine_from_rotation_2d(angle):

    angle_rad = angle * np.pi / 180
    affine_matrix = np.array([
        [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
        [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
    ])
    return affine_matrix

def get_affine_from_rotation_3d(angle_list):

    angle_rad = angle_list[0] * np.pi / 180
    affine_matrix_0 = np.array([
        [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0, 0],
        [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    angle_rad = angle_list[1] * np.pi / 180
    affine_matrix_1 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
        [0, np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
        [0, 0, 0, 1],
    ])

    angle_rad = angle_list[2] * np.pi / 180
    affine_matrix_2 = np.array([
        [np.cos(angle_rad).item(), 0, np.sin(angle_rad).item(), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_rad).item(), 0, np.cos(angle_rad).item(), 0],
        [0, 0, 0, 1],
    ])

    affine_matrix = np.dot(affine_matrix_0, np.dot(affine_matrix_1, affine_matrix_2))
    return affine_matrix

def bilinear_interpolate(im, x, y):
    '''

    :param im: 2D image with size NxM
    :param x: coordinates in 'x' (M columns of a matrix)
    :param y: coordinates in 'y' (N rows of a matrix)
    :return:
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def bilinear_interpolate3d(im, x, y, z):
    '''

    :param im: 2D image with size NxM
    :param x: coordinates in 'x' (M columns of a matrix)
    :param y: coordinates in 'y' (N rows of a matrix)
    :param z: coordinates in 'z' (L)
:return:
    '''

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)
    z0 = np.clip(z0, 0, im.shape[2] - 1)
    z1 = np.clip(z1, 0, im.shape[2] - 1)

    Ia = im[y0, x0, z0]
    Ib = im[y1, x0, z0]
    Ic = im[y0, x1, z0]
    Id = im[y1, x1, z0]
    Ie = im[y0, x0, z1]
    If = im[y1, x0, z1]
    Ig = im[y0, x1, z1]
    Ih = im[y1, x1, z1]

    wa = (x1 - x) * (y1 - y) * (z1 - z)
    wb = (x1 - x) * (y - y0) * (z1 - z)
    wc = (x - x0) * (y1 - y) * (z1 - z)
    wd = (x - x0) * (y - y0) * (z1 - z)
    we = (x1 - x) * (y1 - y) * (z0 - z)
    wf = (x1 - x) * (y - y0) * (z0 - z)
    wg = (x - x0) * (y1 - y) * (z0 - z)
    wh = (x - x0) * (y - y0) * (z0 - z)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih

def deform2D(image, field, mode='bilinear'):
    '''

    :param image: 2D np.array (nrow, ncol)
    :param field: 3D np.array (2, nrow, ncol)
    :param mode: 'bilinear' or 'nearest'
    :return:
    '''

    dx = field[0]
    dy = field[1]
    output_shape = field.shape[1:]
    if len(image.shape) > 2: #RGB
        YY, XX = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')

        XXd = XX + dx
        YYd = YY + dy

        output = np.zeros(output_shape + (3,))
        if mode == 'bilinear':
            ok1 = YYd >= 0
            ok2 = XXd >= 0
            ok3 = YYd <= image.shape[0] - 1
            ok4 = XXd <= image.shape[1] - 1
            ok = ok1 & ok2 & ok3 & ok4

            points = (np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
            xi = np.concatenate((YYd[ok].reshape(-1, 1), XXd[ok].reshape(-1, 1)), axis=1)
            for it_ch in range(3):
                output_flat = interpn(points, image[..., it_ch], xi=xi, method='linear')
                output_ch = np.zeros(output_shape)
                output_ch[ok] = output_flat
                output[..., it_ch] = output_ch

        # elif mode == 'nearest':
        #     ok1 = YYd >= 0
        #     ok2 = XXd >= 0
        #     ok3 = YYd <= image.shape[0] - 1
        #     ok4 = XXd <= image.shape[1] - 1
        #     ok = ok1 & ok2 & ok3 & ok4
        #
        #     points = (np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
        #     xi = np.concatenate((YYd[ok].reshape(-1, 1), XXd[ok].reshape(-1, 1)), axis=1)
        #     output_flat = interpn(points, image, xi=xi, method='nearest')
        #
        #     output = np.zeros(output_shape)
        #     output[ok] = output_flat

        else:
            raise ValueError('Interpolation mode not available')
        # output = np.zeros(output_shape + (3,))
        # YY, XX = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')
        # XXd = XX + dx
        # YYd = YY + dy
        # for it_c in range(3):
        #     if mode == 'bilinear':
        #         output[:,:,it_c] = bilinear_interpolate(image[:,:,it_c], XXd, YYd)
        #     elif mode == 'nearest':
        #         output[:,:,it_c] = griddata((YY.flatten(), XX.flatten()), image[:,:,it_c].flatten(), (YYd, XXd), method='nearest')
        #     else:
        #         raise ValueError('Interpolation mode not available')
    else:
        YY, XX = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')

        XXd = XX+dx
        YYd = YY+dy

        if mode == 'bilinear':
            # output = bilinear_interpolate(image, XXd, YYd)
            ok1 = YYd >= 0
            ok2 = XXd >= 0
            ok3 = YYd <= image.shape[0] - 1
            ok4 = XXd <= image.shape[1] - 1
            ok = ok1 & ok2 & ok3 & ok4

            points = (np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
            xi = np.concatenate((YYd[ok].reshape(-1, 1), XXd[ok].reshape(-1, 1)), axis=1)
            output_flat = interpn(points, image, xi=xi, method='linear')

            output = np.zeros(output_shape)
            output[ok] = output_flat

            # output = griddata((YY.flatten(), XX.flatten()), image.flatten(), (YYd, XXd), method='nearest')
        elif mode == 'nearest':
            ok1 = YYd >= 0
            ok2 = XXd >= 0
            ok3 = YYd <= image.shape[0] - 1
            ok4 = XXd <= image.shape[1] - 1
            ok = ok1 & ok2 & ok3 & ok4

            points = (np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))
            xi = np.concatenate((YYd[ok].reshape(-1, 1), XXd[ok].reshape(-1, 1)), axis=1)
            output_flat = interpn(points, image, xi=xi, method='nearest')

            output = np.zeros(output_shape)
            output[ok] = output_flat

            # output = griddata((YY.flatten(), XX.flatten()), image.flatten(), (YYd, XXd), method='nearest')
        else:
            raise ValueError('Interpolation mode not available')


    return output

def affine_to_dense(affine_matrix, volshape):

    ndims = len(volshape)

    vectors = [np.arange(0, s) for s in volshape]
    YY, XX = np.meshgrid(*vectors, indexing=('ij')) #grid of vectors
    mesh = [XX, YY]
    mesh = [f.astype('float32') for f in mesh]
    mesh = [mesh[f] - (volshape[ndims - f - 1] - 1) / 2 for f in range(ndims)] #shift center

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
    return shift.astype('float32')
