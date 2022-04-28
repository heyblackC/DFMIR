import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class _Loss(nn.Module):
    def __init__(self, name = None):
        super().__init__()
        self.name = name

class IDLoss(_Loss):
    def forward(self, x, *args, **kwargs):
        return x

class SSIM(_Loss):
    def __init__(self, name = None, reduction='mean', *args, **kwargs):
        if name is None:
            name='SSIM'
        super().__init__(name=name)
        self.reduction = reduction

    @NotImplementedError
    def _ssim_loss(self, prediction, target, reduction='mean'):
        pass

    def forward(self, prediction, target, mask=None, weight=None, *args, **kwargs):
        ndims = len(prediction.shape)
        if mask is None and weight is None:
            return self._ssim_loss(prediction, target, reduction=self.reduction)
        else:
            res = self._ssim_loss(prediction, target, reduction='none')
            if mask is not None:
                res = res * mask
            if weight is not None:
                res = res * weight

        if self.reduction == 'mean':
            return 1/torch.sum(mask)*torch.sum(res)
            # norm_factor = torch.sum(mask, dim=[it for it in range(1,ndims)], keepdim=True)
            # wk = 1 / norm_factor * torch.sum(mask, dim=[it for it in range(2,ndims)])
            # res = 1 / norm_factor * torch.sum(res, dim=[it for it in range(2,ndims)])
            # res = torch.sum(wk*res, dim=1)
            # res = torch.mean(res)

        elif self.reduction == 'sum':
            return torch.sum(res)
        else:
            return res

        return res

class L2_Loss(SSIM):
    def _ssim_loss(self, prediction, target, reduction='mean'):
        return F.mse_loss(prediction, target, reduction=reduction)

class L1_Loss(SSIM):
    def _ssim_loss(self, prediction, target, reduction='mean'):
        return F.l1_loss(prediction, target, reduction=reduction)

class TukeyBiweight(SSIM):
    def __init__(self, c = 0.8, name = None, reduction='mean', *args, **kwargs):
        if name is None:
            name = 'tukey_biweight_loss'
        super().__init__(name=name)
        self.c = c

    def _ssim_loss(self, prediction, target, reduction='mean'):
        error = prediction - target

        max_loss = self.c**2/6
        loss = self.c**2/6 *(1 - (1 - (error/self.c)**2)**3)

        loss = torch.clamp(loss, 0, max_loss)

        return loss

class Grad_Loss(_Loss):
    def __init__(self, dim=2, penalty='l2', name=None, loss_mult=None, *args, **kwargs):
        if name is None:
            name='gradient'
        super().__init__(name=name, *args, **kwargs)

        assert dim in [2, 3]
        self.dim = dim
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _grad3d(self, prediction):
        dy = torch.abs(prediction[:, :, 1:] - prediction[:, :, :-1])
        dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        dz = torch.abs(prediction[..., 1:] - prediction[..., :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

        return d / 3.0

    def _grad2d(self, prediction):
        dy = torch.abs(prediction[:, :, 1:] - prediction[:, :, :-1])
        dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)

        return d / 2.0

    def forward(self, prediction, *args, **kwargs):
        if 'mask' in kwargs:
            prediction = prediction * kwargs['mask']

        if self.dim == 2:
            loss = self._grad2d(prediction)
        else:
            loss = self._grad3d(prediction)

        if self.loss_mult is not None:
            loss *= self.loss_mult

        return loss

class NCC_Loss(_Loss):

    def __init__(self, device, kernel_var=None, name=None, kernel_type='mean', eps=1e-5, *args, **kwargs):
        if name is None:
            name = 'ncc'
        super().__init__(name=name)
        self.device = device
        self.kernel_var = kernel_var
        self.kernel_type = kernel_type
        self.eps = eps

        assert kernel_type in ['mean', 'gaussian', 'linear']

    def _get_kernel(self, kernel_type, kernel_sigma):

        if kernel_type == 'mean':
            kernel = torch.ones([1, 1, *kernel_sigma]).to(self.device)

        elif kernel_type == 'linear':
            raise NotImplementedError("Linear kernel for NCC still not implemented")

        elif kernel_type == 'gaussian':
            kernel_size = kernel_sigma[0] * 3
            kernel_size += np.mod(kernel_size + 1, 2)

            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1) / 2.
            variance = kernel_sigma[0] ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            # 2.506628274631 = sqrt(2 * pi)

            kernel = (1. / (2.506628274631 * kernel_sigma[0])) * \
                     torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

            # Make sure sum of values in gaussian kernel equals 1.
            # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.to(self.device)

        return kernel

    def _compute_local_sums(self, I, J, filt, stride, padding):

        ndims = len(list(I.size())) - 2

        I2 = I * I
        J2 = J * J
        IJ = I * J

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I_sum = conv_fn(I, filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

        win_size = torch.sum(filt)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def ncc(self, prediction, target):
        """
            calculate the normalize cross correlation between I and J
            assumes I, J are sized [batch_size, nb_feats, *vol_shape]
            """

        ndims = len(list(prediction.size())) - 2

        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if self.kernel_var is None:
            if self.kernel_type == 'gaussian':
                kernel_var = [3] * ndims  # sigma=3, radius = 9
            else:
                kernel_var = [9] * ndims  # sigma=radius=9 for mean and linear filter

        else:
            kernel_var = self.kernel_var

        sum_filt = self._get_kernel(self.kernel_type, kernel_var)
        radius = sum_filt.shape[-1]
        pad_no = int(np.floor(radius / 2))

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Eugenio: bug fixed where cross was not squared when computing cc
        I_var, J_var, cross = self._compute_local_sums(prediction, target, sum_filt, stride, padding)
        cc = cross * cross / (I_var * J_var + self.eps)

        return cc

    def forward(self, prediction, target, mask=None, *args, **kwargs):

        # if mask is not None:
        #     prediction = prediction * mask
        #     target = target * mask

        cc = self.ncc(prediction, target)
        if mask is None:
            return -1.0 * torch.sqrt(torch.mean(cc))
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return -1.0 * torch.sqrt(norm_factor * torch.sum(cc * mask))

class NMI_Loss(_Loss):

    # Eugenio: what is vol_size needed for?
    def __init__(self, bin_centers, # vol_size,
                 device='cpu', sigma_ratio=0.5, max_clip=1,
                 crop_background=False, patch_size=1, name='nmi'):
        """
        Adapted from:

        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        super().__init__(name=name)
        # self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers, requires_grad=True, device=device, dtype=torch.float32)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * np.square(self.sigma))

    def global_mi(self, y_true, y_pred, padding_size=15, mask=None):

        ndims = len(y_true.shape) - 2
        conv_fn = getattr(F, 'conv%dd' % ndims)

        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            if mask is None:
                if ndims == 2:
                    filt = torch.ones((1, 1, padding_size, padding_size))
                    stride = [1, 1, 1, 1]
                elif ndims == 3:
                    filt = torch.ones((1, 1, padding_size, padding_size, padding_size))
                    stride = [1, 1, 1, 1, 1]
                else:
                    raise ValueError('Not valid image size')
                smooth = conv_fn(y_true, filt, stride=stride, padding=int((padding_size-1)/2))
                mask = smooth > thresh
            else:
                mask = mask > thresh

            y_pred = torch.masked_select(y_pred, mask).view((1,1,-1))
            y_true = torch.masked_select(y_true, mask).view((1,1,-1))

        else:
            # reshape: flatten images into shape (batch_size, 1, heightxwidthxdepthxchan)
            y_true = y_true.view((1,1,-1))
            y_pred = y_pred.view((1,1,-1))

        nb_voxels = y_pred.shape[2]

        # reshape bin centers to be (1, 1, B)
        o = [1, self.vol_bin_centers.numel(), 1]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = torch.exp(- self.preterm * (y_true  - vbc)**2)
        I_a_norm = I_a / torch.sum(I_a, dim=1, keepdim=True)

        I_b = torch.exp(- self.preterm * (y_pred  - vbc)**2)
        I_b_norm = I_b / torch.sum(I_b, dim=1, keepdim=True)

        # compute probabilities
        I_a_norm_permute = torch.transpose(I_a_norm, dim0=1, dim1=2)
        pab = torch.bmm(I_b_norm, I_a_norm_permute)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels

        pa = torch.mean(I_a_norm, dim=-1, keepdim=True)
        pb = torch.mean(I_b_norm, dim=-1, keepdim=True)
        papb = torch.bmm(pb, torch.transpose(pa, dim0=1, dim1=2)) + 1e-5

        return torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-5), dim=1), dim=1)

    def __call__(self, y_true, y_pred, **kwargs):
        y_pred = torch.clamp(y_pred, 0, self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred, **kwargs)

class Dice_Loss(_Loss):
    def __init__(self, name=None, *args, **kwargs):

        if name is None:
            name='dice'
        super().__init__(name=name)



    def forward(self, target, prediction, classes_compute=None, eps = 0.0000001):
        """Dice loss.
        Compute the dice similarity loss (approximation of the DSC). The foreground

        Parameters
        ----------
        prediction : torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing the post-softmax
            values

        target : torch variable of ssize (batch_size, num_classes, d1, d2, ..., dN) representing a 1-hot encoding of the
            target values

        Returns
        -------
        dice_total :

        """
        # smooth = eps #1.
        # pflat = prediction.view(-1)
        # tflat = target.view(-1)
        # intersection = (pflat * tflat).sum()
        #
        # return 1 - ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))



        prediction = torch.clip(prediction / torch.sum(prediction, dim=1, keepdims=True), 0, 1)
        target = torch.clip(target / torch.sum(target, dim=1, keepdims=True), 0, 1)
        if classes_compute is not None:
            prediction = prediction[:, classes_compute]
            target = target[:, classes_compute]

        top = torch.sum(2 * prediction * target, dim=list(range(1, len(prediction.shape))))
        bottom = prediction**2 + target**2 + eps
        bottom = torch.sum(bottom, dim=list(range(1, len(prediction.shape))))

        last_tensor = top / bottom

        return torch.mean(1 - last_tensor)


class CrossEntropy(_Loss):
    def __init__(self, name = None, class_weights=None, ignore_index=-100, reduction='none', *args, **kwargs):
        if name is None:
            name = 'cross_entropy'
        super().__init__(name=name)
        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction=reduction)

    def forward(self, prediction, target, **kwargs):
        """
        Compute the cross-entropy loss between predictions and targets
        Arguments:
            - prediction: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing pre-softmax probabilities for each class
            - target: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing  the ground-truth values
        """

        if target.shape[1] == prediction.shape[1]:
            _, labels = target.max(dim=1)
        else:
            labels = target
        labels = labels.long()
        return self.loss(prediction, labels)

class NLL(_Loss):
    def __init__(self, name = None, reduction='mean', *args, **kwargs):
        if name is None:
            name = 'negative_loglikelihood'
        super().__init__(name=name)
        self.reduction = reduction

    def forward(self, prediction, target, **kwargs):
        """
        Compute the cross-entropy loss between predictions and targets
        Arguments:
            - prediction: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing log probabilities for each class
            - target: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing  the ground-truth values
        """

        loss = target * torch.log(prediction+1e-5)
        loss = torch.sum(loss, dim=1, keepdim=True)

        # slices([loss[0,0], kwargs['mask'][0,0], prediction[0,6], target[0,6]])
        if 'mask' in kwargs:
            loss = kwargs['mask'] * loss

        if self.reduction == 'none':
            return -loss

        elif self.reduction == 'sum':
            return -torch.sum(loss)

        else:
            if 'mask' in kwargs:
                norm_factor = torch.sum(torch.sum(kwargs['mask'], dim=-1), dim=-1)
                loss = 1/norm_factor * torch.sum(torch.sum(loss, dim=-1), dim=-1)

            return -torch.mean(loss)

class GradientPenalty(_Loss):

    def __init__(self, name = None, device='cpu', alpha=0.1):
        if name is None:
            name = 'gradient_penalty'
        super().__init__(name=name)
        self.device = device
        self.alpha = alpha

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = self.alpha * torch.ones((real_samples.size(0), 1, 1, 1)).to(self.device)  # torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size()).to(self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, D, real_samples, fake_samples):
        return self._compute_gradient_penalty(D, real_samples, fake_samples)

class LSGAN(_Loss):
    def __init__(self, name=None, device='cpu'):
        if name is None:
            name = 'LSGAN_loss'
        super().__init__(name=name)
        self.loss = nn.MSELoss(reduction='none')
        self.register_buffer('real_label', torch.tensor(1., device=device))
        self.register_buffer('fake_label', torch.tensor(0., device=device))

    def forward(self, prediction, target_is_real, **kwargs):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

class WGAN(_Loss):
    def __init__(self, name=None, **kwargs):
        if name is None:
            name = 'wasserstein_loss'
        super().__init__(name=name)

    def forward(self, prediction, target_is_real, **kwargs):

        if target_is_real:
            return -torch.mean(prediction)
        else:
            return torch.mean(prediction)



class PatchNCELoss(_Loss):
    def __init__(self, batch_size, nce_T, name=None):
        if name is None:
            name = 'NCE_Loss'
        super().__init__(name=name)
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]# number of patches
        dim = feat_q.shape[1]# number of channels
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)# num_patches x 1

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        # fig, _ = slices(out, cmaps=['gray'], show=False, do_colorbars=True)#, imshow_args=[{'vmin': -10, 'vmax':5}]
        # fig.savefig('out.png')
        # print(out.shape)
        # print(out)

        # True label is 0 because the first column is the l_pos. In l_neg we also have l_pos in the diagonal but
        # with a very small value that will not influence the softmax.
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


DICT_LOSSES = {
    'L1': L1_Loss,
    'L2': L2_Loss,
    'TukeyBiweight': TukeyBiweight,

    'PatchNCE': PatchNCELoss,

    'Grad': Grad_Loss,
    'NCC': NCC_Loss,
    'NMI': NMI_Loss,

    'CrossEntropy': CrossEntropy,
    'NLL': NLL,
    'Dice': Dice_Loss,

    'WGAN': WGAN,
    'LSGAN': LSGAN,
    'GradPenGAN': GradientPenalty,

}
