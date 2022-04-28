import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision import models
from scipy.ndimage.morphology import distance_transform_edt as edt

class DeepSim(nn.Module):
    """
    Deep similarity metric
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        # extract features
        feats0 = y_true
        feats1 = y_pred
        losses = []
        for feat0, feat1 in zip(feats0, feats1):
            # calculate cosine similarity
            prod_ab = torch.sum(feat0 * feat1, dim=1)
            norm_a = torch.sum(feat0 ** 2, dim=1).clamp(self.eps) ** 0.5
            norm_b = torch.sum(feat1 ** 2, dim=1).clamp(self.eps) ** 0.5
            cos_sim = prod_ab / (norm_a * norm_b)
            losses.append(torch.mean(cos_sim))

        # mean and invert for minimization
        return -torch.stack(losses).mean() + 1

class DeepSimRaw(nn.Module):
    """
    Deep similarity metric
    """

    def __init__(self, seg_model, eps=1e-6):
        super().__init__()
        self.seg_model = seg_model
        self.eps = eps

        # fix params
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        # set to eval (deactivate dropout)
        self.seg_model.eval()

        # extract features
        feats0 = self.seg_model.extract_features(y_true)
        feats1 = self.seg_model.extract_features(y_pred)
        losses = []
        for feat0, feat1 in zip(feats0, feats1):
            # calculate cosine similarity
            prod_ab = torch.sum(feat0 * feat1, dim=1)
            norm_a = torch.sum(feat0 ** 2, dim=1).clamp(self.eps) ** 0.5
            norm_b = torch.sum(feat1 ** 2, dim=1).clamp(self.eps) ** 0.5
            cos_sim = prod_ab / (norm_a * norm_b)
            losses.append(torch.mean(cos_sim))

        # mean and invert for minimization
        return -torch.stack(losses).mean() + 1



class VGGFeatureExtractor(nn.Module):
    """
    pretrained VGG-net as a feature extractor
    """

    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.N_slices = 3
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # pad x to RGB input
        x = torch.cat([x, x, x], dim=1)
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3]

    def extract_features(self, x):
        return self(x)

class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)

def cross_correlation_loss(I, J, n):
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding=1 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)

def vox_morph_loss(y, ytrue, n=9):
    cc = cross_correlation_loss(y, ytrue, n)
    #print("CC Loss", cc, "Gradient Loss", sm)
    loss = -1.0 * cc
    return loss
