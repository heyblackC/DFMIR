import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# ==============================================================================
# Define a custom colormap for visualiza Jacobian
# ==============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# ==============================================================================
# plot an array of images for comparison
# ==============================================================================
def show_sample_slices(sample_list, name_list, path, Jac=False, cmap='gray', attentionlist=None):
    #num = len(sample_list)
    #fig, ax = plt.subplots(1, num)
    #plt.figure(1,figsize=(512,512))
    #plt.figure(figsize=(256, 256), dpi=200)
    plt.axis('off')
    plt.imshow(sample_list[0], cmap, norm=MidpointNormalize(midpoint=1))
    #plt.colorbar()
    plt.savefig(path)
    plt.close("all")

    # for i in range(num):
    #     if Jac:
    #         ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=1))
    #     else:
    #         ax[i].imshow(sample_list[i], cmap)
    #     #ax[i].set_title(name_list[i])
    #     ax[i].axis('off')
    #     if attentionlist:
    #         ax[i].add_artist(attentionlist[i])
    # plt.subplots_adjust(wspace=0)
    # plt.savefig(path)
    # plt.close("all")