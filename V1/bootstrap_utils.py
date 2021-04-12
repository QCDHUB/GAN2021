import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc, rcParams
import numpy as np


def define_grid_errband( nrows, ncols, hspace=0.18, wspace=0.14, figsize=(14,10) ):

    tick_labelsize = 10
    rc( 'xtick', labelsize=tick_labelsize )
    rc( 'ytick', labelsize=tick_labelsize )

    f = plt.figure(figsize=figsize)

    outer_gs = gridspec.GridSpec(nrows, ncols, top=0.935, bottom=0.075, left=0.065, right=0.965,
                           hspace=hspace, wspace=wspace)

    axs = []
    ax_histxs = []
    for i in range( nrows * ncols ):
        innner_gs = outer_gs[i].subgridspec(2, 1,  height_ratios=(4, 2), wspace=0.02, hspace=0.02)
        ax = f.add_subplot( innner_gs[0, 0] )
        ax_histx = f.add_subplot( innner_gs[1, 0], sharex=ax )

        axs.append( f.add_subplot( ax ) )
        ax_histxs.append( f.add_subplot( ax_histx ) )

    for ax in axs:
        ax.tick_params( labelbottom=False )

    for i, ax in enumerate( ax_histxs ):
        ax.tick_params( left=False, labelleft=False )
        ax.tick_params( right=True, labelright=True )
        ax.set_ylim(0.80,1.20)

    return f, (axs), (ax_histxs), outer_gs


def get_theta( clasfilter ):
    px = clasfilter[:, 3]
    py = clasfilter[:, 6]
    pz = clasfilter[:, 9]
    pt = np.sqrt( px * px + py * py )
    theta = np.arctan2(pt, pz)
    return theta


def get_ratio_data( ax, b, x1, dx1, x2, dx2 ):
    r = np.where(x1 != 0, x1/x2, x1/x2)
    dr = dx1/x2
    return r, dr
