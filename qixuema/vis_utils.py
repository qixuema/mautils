import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def polylines_to_png(
    polylines, 
    filename='multi_polyline.png',
    dpi=72,
    figsize=(5, 5),
    linewidth=2,
    n_ticks=5,
    markersize=3,
):
    """
    render and save the multiple 3D polylines as a PNG file
    
    Args:
    - polylines: List of (N, 3) numpy array or list of points, each element is a polyline
    - filename: path to save the output PNG file
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # plot each polyline, using the default color cycle
    for pts in polylines:
        pts_arr = np.asarray(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                linewidth=linewidth, markersize=markersize)


    # ==== 设置等比例坐标轴 ====
    for i, axis in enumerate(['x', 'y', 'z']):
        getattr(ax, f'set_{axis}lim')(-1.0, 1.0)


    ax.set_box_aspect((1, 1, 1))
    # ax.set_proj_type('ortho')

    # --------------------------
    # control the number of major ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.zaxis.set_major_locator(MaxNLocator(n_ticks))
    # also adjust the font size
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for label in axis.get_ticklabels():
            label.set_fontsize(8)
    # --------------------------
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
