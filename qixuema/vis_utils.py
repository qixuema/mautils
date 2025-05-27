import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import open3d as o3d

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

def vis_lineset(data):
    if isinstance(data, dict):
        lineset = data
    else:
        shape = data.shape
        # 
        if len(shape) == 3 and shape[1] == 2 and shape[2] == 3:
            print("数组的形状是 nx2x3")
        else:
            print("数组的形状不是 nx2x3")
            return
        
        vertices = data.reshape(-1, 3)
        lines = np.arange(len(vertices)).reshape(-1, 2)
        lineset = {'vertices': vertices, 'lines': lines}

    lineset['vertices'] = lineset['vertices'].astype(np.float32)
    lineset['lines'] = lineset['lines'].astype(np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(lineset['vertices'])
    line_set.lines = o3d.utility.Vector2iVector(lineset['lines'])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lineset['vertices'])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([line_set, pcd, coordinate_frame])




def test_polylines_to_png():
    npz_file_path = 'C:/studio/Datasets/ABC/wireset/dataset/CurveWiframe/curve_wireframe/009/0094/00944567.npz' 
    sample = np.load(npz_file_path)
    edge_points = sample['edge_points']
    print(edge_points.shape)
    
    save_path = 'C:/studio/Datasets/ABC/wireset/dataset/CurveWiframe/tmp/00944567.png'

    polylines_to_png(edge_points, save_path, dpi=150)
    print(f"Saved to {save_path}")




if __name__ == '__main__':
    test_polylines_to_png()

