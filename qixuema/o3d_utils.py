import numpy as np
import open3d as o3d


def point2sphere(points):
    """Convert points to a mesh of small spheres."""
    mesh_balls = []
    for point in points:
        mesh_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        mesh_ball.translate(point)
        mesh_balls.append(mesh_ball)

    combined_mesh = mesh_balls[0]
    for mesh_ball in mesh_balls[1:]:
        combined_mesh += mesh_ball

    return combined_mesh

def get_vertices_obb(vertices, noise_scale=1e-4):
    """Compute oriented bounding box for vertices, adding noise for numerical stability."""
    noise = noise_scale * np.random.uniform(-1, 1, vertices.shape)
    new_vertices = noise + vertices

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(new_vertices)

    obb = point_cloud.get_oriented_bounding_box()
    return obb.center, obb.extent, obb.R



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
