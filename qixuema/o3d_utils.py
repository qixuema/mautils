import os
import open3d as o3d
import numpy as np

# from misc.io_utils import read_vertices_obj

def point2sphere(points):
    # # pcd = o3d.io.read_point_cloud(file_path)
    # # 读取 OBJ 文件
    # if not os.path.exists(file_path):
    #     print(f"The file {file_path} does not exist.")
        
    
    # # mesh = o3d.io.read_triangle_mesh(file_path)
    # points = read_vertices_obj(file_path)

    # 提取点云
    # point_cloud = mesh.vertices
    
    # points = point_cloud
    
    mesh_balls = []
    for point in points:
        # 为每个顶点创建一个 mesh 球
        mesh_ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        mesh_ball.translate(point)
        mesh_balls.append(mesh_ball)
    
    # 合并所有小球到一个 mesh
    combined_mesh = mesh_balls[0]
    for mesh_ball in mesh_balls[1:]:
        combined_mesh += mesh_ball
        
    # o3d.io.write_triangle_mesh(file_path.replace('.obj', '_balls.obj'), combined_mesh)
    return combined_mesh

def get_vertices_obb(vertices, noise_scale=1e-4):
    # 为了数值稳定性，添加一些噪声，避免所有点云都在一个平面上
    noise = noise_scale * np.random.uniform(-1, 1, vertices.shape)
    new_vertices = noise + vertices

    # 将 numpy 数组转换为 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(new_vertices)

    # 计算 OBB（有向包围盒）
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
