import argparse
import numpy as np
import os
import sys
import random
import pickle
import open3d as o3d
from einops import rearrange, repeat
from qixuema.helpers import get_file_list, get_filename_wo_ext, get_file_list_with_extension
from qixuema.io_utils import read_obj_file, write_obj_file



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from misc.helpful_fn import get_file_list, get_filename_wo_ext, get_file_list_with_ext
# from misc.io_utils import read_obj_file, write_obj_file
from qixuema.geo_utils import (
    check_order_single_point,
    inverse_transform_polyline,
    edge_points_to_lineset,
)

# from misc.trimesh_utils import create_column_from_lineset




def rotation_matrix_z(theta):
    theta_rad = np.radians(theta)  # 将角度转换为弧度
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])

def rotation_matrix_x(theta):
    theta_rad = np.radians(theta)  # 将角度转换为弧度
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])

def rotation_matrix_y(theta):
    theta_rad = np.radians(theta)  # 将角度转换为弧度
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

# 旋转点云
def rotate_points(points, theta, axis=2):
    if axis == 0:
        R = rotation_matrix_x(theta)  # 生成旋转矩阵
    elif axis == 1:
        R = rotation_matrix_y(theta)  # 生成旋转矩阵
    elif axis == 2:        
        R = rotation_matrix_z(theta)  # 生成旋转矩阵
    else:
        return points
    
    return points @ R.T  # 应用旋转矩阵


def vis_curveset(lineset, norm_edge_points=None, edge_points=None, use_edge_points=False, is_check_order=False, vis=False):
    # 可视化检查一下是否合理
    segments = lineset['vertices'][lineset['lines']]
    show_original_edge_points = True
    if show_original_edge_points:
        # original_edge_points
        original_edge_points = []
        for i, start_and_end in enumerate(segments):
            if use_edge_points and edge_points is not None:
                original_start = edge_points[i][0]
                original_end = edge_points[i][-1]
                start_and_end = np.stack((original_start, original_end), axis=0)

            if is_check_order:
                if  not check_order_single_point(start_and_end[0], start_and_end[1]):
                    start_and_end = start_and_end[::-1]


            if norm_edge_points is not None:
                start_end = np.array(
                    [[0.0, 0.0, 0.0], 
                    [0.54020254, -0.77711392, 0.32291667]]
                )
                original_edge_points_i = inverse_transform_polyline(norm_edge_points[i], start_and_end=start_end)             
                
                original_edge_points_i = inverse_transform_polyline(original_edge_points_i, start_and_end=start_and_end)             
            else:
                original_edge_points_i = edge_points[i]

            original_edge_points.append(original_edge_points_i)

        edge_points = np.array(original_edge_points)

    new_vertices, new_lines = edge_points_to_lineset(edge_points)

    if vis:
        
        if new_vertices.dtype != np.float64:
            new_vertices = new_vertices.astype(np.float64)
        if new_lines.dtype != np.int32:
            new_lines = new_lines.astype(np.int32)

        new_line_set = o3d.geometry.LineSet()
        new_line_set.points = o3d.utility.Vector3dVector(new_vertices)
        new_line_set.lines = o3d.utility.Vector2iVector(new_lines)

        pcd = o3d.geometry.PointCloud()
        new_vertices = np.unique(new_vertices, axis=0)
        pcd.points = o3d.utility.Vector3dVector(new_vertices)
        
        o3d.visualization.draw_geometries([new_line_set, pcd], window_name='lineset', width=1600, height=1200) 
    else:
        return new_vertices, new_lines, edge_points


def vis_data(sample_iterator):

    def update_mesh(vis):

        print("Generated new mesh, updating visualization...")
        vis.clear_geometries()

        sample = next(sample_iterator)
        
        if isinstance(sample, dict):
            # vertices = sample['vertices']
            # lines = sample['lines']
                        
            segments = sample['segments']
            num_lines = len(segments)
            while num_lines < 128:
                sample = next(sample_iterator)
                segments = sample['segments']
                num_lines = len(segments)
                # return
            
            vertices = segments.reshape(-1, 3)
            lines = np.arange(0, len(vertices)).reshape(-1, 2)
            
            uid = sample['uid']
            print(f"Loading {uid}...")
        elif isinstance(sample, np.ndarray):
            assert sample.ndim == 2
            vertices = sample
            num_points = len(vertices)
            lines = np.array([[i, i+1] for i in range(num_points - 1)])


        # uid = sample['uid']
        # group_id = uid[:3]
        # chunck_id = uid[:4]
        # file_path = 'D:/studio/NewDatasets/ABC/npz_order_curveset/npz_128_aug' + '/' + group_id + '/' + chunck_id + '/' + uid + '.npz'
        # sample = np.load(file_path)

        # vertices = sample['vertices']
        # lines = sample['lines']
        
        print(len(lines))  
        
        # 判断 sample 有没有 norm_edge_points 这个 key
        if 'norm_edge_points' in sample:            
            vertices, lines = vis_curveset(
                    sample, 
                    # edge_points=sample['edge_points'], 
                    norm_edge_points=sample['norm_edge_points'],
                    use_edge_points=False,
                )
        
        line_set = o3d.geometry.LineSet()
        if vertices.dtype != np.float64:
            vertices = vertices.astype(np.float64)
        if lines.dtype != np.int32:
            lines = lines.astype(np.int32)

        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        vis.add_geometry(line_set)
        pcd = o3d.geometry.PointCloud()
        vertices = np.unique(vertices, axis=0)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis.add_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        print("Visualization updated.")
        return False

    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600)

    # 注册空格键回调，刷新网格
    vis.register_key_callback(ord(' '), update_mesh)

    print("Press Space to generate and display a new mesh.")
    print("Press W to show wireframe.")
    print("Press Q to quit.")

    # 开始可视化
    vis.run()
    vis.destroy_window()            


def main(dir_path):

    # file_path_list = get_file_list_with_ext(dir_path, ext='.npy')
    file_path_list = get_file_list(dir_path)
    
    # file_path_list = ['D:/studio/NewDatasets/ABC/edge_points/npy_shape_normalize/00000387.npy']
    # file_path_list = ['D:/studio/NewDatasets/ABC/edge_points/npy_shape_normalize/00480291_1.npy']
    

    # test_uid_file_path = "D:/studio/Projects/wireflow_exp/wireflow_exp/data/pc_cond/test_uids.txt"

    # with open(test_uid_file_path, "r") as f:
    #     test_uid_list = [line.strip().split('_')[0] for line in f.readlines()]
    
    
    random.seed(4231)

    # file_path_list.sort(reverse=True)
    random.shuffle(file_path_list)

    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/discretized_curveset/006/0069/00691882.npz'] #
    # file_path_list = ['D:/studio/NewDatasets/ABC/curve_vae_recon/curveset_recon/recon_00000003_curves.npy']
    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/npz_128_aug/002/0026/00265263_0.npz'] #
    # file_path_list = ['D:/studio/NewDatasets/ABC/recon/curveset/00000.npy']
    # file_path_list = ['D:/studio/NewDatasets/ABC/recon/lineset/00265263_1.npy']
    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/000/0000/00006449.npz']
    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/001/0016/00167957.npz']
    # file_path_list = ['C:/studio/Datasets/ABC/npz_order_curveset/subdivided_curveset/00967995.npz']
    # file_path_list = ['C:/studio/Datasets/ABC/npz_order_curveset/wireset_data_16/00691882.npz']
    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/unique_data/00269463.npz']
    # file_path_list = ['D:/studio/NewDatasets/ABC/npz_order_curveset/006/0069/00691882.npz']
    # file_path_list = ['C:/studio/Datasets/ABC/npz_order_curveset/wireset_data_16/00000003.npz']

    # file_path_list = ['C:/studio/Datasets/ABC/npz_order_curveset/wireset_data_for_condition/105416_00024677.npz']
    # file_path_list = ['D:/studio/Projects/wiregen/data/wireset_recon/wireset_continuous/00024677_6.npy']
    
    # file_path_list = ['D:/studio/Projects/wiregen/data/wireset_recon/vae_interpolation/00356153_00702088/00356153_00702088_z_53.npy']
    # file_path_list = ['D:/studio/Projects/wiregen/data/wireset_recon/wireset_32to96_80pt_fixstd/114572_00114673.npy']
    # file_path_list = ['D:/studio/Projects/wireflow_exp/wireflow_exp/data/img_cond/img_cond_dino_slim/00705082'+'_0.npy']
    
    # file_path_list = ["C:/studio/Datasets/ABC/npz_order_curveset/64to128_unqiue/00000003.npz"]
    
    # file_path_list = ['D:/studio/Projects/wireflow_exp/wireflow_exp/data/pipeline/overview/00000003_norm_edge_points.npy']
    # file_path_list = ["D:/studio/Projects/wireflow_exp/wireflow_exp/data/pc_cond/wireset_pc_cond_pointnet/00634380.npy"]

    file_path_list_iterator = iter(file_path_list)
    
    current_file_path = None

    check = True

    def update_mesh(vis):
        
        nonlocal check
        
        global current_file_path

        print("Generated new mesh, updating visualization...")
        vis.clear_geometries()

        file_path = next(file_path_list_iterator)

        uid = get_filename_wo_ext(file_path)
        
        # if check:
        #     while uid != '00696934':
        #         file_path = next(file_path_list_iterator)
        #         uid = get_filename_wo_ext(file_path)
        #     check = False

        if file_path.endswith('npz'):
            sample = np.load(file_path, allow_pickle=True)
            uid = get_filename_wo_ext(file_path)
            # uid = uid.split('_')[1]
            
            abc_edge = True

            if abc_edge:
                

                while len(sample['lines']) < 48 or len(sample['lines']) > 128:
                    file_path = next(file_path_list_iterator)
                    sample = np.load(file_path, allow_pickle=True)
                    # return
                    pass                
                
                # get_it = False
                
                # while not get_it:
                #     norm_edge_points = sample['norm_edge_points']
                #     # 计算每条 norm edge points 的 bbox 的 体积
                #     for i in range(len(norm_edge_points)):
                #         # bbox = get_bbox(norm_edge_points[i])
                #         bbox = np.array([np.min(norm_edge_points[i], axis=0), np.max(norm_edge_points[i], axis=0)])
                #         volume = np.prod(bbox[1] - bbox[0])
                #         if volume > 0.5:
                #             get_it = True
                #             break
                    
                #     if not get_it:
                #         file_path = next(file_path_list_iterator)
                #         sample = np.load(file_path, allow_pickle=True)
                #         # return
                #         pass
                    
                

                
                # while uid not in test_uid_list:
                #     file_path = next(file_path_list_iterator)
                #     sample = np.load(file_path, allow_pickle=True)
                #     uid = get_filename_wo_ext(file_path)
                #     uid = uid.split('_')[1]
                
                # print(sample['lines'].shape)
                
                
                # edge_points = sample['edge_points']
                # norm_edge_points = normalize_edges_points(edge_points)
                
                
                # vertices = sample['vertices']
                
                # # 将 vertices z 值下雨 -0.6 的都增加 0.5
                # vertices[:, 2] = np.where(vertices[:, 2] < -0.6, vertices[:, 2] + 0.5, vertices[:, 2])
                # vertices[:, 2] = np.where(vertices[:, 2] > 0.58, vertices[:, 2] + 0.1, vertices[:, 2])
                
                
                
                # sample['vertices'] = vertices
                
                # sample = {
                #     'vertices': vertices,
                #     'lines': sample['lines'],
                #     'norm_edge_points': sample['norm_edge_points'],
                #     # 'edge_points': sample['edge_points'],
                # }

                vertices, lines, edge_points = vis_curveset(
                    sample, 
                    # edge_points=sample['edge_points'], 
                    norm_edge_points=sample['norm_edge_points'],
                    # norm_edge_points=norm_edge_points,
                    # use_edge_points=False,
                    use_edge_points=True,
                )
                
                # vertices = sample['norm_edge_points'].reshape(-1, 3)
                # lines = np.arange(0, len(vertices)).reshape(-1, 2)



                # print(vertices.shape)
                
                
                # np.save(f"D:/studio/Projects/wireflow_exp/wireflow_exp/data/wf2face/{uid}.npy", edge_points)
                
                
                # # # save edge_points
                # np.save(f"D:/studio/Projects/wireflow_exp/wireflow_exp/data/DiffAdjs/{uid}.npy", edge_points)

            if False:

                # vertices = sample[:, :256, :]

                # 生成 64 个均匀分布的索引，包含第一个和最后一个点
                # indices = np.linspace(0, 256 - 1, 256, dtype=int)

                # 使用生成的索引进行采样
                # sampled_points = vertices[:, indices, :]
                
                # sampled_points = sample['edge_points']
                vertices = sample['vertices']
                lines = sample['lines']
                sampled_points = vertices[lines]

                # sampled_points = vertices[:, ::4, :]
                num_points = sampled_points.shape[1]
                points_index = np.arange(num_points)
                lines = np.column_stack((points_index, np.roll(points_index, -1)))[:-1]
                bs = sampled_points.shape[0]
                lines = repeat(lines, 'nl c -> b nl c', b=bs)
                # lines = np.repeat(lines, len(sample), axis=0)
                # 再添加一些 offset，因为第二个 batch 的点云是从 256 开始的
                
                # 创建 len(edge_pnts) 个 offset, 每个 offset 都是 256 的倍数
                offset = np.arange(bs) * num_points
                offsets = rearrange(offset, 'b -> b 1 1')
                lines += offsets
                lines = lines.reshape(-1, 2)
                vertices = sampled_points.reshape(-1, 3)
            
            elif False:
                vertices = sample['vertices']
                lines = sample['lines']

        elif file_path.endswith('npy'):
            
            batch_id = uid.split('_')[0]
            

            
            sample = np.load(file_path)
            
            
            # sample = sample['curves']
            
            # print(sample.shape)


            # while sample.shape[0] > 64 or sample.shape[0] < 48:
            #     file_path = next(file_path_list_iterator)
            #     sample = np.load(file_path)
            
            print(sample.shape)
            
            # sample = sample['curves']
            
            abc_edge = True
            is_line_segments = False


            if is_line_segments:
                # sample = rearrange(sample, 'b (n c) -> b n c', n=2)
                # first_corner = sample[..., 0, :]
                # second_corner = sample[..., -1, :]
                # vertices = np.concatenate([first_corner, second_corner], axis=1)

                vertices = sample.reshape(-1, 3)
                # vertices = vertices.reshape(-1, 3)
                lines = np.arange(0, len(vertices)).reshape(-1, 2)

            if abc_edge:

                # vertices = sample[:, :256, :]
                vertices = sample

                # 生成 64 个均匀分布的索引，包含第一个和最后一个点
                indices = np.linspace(0, 256 - 1, 64, dtype=int)

                # 使用生成的索引进行采样
                sampled_points = vertices[:, indices, :]
                
                # sampled_points = sample
                
                # print(sample.shape)

                # sampled_points = vertices[:, ::4, :]
                num_points = sampled_points.shape[1]
                points_index = np.arange(num_points)
                lines = np.column_stack((points_index, np.roll(points_index, -1)))[:-1]
                bs = sample.shape[0]
                lines = repeat(lines, 'nl c -> b nl c', b=bs)
                # lines = np.repeat(lines, len(sample), axis=0)
                # 再添加一些 offset，因为第二个 batch 的点云是从 256 开始的
                
                # 创建 len(edge_pnts) 个 offset, 每个 offset 都是 256 的倍数
                offset = np.arange(bs) * num_points
                offsets = rearrange(offset, 'b -> b 1 1')
                lines += offsets
                lines = lines.reshape(-1, 2)
                vertices = sampled_points.reshape(-1, 3)

                # first_corner = sample[:, 0]
                # last_corner = sample[:, -1]

                # corners = np.concatenate([first_corner, last_corner], axis=1)
                # vertices = corners.reshape(-1, 3)
                # lines = np.arange(len(vertices)).reshape(-1, 2)


            # lines = np.arange(0, len(vertices))
            # lines = np.roll(lines, 1).reshape(-1, 2)
            # lines = np.concatenate([lines, np.roll(lines, 1, axis=1)], axis=0)

        elif file_path.endswith('obj'):
            lineset = read_obj_file(file_path)
            vertices = lineset['vertices']
            lines = lineset['lines']
        elif file_path.endswith('pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # sample = random.choice(data)
            # vertices = sample['vertices']
            # lines = sample['lines']
            edge_pnts = data['edge_pnts']
            points_index = np.arange(256)
            lines = np.column_stack((points_index, np.roll(points_index, -1)))
            lines = np.repeat(lines, len(edge_pnts), axis=0)
            # 再添加一些 offset，因为第二个 batch 的点云是从 256 开始的
            
            # 创建 len(edge_pnts) 个 offset, 每个 offset 都是 256 的倍数
            offset = np.arange(len(edge_pnts)) * 256
            offsets = rearrange(offset, 'b -> b 1 1')
            lines += offsets


            vertices = edge_pnts.reshape(-1, 3)
            # lines = np.arange(len(vertices)).reshape(-1, 2)
            
        else:
            print(f"Unsupported file type: {file_path}")
            return False


        # write_obj_file(
        #     'D:/studio/NewDatasets/ABC/npz_order_curveset/obj_tmp/' + uid + '.obj', vertices, lines, is_line=True)

        print(f"Loading {file_path}...")

        current_file_path = file_path

        # save edge_points
        # np.save(f"D:/studio/Projects/wireflow_exp/wireflow_exp/data/pc_cond/edge_points/{uid}.npy", edge_points)


        # 将顶点绕 y 轴旋转 90 度
        # vertices = rotate_points(vertices, 90, axis=0)
        # vertices = rotate_points(vertices, 18, axis=1)
    
        
        line_set = o3d.geometry.LineSet()
        if vertices.dtype != np.float64:
            vertices = vertices.astype(np.float64)
        if lines.dtype != np.int32:
            lines = lines.astype(np.int32)

        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        vis.add_geometry(line_set)

        # lineset = {'vertices': vertices, 'lines': lines}
        # columns = create_column_from_lineset(lineset, radius=0.005)

        # # 1. 从trimesh中获取顶点和面
        # mesh_vertices = np.asarray(columns.vertices)  # (N, 3) 数组，表示顶点坐标
        # mesh_faces = np.asarray(columns.faces)  # (M, 3) 数组，表示每个三角形的顶点索引

        # # 2. 创建一个open3d的TriangleMesh对象
        # o3d_mesh = o3d.geometry.TriangleMesh()

        # # 3. 设置顶点和三角形面
        # o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        # o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)

        # vis.add_geometry(o3d_mesh)


        pcd = o3d.geometry.PointCloud()
        vertices = np.unique(vertices, axis=0)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis.add_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        print("Visualization updated.")
        return False


    # def copy_file():
    #     global current_file_path
        
    #     print(current_file_path)
    


    # 创建可视化器
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1200, height=900)

    # 注册空格键回调，刷新网格
    vis.register_key_callback(ord(' '), update_mesh)
    # vis.register_key_callback(ord('k'), copy_file)
    

    print("Press Space to generate and display a new mesh.")
    print("Press W to show wireframe.")
    print("Press Q to quit.")

    # 开始可视化
    vis.run()
    vis.destroy_window()    
    exit(0)

def vis_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)


    random.shuffle(data)
    # file_path_list.sort()

    # data = sorted(data, key=lambda x: x['uid'])

    # file_path_list = ['D:/studio/Datasets/ABC/wf_gen/10_to_256/npy_256_float16/00183605_2.npy']

    sample_iterator = iter(data)

    vis_data(sample_iterator)

def vis_npy(npy_path):
    sample = np.load(npy_path)
    sample_iterator = iter(sample)
    vis_data(sample_iterator)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert big3 to lines_with_room_codes.")
    parser.add_argument("--path", required=True, type=str,
                        help="big3 path (outer wall, room lines, roof lines) ", metavar="DIR")
    parser.add_argument("--path_2", required=False, type=str,
                        help="big3 path (outer wall, room lines, roof lines) ", metavar="DIR")
    parser.add_argument("--num_processes", required=False, type=int,
                        help="number of processes", metavar="INT", default=2)

    args, unknown = parser.parse_known_args()  # unknown contains any extra arguments
    
    if unknown:
            print(f"Ignoring unknown arguments: {unknown}")
    
    return args

if __name__=='__main__':

    args = parse_args()
    dir_path = args.path

    main(dir_path=dir_path)
    file_path = 'D:/studio/NewDatasets/ABC/npz_order_curveset/pkls/unique_lineset_with_128_aug.pkl'
    # vis_data(file_path)
    # vis_pkl(file_path)

    # npy_path = 'raw_fwe_pts.npy'
    # npy_path = 'ditem.npy'
    npy_path = 'C:/studio/Datasets/ABC/npz_order_curveset/dataset/all_curves_unique.npy'
    vis_npy(npy_path)