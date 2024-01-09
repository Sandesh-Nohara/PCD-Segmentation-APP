import time
# from tqdm.auto import tqdm
from detect_delimiter import detect
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from kneed import KneeLocator
# from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import open3d as o3d
# from collections import Counter
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# import torch
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
from sklearn.neighbors import radius_neighbors_graph
from utils import part_seg2, normalize_tensor


##  Visualization of segment wise point cloud data
def visualization_seg(segment_plane, fig_size=(8,6)):

    color_data = pd.read_excel("pcdfiles/pcd_colors.xlsm")
    color_name = color_data["color_name"].tolist()
    color_code = color_data["rgb_color"].tolist()
    
    fig = plt.figure(figsize = fig_size)
    ax = fig.add_subplot(111, projection='3d')
    legendFig = plt.figure("Legend plot", figsize = fig_size)

    # modelPCD = o3d.geometry.PointCloud()
    seg_color_name = []
    seg_color_code = []
    color_idx = 0
    lines = []
    
    for i in range(len(segment_plane)):# tqdm(range(len(segment_plane)), desc="color_segment: "):
        
        data_points = segment_plane[i]

        # scan_pcd = o3d.geometry.PointCloud()
        # scan_pcd.points = o3d.utility.Vector3dVector(data_points)
        # modelPCD += scan_pcd

        if color_idx > len(color_code)-1:
            color_idx = 0
        paint_color = [float(j) for j in color_code[color_idx].split(",")]
        # scan_pcd.paint_uniform_color(paint_color)
        curr_seg_color_name = f"{i+1}.{color_name[color_idx]}"

        # covnert points numpy array to pandas dataframe
        point_dataframe = pd.DataFrame(data_points, columns = ['x', 'y', 'z'])
        x = point_dataframe['x'].values 
        y = point_dataframe['y'].values 
        z = point_dataframe['z'].values 
        sc = ax.scatter(x, y, z, c = [paint_color], label=curr_seg_color_name)
        lines.append(sc)

        seg_color_name.append(curr_seg_color_name)
        seg_color_code.append(paint_color)
        color_idx += 1

    # # Adding a title
    # plt.suptitle(f"Segmentation Plot \n セグメンテーションプロット", fontname="MS Gothic", fontsize=20)

    # # Adding axes labels
    # # ax.set_xlabel("X Coordinate")
    # # ax.set_ylabel("Y Coordinate")
    # # ax.set_zlabel("Z Coordinate")

    # # Adding legend, which helps us recognize the segment according to it's color
    # plt.legend(bbox_to_anchor=(0.2, 1), fontsize=8, ncol=2)#, loc='best')
    # # plt.legend(fontsize=8, ncol=2, loc='best')
    # # plt.tight_layout()
    # ax.set_facecolor("white")

    legendFig.suptitle("名前付きセグメンテーションカラー \n Segmentation colors with names", fontname="MS Gothic")#, fontsize=20)
    legendFig.legend(lines, seg_color_name, ncol=3, loc='center')#, fontsize=20)
    image_name = f"./static/images/segment_plot_3d.png"
    legendFig.savefig(image_name)


    # n = len(seg_color_name)
    # num_x = n // 10 + 2
    # fig = plt.figure(figsize=(num_x,5))
    # x, y = 0,0
    # for i in range(len(seg_color_name)):
    #     plt.scatter(x, y, color=seg_color_code[i], marker='o', s=100)
    #     plt.text(x+0.3,y, f'{seg_color_name[i]}', c='black')
    #     # x += 1
    #     y -= 1
    #     if y < -9: #-9
    #         x += 1.5
    #         y = 0
    # plt.xlim(-0.3, num_x)

    # plt.axis('off')
    # image_name = f"./static/images/segment_plot_3d.png"
    # plt.savefig(image_name)
    
    plt.close('all')

    return seg_color_name, seg_color_code



def ransac(xyz, threshold, iterations=70000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, plane_cloud = pcd.segment_plane(distance_threshold = threshold, ransac_n = 3,num_iterations = iterations)
    inlier_pcd = pcd.select_by_index(plane_cloud)
    outlier_pcd = pcd.select_by_index(plane_cloud, invert=True)
    inliers = np.asarray(inlier_pcd.points)
    outliers = np.asarray(outlier_pcd.points)

    return inliers, outliers

# def find_node(xyz, method="knn"):
    
#     # Step 1: Point Cloud to Mesh
#     pcd_raw = o3d.geometry.PointCloud()
#     pcd_raw.points = o3d.utility.Vector3dVector(xyz)
#     pcd_raw.estimate_normals()

#     #radius determination
#     distances = pcd_raw.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#     factor = 1
#     if method=="knn":
#         arr = np.asarray(pcd_raw.points)
#         A_sparse = radius_neighbors_graph(arr, avg_dist*factor, mode='connectivity', include_self=False)
#         e1 = A_sparse.nonzero()[0].reshape(-1, 1)
#         e2 = A_sparse.nonzero()[1].reshape(-1, 1)
#         edges = np.concatenate((e1, e2), axis=1)
        
#     arr = np.hstack((np.asarray(pcd_raw.points), np.asarray(pcd_raw.normals)))
#     nodes = torch.tensor(arr, dtype=torch.float32)
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
      
#     data1 = Data(x=nodes, edge_index=edge_index)

#     return data1

# def predict_GCN(pcd, req_label, sequence_length, batch_size, torch_model, device, shuffle_data = True):
#     n = len(pcd)
#     xyz = normalize_tensor(pcd[:, :3])  # normalize the data
#     if shuffle_data:
#         randomize = np.arange(n)
#         np.random.shuffle(randomize)
#         pcd = pcd[randomize]
#         xyz = xyz[randomize]
    
#     Tbatch = n // (batch_size*sequence_length)
#     xyz1 = xyz[: Tbatch * batch_size * sequence_length]

#     split_xyz1 = np.split(xyz1, Tbatch)
#     train_list = []
#     for i in range(Tbatch):
#         xyz11 = split_xyz1[i]
#         data1 = find_node(xyz11[:, :3], method="knn")
#         train_list.append(data1)

#     data_loader =  DataLoader(train_list, batch_size=1, shuffle=False)

#     pred_label = []
#     for pred_data in data_loader:
#         pred_data.to(device)
        
#         predicted_labels = torch_model(pred_data.x, pred_data.edge_index)
#         predicted_labels = torch.argmax(predicted_labels, dim=1, keepdim=True)
#         pred_label.append(predicted_labels.detach().cpu().numpy())

#     pred_label = np.concatenate(pred_label)
#     sequence = pcd[: Tbatch * batch_size * sequence_length]
#     rem_sequence = pcd[Tbatch * batch_size * sequence_length: ]

#     new_xyz = sequence[np.where(pred_label == req_label)[0]]
#     no_new_xyz = sequence[np.where(pred_label != req_label)[0]]
#     if len(no_new_xyz) > 0:
#         if len(rem_sequence) > 0:
#             no_new_xyz = np.vstack((no_new_xyz, rem_sequence))

#     return new_xyz, no_new_xyz


def normal_split(pcd_raw, xyz, threshold):
    pcd_raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # normal threshold
    thresh = 0.95
    # get min max along 3 axes
    min_arr = np.min(xyz, axis=0)
    max_arr = np.max(xyz, axis=0)
    # get spitting list
    int_size = []
    interval_points = []
    for i in range(3):
        ax_range = max_arr[i] - min_arr[i]
        num_intervals = int(ax_range // threshold)
        size_interval = ax_range / num_intervals
        points_interval = [[] for _ in range(num_intervals + 1)]

        int_size.append(size_interval)
        interval_points.append(points_interval)

    
    for j in range(len(pcd_raw.points)):# tqdm(range(len(pcd_raw.points)), desc="normal_split: "):
        normal_vec = pcd_raw.normals[j]
        cos_z = np.abs(normal_vec[2])
        cos_y = np.abs(normal_vec[1])
        # cos_x = np.abs(normal_vec[0])
        curr_point  = pcd_raw.points[j].tolist()
        if (cos_z > thresh) or (cos_y > thresh):
            if cos_z > thresh:
                z_int_idx = int((curr_point[2] - min_arr[2]) / int_size[2])
                interval_points[2][z_int_idx].append(curr_point)
            if cos_y > thresh:
                y_int_idx = int((curr_point[1] - min_arr[1]) / int_size[1])
                interval_points[1][y_int_idx].append(curr_point)
        else:
            x_int_idx = int((curr_point[0] - min_arr[0]) / int_size[0])
            interval_points[0][x_int_idx].append(curr_point)

    return interval_points


def outlier_removal(xyz, num_neighbors = 30, contamination = 0.01):
    
    clf = LocalOutlierFactor(n_neighbors=num_neighbors, contamination=contamination)
    outlier_mask = clf.fit_predict(xyz) == -1
    inliers = xyz[~outlier_mask]
    outliers = xyz[outlier_mask]

    return inliers, outliers



def read_pcd_file(filepath, downsample=True):

    if filepath.endswith('.ply'):
        pcd_raw = o3d.io.read_point_cloud(filepath)

    elif filepath.endswith('.xyz') or filepath.endswith('.txt'):
        #detect delimiter
        f = open(filepath)
        for _ in range(3):
            line = f.readline()
        delimiters = detect(line)

        pcd = np.loadtxt(filepath, skiprows=1, delimiter=delimiters)
        xyz = pcd[:, :3]

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(xyz)
    
    xyz = np.asarray(pcd_raw.points)
    # print(f"Original point cloud shape : {xyz.shape}")

    if len(xyz) > 30000000 and downsample:
        k_points = int(np.round(len(xyz) / 25000000))
        pcd_raw = pcd_raw.uniform_down_sample(every_k_points = k_points)
        xyz = np.asarray(pcd_raw.points)
        # print(f"(Uniformly) Downsampled point cloud shape : {xyz.shape}")

    distances = pcd_raw.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    thresh_distance = avg_dist * 10
    return pcd_raw, xyz, thresh_distance


def main1(pcd_raw, xyz, threshold_org):

    sequence_length = 1024
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model_name = "gcn"
    
    # if model_name == "gcn":
    #     batch_size = 16
    #     torch_model = part_seg2(6, 3)
    # elif model_name == "gcn1":
    #     batch_size = 32
    #     torch_model = part_seg2(6, 3)
        
    batch_size = 32
    # torch_model = part_seg2(6, 3)
    # model_path = f"{model_name}_Checkpoint.pt"
    # checkpoint = torch.load(model_path)
    # torch_model.load_state_dict(checkpoint['state_dict'])
    # torch_model.to(device)
    # torch_model.eval()


    interval_points = normal_split(pcd_raw, xyz, threshold_org*12)

    segment_planes = []
    still_not_used = []
    for j in range(3):
        num = 2-j
        req_label = 1
        if num == 2:
            req_label = 0
        split_xyz = interval_points[num]

        for i in range(len(split_xyz)): # tqdm(range(len(split_xyz)), desc=f"segment_{num}: "):
            req_xyz = np.array(split_xyz[i])
            if len(req_xyz) < 1024:
                continue
            inlier2, outlier2 = ransac(req_xyz, threshold_org)

            # ignore along z axis
            if num != 3:

                if len(inlier2) > batch_size*sequence_length:
                    # perform GNN
                    if num == 4:
                        # inlier, outlier = predict_GCN(inlier2, req_label, sequence_length, batch_size, torch_model, device, shuffle_data = True)
                        # inlier, outlier = outlier_removal(inlier1, 100, threshold_org)
                        
                        if len(inlier) > int(len(inlier2) * (2/3)):
                            segment_planes.append(inlier)
                            still_not_used.append(outlier)
                            # still_not_used.append(outlier1)
                            still_not_used.append(outlier2)
                        else:
                            still_not_used.append(req_xyz)
                    
                    else:
                        inlier, outlier = inlier2, outlier2

                        if len(inlier) > int(len(req_xyz) * (2/3)):
                            segment_planes.append(inlier)
                            still_not_used.append(outlier)
                        else:
                            still_not_used.append(req_xyz)

                else:
                    still_not_used.append(req_xyz)
            
            else:
                still_not_used.append(req_xyz)
                
    not_used_pcd = []
    for k in still_not_used:
        if len(k) > 0:
            not_used_pcd.append(k)
    rem_pcd = o3d.geometry.PointCloud()
    if len(not_used_pcd) > 0:
        not_used_pcd = np.vstack(not_used_pcd)
        rem_pcd.points = o3d.utility.Vector3dVector(not_used_pcd)

    # segment_planes.append(not_used_pcd)
    return segment_planes, rem_pcd


def main(filepath):
    start = time.time()

    pcd_raw, xyz, threshold_org = read_pcd_file(filepath, downsample=True)

    segment_planes, rem_pcd = main1(pcd_raw, xyz, threshold_org)
    
    seg_color_name, seg_color_code = visualization_seg(segment_planes, fig_size=(4,3))

    segment_planes.append(rem_pcd.points)
    seg_color_name.append('RestPCD')
    seg_color_code.append([0.6, 0.6, 0.6])


    end = time.time()
    print(f"Total time for segmentation: {(end - start)/60:.2f} mins")
    return segment_planes, seg_color_name, seg_color_code
