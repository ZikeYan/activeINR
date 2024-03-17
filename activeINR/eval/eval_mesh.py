import numpy as np

import open3d as o3d
import trimesh
from scipy.spatial import KDTree
import argparse
import json

parser = argparse.ArgumentParser(description="NExplore.")
parser.add_argument("--config", type=str, required=True, help="input json config")
parser.add_argument("--file", default="None", help="recorded mesh")
parser.add_argument("--scene_id", default="None", help="specify test scene")
parser.add_argument("--visualize", action="store_true", default=False, help="visualize the mesh")
args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args
scene_id = args.scene_id
with open(args.config) as f:
    config = json.load(f)
    gt_folder = config["dataset"]["root"]

gt_mesh_file = gt_folder + scene_id + ".glb"
ours_file = args.file

our_mesh = o3d.io.read_triangle_mesh(ours_file)
our_mesh.paint_uniform_color([1, 0.706, 0])
our_mesh.compute_vertex_normals()
gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file)
gt_mesh.paint_uniform_color([0, 0.906, 0])
gt_mesh.compute_vertex_normals()
T_rot = np.eye(4)
T_rot[1, 1], T_rot[2, 2] = 0, 0
T_rot[1, 2], T_rot[2, 1] = -1, 1
T_rot[1, 3] = 1.25
gt_mesh.transform(T_rot)
if args.visualize:
    o3d.visualization.draw_geometries([our_mesh, gt_mesh])

our_tri_mesh = trimesh.Trimesh(np.asarray(our_mesh.vertices), np.asarray(our_mesh.triangles))
gt_tri_mesh= trimesh.Trimesh(np.asarray(gt_mesh.vertices), np.asarray(gt_mesh.triangles))

def accuracy_comp_ratio(mesh_gt, mesh_rec, samples=200000):

    rec_pc = trimesh.sample.sample_surface(mesh_rec, samples)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, samples)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    # ------------- evaluation metrics
    def completion_ratio(gt_points, rec_points, dist_th=0.05):
        gen_points_kd_tree = KDTree(rec_points)
        distances, _ = gen_points_kd_tree.query(gt_points)
        comp_ratio = np.mean((distances < dist_th).astype(np.float))
        return distances, comp_ratio

    def accuracy(gt_points, rec_points):
        gt_points_kd_tree = KDTree(gt_points)
        distances, _ = gt_points_kd_tree.query(rec_points)
        acc = np.mean(distances)
        return distances, acc

    def completion(gt_points, rec_points):
        rec_points_kd_tree = KDTree(rec_points)
        distances, _ = rec_points_kd_tree.query(gt_points)
        comp = np.mean(distances)
        return distances, comp

    _, acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    _, comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    _, ratio = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)

    return acc, comp, ratio

acc, comp, ratio = accuracy_comp_ratio(gt_tri_mesh, our_tri_mesh)
iacc, icomp, iratio = accuracy_comp_ratio(our_tri_mesh, gt_tri_mesh)
# acc: ACC
# 1-iratio: FPR
# print(acc, comp, 1-ratio)
# print(iacc, icomp, 1-iratio)

print(f"ACC: {acc*100:.2f}, FPR: {(1-iratio)*100:.2f}")
with open(ours_file + ".txt", "w") as result_file:
    result_file.write(f"ACC: {acc*100:.2f}, FPR: {(1-iratio)*100:.2f}")