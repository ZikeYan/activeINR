import torch
import numpy as np
import os

import open3d as o3d
import trimesh

scene_id = "Denmark"
gt_folder = "/home/yan/Dataset/gibson_habitat/gibson/"
gt_mesh_file = gt_folder + scene_id + "/" + scene_id + "_mesh.ply"
ours_file = "logs/final/gibson/" + scene_id + "/step+_0/mesh.ply"

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

    acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    ratio = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)

    return acc, comp, ratio

acc, comp, ratio = accuracy_comp_ratio(gt_tri_mesh, our_tri_mesh)
iacc, icomp, iratio = accuracy_comp_ratio(our_tri_mesh, gt_tri_mesh)
print(acc, comp, 1-ratio)
print(iacc, icomp, 1-iratio)