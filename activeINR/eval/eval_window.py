import numpy as np
import threading
import time
import imgviz
import cv2
import skimage.measure
import matplotlib.pylab as plt
import torch

import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from activeINR.visualisation import draw3D
from scipy.spatial import KDTree
from copy import deepcopy

from activeINR import geometry
from activeINR.datasets import sdf_util
from activeINR.datasets.dataloader import HabitatDataScene
from habitat.utils.visualizations import maps
import matplotlib.cm
from activeINR.modules.planner.ddppo_policy import DdppoPolicy
import os
import json


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

def compute_current_pcd(self, obs):
    data = self.trainer.scene_dataset.process(obs)
    image = data['image']
    depth = data['depth']
    T_WC_np = data['T']
    depth_resize = imgviz.resize(
        depth, width=self.trainer.W_vis_curr,
        height=self.trainer.H_vis_curr,
        interpolation="nearest")
    img_resize = imgviz.resize(
        image, width=self.trainer.W_vis_curr,
        height=self.trainer.H_vis_curr)
    pcd_cam = geometry.transform.pointcloud_from_depth(depth_resize, self.trainer.fx_vis_curr, self.trainer.fy_vis_curr,self.trainer.cx_vis_curr,self.trainer.cy_vis_curr)
    pcd_cam = pcd_cam.reshape(-1, 3).astype(np.float32)
    pcd_cam = np.einsum('ij,kj->ki', T_WC_np[:3, :3], pcd_cam)
    pcd_curr = pcd_cam + T_WC_np[:3, 3]

    return pcd_curr

class EvaWindow:

    def __init__(self, trainer, explorer, mapper, font_id, action_file):
        self.trainer = trainer
        self.explorer = explorer
        self.mapper = mapper
        self.step = 0
        self.optim_times = []

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            trainer.W, trainer.H, trainer.fx, trainer.fy, trainer.cx, trainer.cy)

        mode = "incremental"
        self.action_file = action_file
        
        self.data_source = HabitatDataScene([self.trainer.H, self.trainer.W, self.trainer.hfov], self.explorer.config,
                                            config_file=self.explorer.config_file,
                                            scene_id=self.explorer.config.scene_id)
        print("========================== ", self.trainer.dataset_format, " - ", self.explorer.config.scene_id)

        f = open(self.action_file)
        actions = f.readline()

        if self.trainer.gt_scene:
            self.gt_mesh = o3d.io.read_triangle_mesh(self.trainer.scene_file)
            self.gt_mesh.compute_vertex_normals()
            T_rot = np.eye(4)
            T_rot[1, 1], T_rot[2, 2] = 0, 0
            T_rot[1, 2], T_rot[2, 1] = -1, 1
            T_rot[1, 3] = self.data_source.agent_height
            self.gt_mesh.transform(T_rot)

        env = self.data_source.sim
        env.reset()

        output_dir = self.trainer.log_dir + self.trainer.dataset_format \
                     + "/" + self.explorer.config.scene_id + "/results"
        print(output_dir)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        f_error = open(output_dir + "/action_error.txt", "w")

        import trimesh
        gt_mesh_tri = trimesh.Trimesh(np.asarray(self.gt_mesh.vertices), np.asarray(self.gt_mesh.triangles))

        gt_samples = trimesh.sample.sample_surface(gt_mesh_tri, 200000)
        import trimesh
        gt_pc_tri = trimesh.PointCloud(vertices=gt_samples[0])
        min_distance = np.ones(gt_samples[0].shape[0])

        while True:
            t0 = time.time()
            print(self.step, " ", self.explorer.config.scene_id)
            # training steps ------------------------------
            sim_obs = env.get_sensor_observations()
            data_source = env._sensor_suite.get_observations(sim_obs)
            curr_pose = env.get_agent_state()
            pose_position = curr_pose.position
            pose_rotation = curr_pose.rotation
            self.trainer.frame_id = self.step
            image = data_source['rgb'][:, :, :3].detach().cpu().numpy()  # [:,:,:3]
            depth = data_source['depth'].detach().cpu().numpy()
            depth = np.squeeze(depth)
            pcd =self.compute_current_pcd([image, depth, pose_position, pose_rotation])

            curr_pcd_tri = trimesh.PointCloud(vertices=np.array(pcd))

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


            test1, accuracy = accuracy(gt_pc_tri.vertices, curr_pcd_tri.vertices)
            distance, completion = completion(gt_pc_tri.vertices, curr_pcd_tri.vertices)
            test2, ratio = completion_ratio(gt_pc_tri.vertices, curr_pcd_tri.vertices)
            print(test1.shape, test2.shape, distance.shape)
            print("instant ", accuracy, completion, ratio)
            min_distance[min_distance>distance] = distance[min_distance>distance]
            min_comp = np.mean(min_distance)
            min_comp_ratio = np.mean((min_distance < 0.05).astype(np.float))
            print("updated", min_comp, min_comp_ratio)


            act = int(actions[self.step])
            #act = np.random.randint(0, 4)
            f_error.write(f"{min_comp}  {min_comp_ratio}\n")

            t1 = time.time()
            self.optim_times.append(t1 - t0)

            self.step += 1

            #action_space = ["STOP","MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
            if (act in [1,2,3]):
                env.step(act)
            else:
                print("Action out of the action space: ", act)
                act = 0

            if self.step == self.trainer.n_steps:
                f_error.close()
                t_end = time.time()
                break
        time.sleep(0.5)
        
    def compute_current_pcd(self, obs):
        data = self.trainer.scene_dataset.process(obs)
        image = data['image']
        depth = data['depth']
        T_WC_np = data['T']
        depth_resize = imgviz.resize(
            depth, width=self.trainer.W_vis_curr,
            height=self.trainer.H_vis_curr,
            interpolation="nearest")
        img_resize = imgviz.resize(
            image, width=self.trainer.W_vis_curr,
            height=self.trainer.H_vis_curr)
        pcd_cam = geometry.transform.pointcloud_from_depth(depth_resize, self.trainer.fx_vis_curr, self.trainer.fy_vis_curr,self.trainer.cx_vis_curr,self.trainer.cy_vis_curr)
        pcd_cam = pcd_cam.reshape(-1, 3).astype(np.float32)
        pcd_cam = np.einsum('ij,kj->ki', T_WC_np[:3, :3], pcd_cam)
        pcd_curr = pcd_cam + T_WC_np[:3, 3]

        return pcd_curr