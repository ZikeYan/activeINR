import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import numpy as np
import time
import imgviz

import open3d as o3d
from scipy.spatial import KDTree

from activeINR import geometry
from activeINR.datasets.dataloader import HabitatDataScene
from activeINR.datasets import image_transforms
import trimesh
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor


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

def process_data(obs, config):
    inv_depth_scale = 1. / config["dataset"]["depth_scale"]
    max_depth = config["sample"]["depth_range"][1]
    min_depth = config["sample"]["depth_range"][0]
    rgb_transform = None#transforms.Compose(
            #[image_transforms.BGRtoRGB()])
    depth_transform = transforms.Compose(
        [image_transforms.DepthScale(inv_depth_scale),
            image_transforms.DepthFilter(min_depth, max_depth)])
    assert obs is not None
    image = obs[0]
    depth = obs[1]
    pose_position = obs[2]
    pose_rotation = obs[3]
    import quaternion
    r = quaternion.as_rotation_matrix(pose_rotation)
    #r = R.from_quat(quaternion.as_float_array(rotation))
    Twc = np.eye(4)
    Twc[:3, :3] = r#.as_matrix()   #  np.identity(3)
    Twc[:3, 3] = pose_position# np.array([0, 0, 0])
    rTl = np.eye(4)
    rTl[1,1] = -1
    rTl[2,2] = -1
    Twc_rl = rTl@Twc@rTl

    if rgb_transform:
        image = rgb_transform(image)
    if depth_transform:
        depth = depth_transform(depth)

    sample = {
        "image": image,
        "depth": depth,
        "T": Twc_rl,
    }

    return sample

def compute_current_pcd(obs, config):
    reduce_factor_curr = 2
    H = config["dataset"]["height"]
    W = config["dataset"]["width"]
    hfov = config["dataset"]["hfov"] * np.pi / 180.
    fx = 0.5 * W / np.tan(hfov / 2.)
    fy = 0.5 * H / np.tan(hfov / 2.)
    cx = W / 2 - 1
    cy = H / 2 - 1
    H_vis_curr = H // reduce_factor_curr
    W_vis_curr = W // reduce_factor_curr
    fx_vis_curr = fx / reduce_factor_curr
    fy_vis_curr = fy / reduce_factor_curr
    cx_vis_curr = cx / reduce_factor_curr
    cy_vis_curr = cy / reduce_factor_curr
    data = process_data(obs, config)
    depth = data['depth']
    T_WC_np = data['T']
    depth_resize = imgviz.resize(
        depth, width=W_vis_curr,
        height=H_vis_curr,
        interpolation="nearest")
    pcd_cam = geometry.transform.pointcloud_from_depth(depth_resize, fx_vis_curr, fy_vis_curr, cx_vis_curr, cy_vis_curr)
    pcd_cam = pcd_cam.reshape(-1, 3).astype(np.float32)
    pcd_cam = np.einsum('ij,kj->ki', T_WC_np[:3, :3], pcd_cam)
    pcd_curr = pcd_cam + T_WC_np[:3, 3]

    return pcd_curr

def eval_function(step:int, image:np.ndarray, depth:np.ndarray, pose_position:np.ndarray, pose_rotation:np.ndarray, config:dict, gt_pc_tri:trimesh.PointCloud):
    pcd = compute_current_pcd([image, depth, pose_position, pose_rotation], config)

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
    print(f"Step {step}:", test1.shape, test2.shape, distance.shape)
    print(f"Step {step}:", "instant ", accuracy, completion, ratio)
    return distance

class EvaWindow:

    def __init__(self, trainer, explorer, mapper, action_file):
        self.trainer = trainer
        self.explorer = explorer
        self.mapper = mapper
        self.step = 0

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            trainer.W, trainer.H, trainer.fx, trainer.fy, trainer.cx, trainer.cy)

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

        output_dir = os.path.dirname(self.action_file)
        print(output_dir)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        f_error = open(output_dir + "/action_error.txt", "w")

        gt_mesh_tri = trimesh.Trimesh(np.asarray(self.gt_mesh.vertices), np.asarray(self.gt_mesh.triangles))

        gt_samples = trimesh.sample.sample_surface(gt_mesh_tri, 200000)
        gt_pc_tri = trimesh.PointCloud(vertices=gt_samples[0])
        min_distance = np.ones(gt_samples[0].shape[0])

        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            my_futures = []
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
                
                my_futures.append(executor.submit(eval_function, self.step, image, depth, pose_position, pose_rotation, self.trainer.config, gt_pc_tri))

                act = int(actions[self.step])
                #act = np.random.randint(0, 4)

                self.step += 1

                #action_space = ["STOP","MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
                if (act in [1,2,3]):
                    env.step(act)
                else:
                    print("Action out of the action space: ", act)
                    act = 0

                if self.step == self.trainer.n_steps:
                    print("End of the action file")
                    break
                
            for my_future in my_futures:
                distance = my_future.result()
                min_distance[min_distance>distance] = distance[min_distance>distance]
                min_comp = np.mean(min_distance)
                min_comp_ratio = np.mean((min_distance < 0.05).astype(np.float))
                print("updated", min_comp, min_comp_ratio)
                f_error.write(f"{min_comp}  {min_comp_ratio}\n")
                
            f_error.close()
            print(f"Use time {time.time() - t0} s")