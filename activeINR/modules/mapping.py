import numpy as np
import matplotlib.pylab as plt
import torch

import torch.optim as optim
from torchvision import transforms
import time
import trimesh
import imgviz
import json
import cv2
import copy
import os
from scipy import ndimage
from scipy.spatial import KDTree
from copy import deepcopy
from torch.autograd import Variable

from activeINR.datasets import (
    dataset, image_transforms, sdf_util, data_util
)
from activeINR.datasets.data_util import FrameData
from activeINR.modules import (
    fc_map, embedding, render, sample, loss
)
from activeINR import geometry, visualisation
from activeINR.visualisation import draw3D

def start_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end


def end_timing(start, end):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time

def mapper(trainer, t, iter, obs=None):
    # get/add data---------------------------------------------------------
    new_kf = None
    end = False
    finish_optim = trainer.steps_since_frame == trainer.optim_frames
    if obs is not None:  # and (finish_optim or t == 0):
        if t == 0:
            add_new_frame = True
        else:
            add_new_frame = trainer.check_keyframe_latest()
        if add_new_frame:
            new_frame_id = trainer.get_latest_frame_id()
            size_dataset = len(trainer.scene_dataset)
            if new_frame_id > size_dataset:
                end = True
                print("**************************************",
                      "End of sequence",
                      "**************************************")
                exit()
            else:
                # print("Total step time", trainer.tot_step_time)
                #print("frame______________________", new_frame_id)

                frame_data = trainer.get_data([new_frame_id], obs)
                trainer.add_frame(frame_data)
                if t == 0:
                    trainer.last_is_keyframe = True
                    trainer.optim_frames = 200
        if trainer.last_is_keyframe:
            new_kf = trainer.frames.im_batch_np[-1]
            h = int(new_kf.shape[0] / 6)
            w = int(new_kf.shape[1] / 6)
            new_kf = cv2.resize(new_kf, (w, h))
            iter = trainer.optim_frames
        else:
            new_kf = None

    # print(iter)
    # optimisation step---------------------------------------------
    for i in range(iter):
        losses, step_time = trainer.step()
        status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
        status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
        trainer.iter += 1

    return status, new_kf, end



class Trainer():
    def __init__(
        self,
        device,
        config_file,
        ckpt_load_file=None,
        incremental=True,
        grid_dim=100,
        scene_id = None
    ):
        super(Trainer, self).__init__()

        self.device = device
        self.incremental = incremental
        self.tot_step_time = 0.
        self.last_is_keyframe = False
        self.steps_since_frame = 0
        self.optim_frames = 0

        self.gt_depth_vis = None
        self.gt_im_vis = None

        # eval params
        self.gt_sdf_interp = None
        self.stage_sdf_interp = None
        self.sdf_dims = None
        self.sdf_transform = None

        self.grid_dim = grid_dim
        self.new_grid_dim = None
        self.chunk_size = 100000
        self.num_perturb = 20

        self.iter = 0
        self.frame_id = 0
        self.below_th_prop = 0

        with open(config_file) as json_file:
            self.config = json.load(json_file)

        if scene_id is not None:
            self.config["dataset"]["scenes_list"] = scene_id

        self.frames = FrameData()  # keyframes

        self.set_params()
        self.set_cam()
        self.set_directions()
        self.load_data()

        # scene params for visualisation
        self.scene_center = None
        self.inv_bounds_transform = None
        self.active_idxs = None
        self.active_pixels = None
        if self.gt_scene:
            scene_mesh = trimesh.exchange.load.load(
                self.scene_file, process=False)
            T_rot = np.eye(4)
            T_rot[1, 1], T_rot[2, 2] = 0, 0
            T_rot[1, 2], T_rot[2, 1] = -1, 1
            T_rot[1, 3] = 1.25  # not sure
            scene_mesh.apply_transform(T_rot)
            self.set_scene_properties(scene_mesh)
        if self.dataset_format == "realsense_franka_offline":
            self.set_scene_properties()

        self.load_networks()
        if ckpt_load_file is not None:
            self.load_checkpoint(ckpt_load_file)
        self.sdf_map.train()
        self.cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # Init functions ---------------------------------------

    def get_latest_frame_id(self):
        return self.frame_id#int(self.tot_step_time * self.fps)

    def set_scene_properties(self, scene_mesh = None):
        T_extent_to_scene, bounds_extents = \
                trimesh.bounds.oriented_bounds(scene_mesh)
        self.scene_center = scene_mesh.bounds.mean(axis=0)

        self.inv_bounds_transform = torch.from_numpy(
            T_extent_to_scene).float().to(self.device)
        self.bounds_transform_np = np.linalg.inv(T_extent_to_scene)
        self.bounds_transform = torch.from_numpy(
            self.bounds_transform_np).float().to(self.device)

        # Need to divide by range_dist as it will scale the grid which
        # is created in range = [-1, 1]
        # Also divide by 0.9 so extents are a bit larger than gt mesh
        grid_range = [-1.0, 1.0]
        range_dist = grid_range[1] - grid_range[0]
        self.scene_scale_np = bounds_extents / (range_dist * 0.9)
        self.scene_scale = torch.from_numpy(
            self.scene_scale_np).float().to(self.device)
        self.inv_scene_scale = 1. / self.scene_scale

        self.grid_pc = geometry.transform.make_3D_grid(
            grid_range,
            self.grid_dim,
            self.device,
            transform=self.bounds_transform,
            scale=self.scene_scale,
        )
        self.grid_pc = self.grid_pc.view(-1, 3).to(self.device)

        self.up_ix = np.argmax(np.abs(np.matmul(
            self.up, self.bounds_transform_np[:3, :3])))
        self.grid_up = self.bounds_transform_np[:3, self.up_ix]
        self.up_aligned = np.dot(self.grid_up, self.up) > 0

        self.crop_dist = 0.1 if "franka" in self.dataset_format else 0.25

    def set_params(self):
        # Dataset
        self.dataset_format = self.config["dataset"]["format"]
        self.ext_calib = None
        self.inv_depth_scale = 1. / self.config["dataset"]["depth_scale"]
        self.distortion_coeffs = []
        self.H = self.config["dataset"]["height"]
        self.W = self.config["dataset"]["width"]
        self.hfov = self.config["dataset"]["hfov"] * np.pi / 180.
        self.fx = 0.5 * self.W / np.tan(self.hfov / 2.)
        self.fy = 0.5 * self.H / np.tan(self.hfov / 2.)
        self.cx = self.W / 2 - 1
        self.cy = self.H / 2 - 1

        self.gt_scene = False
        if "gt_sdf_dir" in self.config["dataset"]:
            self.gt_scene = True
            root = self.config["dataset"]["root"]
            scene_list = self.config["dataset"]["scenes_list"]
            if self.dataset_format == "replica":
                self.scene_file = root + scene_list + "/mesh.ply"
            elif self.dataset_format == "mp3d":
                self.scene_file = root + "/v1/tasks/" + scene_list + '/' + scene_list + '_semantic.ply'
            elif self.dataset_format == "gibson":
                self.scene_file = root + scene_list + ".glb"
                #self.scene_file = root + "../../gibson_v2_selected/" + scene_list + '/' + scene_list + '_mesh.ply'
        #print(self.gt_scene)

        self.fps = 30  # this can be set to anything when in live mode
        self.obj_bounds_file = None
        self.noisy_depth = False
        if "noisy_depth" in self.config["dataset"]:
            self.noisy_depth = bool(self.config["dataset"]["noisy_depth"])
        self.gt_traj = None
        self.n_steps = self.config["trainer"]["steps"]

        # Model
        self.do_active = bool(self.config["model"]["do_active"])
        # scaling applied to network output before interpreting value as sdf
        self.scale_output = self.config["model"]["scale_output"]
        # noise applied to network output
        self.noise_std = self.config["model"]["noise_std"]
        self.noise_kf = self.config["model"]["noise_kf"]
        self.noise_frame = self.config["model"]["noise_frame"]
        # sliding window size for optimising latest frames
        self.window_size = self.config["model"]["window_size"]
        self.hidden_layers_block = self.config["model"]["hidden_layers_block"]
        self.hidden_feature_size = self.config["model"]["hidden_feature_size"]
        # multiplier for time spent doing training vs time elapsed
        # to simulate scenarios with e.g. 50% perception time, 50% planning
        self.frac_time_perception = \
            self.config["model"]["frac_time_perception"]
        # optimisation steps per kf
        self.iters_per_kf = self.config["model"]["iters_per_kf"]
        self.iters_per_frame = self.config["model"]["iters_per_frame"]
        # thresholds for adding frame to keyframe set
        self.kf_dist_th = self.config["model"]["kf_dist_th"]
        self.kf_pixel_ratio = self.config["model"]["kf_pixel_ratio"]

        embed_config = self.config["model"]["embedding"]
        # scaling applied to coords before embedding
        self.scale_input = embed_config["scale_input"]
        self.n_embed_funcs = embed_config["n_embed_funcs"]
        # boolean to use gaussian embedding
        self.gauss_embed = bool(embed_config["gauss_embed"])
        self.gauss_embed_std = embed_config["gauss_embed_std"]
        self.optim_embedding = bool(embed_config["optim_embedding"])

        # save
        self.save_period = self.config["save"]["save_period"]
        self.save_times = np.arange(
            self.save_period, 2000, self.save_period).tolist()
        self.save_checkpoints = bool(self.config["save"]["save_checkpoints"])
        self.save_slices = bool(self.config["save"]["save_slices"])
        self.save_meshes = bool(self.config["save"]["save_meshes"])
        self.log_dir = self.config["save"]["log_dir"]

        # Loss
        self.bounds_method = self.config["loss"]["bounds_method"]
        assert self.bounds_method in ["ray", "normal", "pc"]
        self.loss_type = self.config["loss"]["loss_type"]
        assert self.loss_type in ["L1", "L2"]
        self.trunc_weight = self.config["loss"]["trunc_weight"]
        # distance at which losses transition (see paper for details)
        self.trunc_distance = self.config["loss"]["trunc_distance"]
        self.eik_weight = self.config["loss"]["eik_weight"]
        # where to apply the eikonal loss
        self.eik_apply_dist = self.config["loss"]["eik_apply_dist"]
        self.grad_weight = self.config["loss"]["grad_weight"]
        self.orien_loss = bool(self.config["loss"]["orien_loss"])

        self.do_normal = False
        if self.bounds_method == "normal" or self.grad_weight != 0:
            self.do_normal = True

        # optimiser
        self.learning_rate = self.config["optimiser"]["lr"]
        self.weight_decay = self.config["optimiser"]["weight_decay"]

        # Sampling
        self.max_depth = self.config["sample"]["depth_range"][1]
        self.min_depth = self.config["sample"]["depth_range"][0]
        self.dist_behind_surf = self.config["sample"]["dist_behind_surf"]
        self.n_rays = self.config["sample"]["n_rays"]
        self.n_rays_is_kf = self.config["sample"]["n_rays_is_kf"]
        # num stratified samples per ray
        self.n_strat_samples = self.config["sample"]["n_strat_samples"]
        # num surface samples per ray
        self.n_surf_samples = self.config["sample"]["n_surf_samples"]

    def set_cam(self):
        reduce_factor = 4
        self.H_vis = self.H // reduce_factor
        self.W_vis = self.W // reduce_factor
        self.fx_vis = self.fx / reduce_factor
        self.fy_vis = self.fy / reduce_factor
        self.cx_vis = self.cx / reduce_factor
        self.cy_vis = self.cy / reduce_factor

        reduce_factor_up = 4
        self.H_vis_up = self.H // reduce_factor_up
        self.W_vis_up = self.W // reduce_factor_up
        self.fx_vis_up = self.fx / reduce_factor_up
        self.fy_vis_up = self.fy / reduce_factor_up
        self.cx_vis_up = self.cx / reduce_factor_up
        self.cy_vis_up = self.cy / reduce_factor_up

        reduce_factor_curr = 2
        self.H_vis_curr = self.H // reduce_factor_curr
        self.W_vis_curr = self.W // reduce_factor_curr
        self.fx_vis_curr = self.fx / reduce_factor_curr
        self.fy_vis_curr = self.fy / reduce_factor_curr
        self.cx_vis_curr = self.cx / reduce_factor_curr
        self.cy_vis_curr = self.cy / reduce_factor_curr

        self.loss_approx_factor = 8
        w_block = self.W // self.loss_approx_factor
        h_block = self.H // self.loss_approx_factor
        increments_w = torch.arange(
            self.loss_approx_factor, device=self.device) * w_block
        increments_h = torch.arange(
            self.loss_approx_factor, device=self.device) * h_block
        c, r = torch.meshgrid(increments_w, increments_h)
        c, r = c.t(), r.t()
        self.increments_single = torch.stack((r, c), dim=2).view(-1, 2)

    def set_directions(self):
        self.dirs_C = geometry.transform.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        self.dirs_C_vis = geometry.transform.ray_dirs_C(
            1,
            self.H_vis,
            self.W_vis,
            self.fx_vis,
            self.fy_vis,
            self.cx_vis,
            self.cy_vis,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

        self.dirs_C_vis_up = geometry.transform.ray_dirs_C(
            1,
            self.H_vis_up,
            self.W_vis_up,
            self.fx_vis_up,
            self.fy_vis_up,
            self.cx_vis_up,
            self.cy_vis_up,
            self.device,
            depth_type="z",
        ).view(1, -1, 3)

    def load_networks(self):
        #positional_encoding = embedding.FFPositionalEncoding()
        positional_encoding = embedding.PostionalEncoding(
            min_deg=0,
            max_deg=self.n_embed_funcs,
            scale=self.scale_input,
            transform=self.inv_bounds_transform,
        )

        self.sdf_map = fc_map.SDFMap(
            positional_encoding,
            hidden_size=self.hidden_feature_size,
            hidden_layers_block=self.hidden_layers_block,
            scale_output=self.scale_output,
        ).to(self.device)

        self.optimiser = optim.AdamW(
            self.sdf_map.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def load_checkpoint(self, checkpoint_load_file):
        checkpoint = torch.load(checkpoint_load_file)
        self.sdf_map.load_state_dict(checkpoint["model_state_dict"])
        # self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_gt_sdf(self):
        sdf_grid = np.load(self.gt_sdf_file)
        if self.dataset_format == "ScanNet":
            sdf_grid = np.abs(sdf_grid)
        self.sdf_transform = np.loadtxt(self.sdf_transf_file)
        self.gt_sdf_interp = sdf_util.sdf_interpolator(
            sdf_grid, self.sdf_transform)
        self.sdf_dims = torch.tensor(sdf_grid.shape)

    # Data methods ---------------------------------------

    def load_data(self):

        rgb_transform = None#transforms.Compose(
            #[image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(self.inv_depth_scale),
             image_transforms.DepthFilter(self.min_depth,self.max_depth)])

        noisy_depth = self.noisy_depth
        dataset_class = dataset.HabitatSimulator
        self.up = np.array([1., 0., 0.])
        self.traj_file = None
        camera_matrix = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

        self.scene_dataset = dataset_class(
            rgb_transform=rgb_transform,
            depth_transform=depth_transform,
            noisy_depth=noisy_depth,
            max_length=self.n_steps
        )

    def get_data(self, idxs, obs):
        frames_data = FrameData()
        sample = self.scene_dataset.process(obs)

        im_np = sample["image"][None, ...]
        depth_np = sample["depth"][None, ...]
        T_np = sample["T"][None, ...]

        #im = torch.from_numpy(im_np).float().to(self.device) / 255.
        depth = torch.from_numpy(depth_np).float().to(self.device)
        T = torch.from_numpy(T_np).float().to(self.device)

        data = FrameData(
            frame_id=np.array([idxs]),
            #im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
        )
        if self.do_normal:
            pc = geometry.transform.pointcloud_from_depth_torch(
                depth[0], self.fx, self.fy, self.cx, self.cy)
            normals = geometry.transform.estimate_pointcloud_normals(pc)
            data.normal_batch = normals[None, :]
        if self.gt_traj is not None:
            data.T_WC_gt = self.gt_traj[idxs][None, ...]

        frames_data.add_frame_data(data, replace=False)

        return frames_data

    def add_data(self, data, replace=False):
        # if last frame isn't a keyframe then the new frame
        # replaces last frame in batch.
        replace = self.last_is_keyframe is False
        self.frames.add_frame_data(data, replace)
        #
        # if self.last_is_keyframe:
        #     print("New keyframe. KF ids:", self.frames.frame_id[:-1])

    def add_frame(self, frame_data):
        if self.last_is_keyframe:
            self.frozen_sdf_map = copy.deepcopy(self.sdf_map)
        self.add_data(frame_data)
        self.steps_since_frame = 0
        self.last_is_keyframe = False
        self.optim_frames = self.iters_per_frame
        self.noise_std = self.noise_frame

    # Keyframe methods ----------------------------------

    def is_keyframe(self, T_WC, depth_gt):
        sample_pts = self.sample_points(
            depth_gt, T_WC, n_rays=self.n_rays_is_kf, dist_behind_surf=0.8)

        pc = sample_pts["pc"]
        z_vals = sample_pts["z_vals"]
        depth_sample = sample_pts["depth_sample"]
        # if self.last_is_keyframe:
        #     self.frozen_sdf_map = copy.deepcopy(self.sdf_map)
        with torch.set_grad_enabled(False):
            sdf = self.frozen_sdf_map(pc, noise_std=self.noise_std)

        z_vals, ind1 = z_vals.sort(dim=-1)
        ind0 = torch.arange(z_vals.shape[0])[:, None].repeat(
            1, z_vals.shape[1])
        sdf = sdf[ind0, ind1]

        view_depth = render.sdf_render_depth(z_vals, sdf)

        loss = torch.abs(view_depth - depth_sample) / depth_sample

        below_th = loss < self.kf_dist_th
        size_loss = below_th.shape[0]
        below_th_prop = below_th.sum().float() / size_loss
        self.below_th_prop = below_th_prop
        is_keyframe = below_th_prop.item() < self.kf_pixel_ratio
        # if is_keyframe:
        #     print(
        #         "The loss for the KF is ",
        #         below_th_prop.item(),
        #         ", less than the threshold ",
        #         self.kf_pixel_ratio,
        #         # " ---> is keyframe:",
        #         # is_keyframe
        #     )

        return is_keyframe

    def check_keyframe_latest(self):
        """
        check if current frame is keyframe and store for the next iter
        """
        add_new_frame = False
        if self.last_is_keyframe:
            add_new_frame = True
        else:
            T_WC = self.frames.T_WC_batch[-1].unsqueeze(0)
            depth_gt = self.frames.depth_batch[-1].unsqueeze(0)
            self.last_is_keyframe = self.is_keyframe(T_WC, depth_gt)

        if len(self.frames.frame_id) > 1:
            if self.frames.frame_id[-1] - self.frames.frame_id[-2] > 100. and (
                self.get_latest_frame_id() < len(self.scene_dataset)):
                #print("More than 100 frames since last kf, so add new")
                self.last_is_keyframe = True

        # time_since_kf = self.tot_step_time - self.frames.frame_id[-2] / 30.
        # if time_since_kf > 15. and not self.live and (self.get_latest_frame_id()< len(self.scene_dataset)):
        #     print("More than 15 seconds since last kf, so add new")
        #     add_new_frame = True

        if self.last_is_keyframe:
            self.optim_frames = self.iters_per_kf
            self.noise_std = self.noise_kf
        else:
            add_new_frame = True

        return add_new_frame

    def select_keyframes(self):
        """
        Use most recent two keyframes then fill rest of window
        based on loss distribution across the remaining keyframes.
        """
        n_frames = len(self.frames)
        limit = n_frames - 2
        denom = self.frames.frame_avg_losses[:-2].sum()
        loss_dist = self.frames.frame_avg_losses[:-2] / denom
        loss_dist_np = loss_dist.cpu().numpy()

        select_size = self.window_size - 2

        rand_ints = np.random.choice(
            np.arange(0, limit),
            size=select_size,
            replace=False,
            p=loss_dist_np)

        last = n_frames - 1
        idxs = [*rand_ints, last - 1, last]

        return idxs

    def clear_keyframes(self):
        self.frames = FrameData()  # keyframes
        self.gt_depth_vis = None
        self.gt_im_vis = None

    # Main training methods ----------------------------------

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
        norm_batch=None,
        active_loss_approx=None,
        n_rays=None,
        dist_behind_surf=None,
        n_strat_samples=None,
        n_surf_samples=None,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """
        if n_rays is None:
            n_rays = self.n_rays
        if dist_behind_surf is None:
            dist_behind_surf = self.dist_behind_surf
        if n_strat_samples is None:
            n_strat_samples = self.n_strat_samples
        if n_surf_samples is None:
            n_surf_samples = self.n_surf_samples

        n_frames = depth_batch.shape[0]
        if active_loss_approx is None:
            indices_b, indices_h, indices_w = sample.sample_pixels(
                n_rays, n_frames, self.H, self.W, device=self.device)
        else:
            # indices_b, indices_h, indices_w = \
            #     active_sample.active_sample_pixels(
            #         n_rays, n_frames, self.H, self.W, device=self.device,
            #         loss_approx=active_loss_approx,
            #         increments_single=self.increments_single
            #     )
            raise Exception('Active sampling not currently supported.')

        get_masks = active_loss_approx is None

        (
            dirs_C_sample,
            depth_sample,
            norm_sample,
            T_WC_sample,
            binary_masks,
            indices_b,
            indices_h,
            indices_w
        ) = sample.get_batch_data(
            depth_batch,
            T_WC_batch,
            self.dirs_C,
            indices_b,
            indices_h,
            indices_w,
            norm_batch=norm_batch,
            get_masks=get_masks,
        )

        max_depth = depth_sample + dist_behind_surf
        pc, z_vals = sample.sample_along_rays(
            T_WC_sample,
            self.min_depth,
            max_depth,
            n_strat_samples,
            n_surf_samples,
            dirs_C_sample,
            gt_depth=depth_sample,
            grad=False,
        )

        sample_pts = {
            "depth_batch": depth_batch,
            "pc": pc,
            "z_vals": z_vals,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "dirs_C_sample": dirs_C_sample,
            "depth_sample": depth_sample,
            "T_WC_sample": T_WC_sample,
            "norm_sample": norm_sample,
            "binary_masks": binary_masks,
        }
        return sample_pts

    def sdf_eval_and_loss(
        self,
        sample,
        do_avg_loss=True,
    ):
        pc = sample["pc"]
        z_vals = sample["z_vals"]
        indices_b = sample["indices_b"]
        indices_h = sample["indices_h"]
        indices_w = sample["indices_w"]
        dirs_C_sample = sample["dirs_C_sample"]
        depth_sample = sample["depth_sample"]
        T_WC_sample = sample["T_WC_sample"]
        norm_sample = sample["norm_sample"]
        binary_masks = sample["binary_masks"]
        depth_batch = sample["depth_batch"]

        do_sdf_grad = self.eik_weight != 0 or self.grad_weight != 0
        if do_sdf_grad:
            pc.requires_grad_()

        sdf = self.sdf_map(pc, noise_std=self.noise_std)

        sdf_grad = None
        if do_sdf_grad:
            sdf_grad = fc_map.gradient(pc, sdf)


        # compute bounds

        bounds, grad_vec = loss.bounds(
            self.bounds_method,
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            z_vals,
            pc,
            self.trunc_distance,
            norm_sample,
            do_grad=True,
        )

        # compute loss

        sdf_loss_mat, free_space_ixs = loss.sdf_loss(
            sdf, bounds, self.trunc_distance, loss_type=self.loss_type)

        eik_loss_mat = None
        if self.eik_weight != 0:
            eik_loss_mat = torch.abs(sdf_grad.norm(2, dim=-1) - 1)

        grad_loss_mat = None
        if self.grad_weight != 0:
            pred_norms = sdf_grad[:, 0]
            surf_loss_mat = 1 - self.cosSim(pred_norms, norm_sample)

            grad_vec[torch.where(grad_vec[..., 0].isnan())] = \
                norm_sample[torch.where(grad_vec[..., 0].isnan())[0]]
            grad_loss_mat = 1 - self.cosSim(grad_vec, sdf_grad[:, 1:])
            grad_loss_mat = torch.cat(
                (surf_loss_mat[:, None], grad_loss_mat), dim=1)

            if self.orien_loss:
                grad_loss_mat = (grad_loss_mat > 1).float()

        total_loss, total_loss_mat, losses = loss.tot_loss(
            sdf_loss_mat, grad_loss_mat, eik_loss_mat,
            free_space_ixs, bounds, self.eik_apply_dist,
            self.trunc_weight, self.grad_weight, self.eik_weight,
        )

        loss_approx, frame_avg_loss = None, None
        if do_avg_loss:
            loss_approx, frame_avg_loss = loss.frame_avg(
                total_loss_mat, depth_batch, indices_b, indices_h, indices_w,
                self.W, self.H, self.loss_approx_factor, binary_masks)

        # # # for plot
        # z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
        # ray_target = depth_sample[:, None] - z_vals
        # ray_target = z_to_euclidean_depth[:, None] * ray_target

        # # apply correction based on angle between ray and normal
        # costheta = torch.abs(self.cosSim(-dirs_C_sample, norm_sample))
        # # only apply correction out to truncation distance
        # sub = self.trunc_distance * (1. - costheta)
        # normal_target_fs = ray_target - sub[:, None]
        # # target_normal = ray_target.clone()
        # target_normal = normal_target_fs
        # ixs = target_normal < self.trunc_distance
        # target_normal[ixs] = (ray_target * costheta[:, None])[ixs]

        # self.check_gt_sdf(
        #     depth_sample, z_vals, dirs_C_sample, pc,
        #     ray_target, target_sdf, target_normal)

        return (
            total_loss,
            losses,
            loss_approx,
            frame_avg_loss,
        )

    def check_gt_sdf(self, depth_sample, z_vals, dirs_C_sample, pc,
                     target_ray, target_pc, target_normal):
                     # origins, dirs_W):
        # reorder in increasing z vals
        z_vals, indices = z_vals.sort(dim=-1)
        row_ixs = torch.arange(pc.shape[0])[:, None].repeat(1, pc.shape[1])

        pc = pc[row_ixs, indices]
        target_ray = target_ray[row_ixs, indices]
        target_pc = target_pc[row_ixs, indices]
        if target_normal is not None:
            target_normal = target_normal[row_ixs, indices]

        z2euc_sample = torch.norm(dirs_C_sample, dim=-1)
        z_vals = z_vals * z2euc_sample[:, None]

        scene = trimesh.Scene(trimesh.load(self.scene_file))

        with torch.set_grad_enabled(False):
            j = 0
            fig, ax = plt.subplots(3, 1, figsize=(11, 10))

            for i in [9, 19, 23]:  # range(0, 100):
                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp,
                    pc[i].reshape(-1, 3).detach().cpu().numpy(),
                    handle_oob='fill', oob_val=np.nan)

                x = z_vals[i].cpu()
                lw = 2.5
                ax[j].hlines(0, x[0], x[-1], color="gray", linestyle="--")
                ax[j].plot(
                    x, gt_sdf, label="True signed distance",
                    color="C1", linewidth=lw
                )
                ax[j].plot(
                    x, target_ray[i].cpu(), label="Ray",
                    color="C3", linewidth=lw
                )
                if target_normal is not None:
                    ax[j].plot(
                        x, target_normal[i].cpu(), label="Normal",
                        color="C2", linewidth=lw
                    )
                ax[j].plot(
                    x, target_pc[i].cpu(), label="Batch distance",
                    color="C0", linewidth=lw
                )

                # print("diffs", target_sdf[i].cpu() - gt_sdf)
                if j == 2:
                    ax[j].set_xlabel("Distance along ray, d [m]", fontsize=24)
                    ax[j].set_yticks([0, 4, 8])
                # ax[j].set_ylabel("Signed distance (m)", fontsize=21)
                ax[j].tick_params(axis='both', which='major', labelsize=24)
                # ax[j].set_xticks(fontsize=20)
                # ax[j].set_yticks(fontsize=20)
                # if j == 0:
                #     ax[j].legend(fontsize=20)
                j += 1

            fig.text(
                0.05, 0.5, 'Signed distance [m]',
                va='center', rotation='vertical', fontsize=24
            )
            # plt.tight_layout()
            plt.show()

            # intersection = dirs_W[i] * depth_sample[i] + origins[i]
            # int_pc = trimesh.PointCloud(
            #     intersection[None, :].cpu(), [255, 0, 0, 255])
            # scene.add_geometry(int_pc)

            # pts = pc[i].detach().cpu().numpy()
            # colormap_fn = sdf_util.get_colormap()
            # col = colormap_fn.to_rgba(gt_sdf, alpha=1., bytes=False)
            # tm_pc = trimesh.PointCloud(pts, col)
            # scene.add_geometry(tm_pc)

        scene.show()

    def step(self):
        start, end = start_timing()

        depth_batch = self.frames.depth_batch
        T_WC_batch = self.frames.T_WC_batch
        norm_batch = self.frames.normal_batch if self.do_normal else None

        if len(self.frames) > self.window_size and self.incremental:
            idxs = self.select_keyframes()
            # print("selected frame ids", self.frames.frame_id[idxs[:-1]])
        else:
            idxs = np.arange(T_WC_batch.shape[0])
        self.active_idxs = idxs

        depth_select = depth_batch[idxs]
        T_WC_select = T_WC_batch[idxs]

        sample_pts = self.sample_points(
            depth_select, T_WC_select, norm_batch=norm_batch)
        self.active_pixels = {
            'indices_b': sample_pts['indices_b'],
            'indices_h': sample_pts['indices_h'],
            'indices_w': sample_pts['indices_w'],
        }

        total_loss, losses, active_loss_approx, frame_avg_loss = \
            self.sdf_eval_and_loss(sample_pts, do_avg_loss=True)

        self.frames.frame_avg_losses[idxs] = frame_avg_loss

        total_loss.backward()
        self.optimiser.step()
        for param_group in self.optimiser.param_groups:
            params = param_group["params"]
            for param in params:
                param.grad = None

        # if self.do_active:
        #     sample_pts = self.sample_points(
        #         depth_select, T_WC_select, norm_batch=norm_batch,
        #         active_loss_approx=active_loss_approx)

        #     loss_active, _, _, _ = \
        #         self.sdf_eval_and_loss(sample_pts, do_avg_loss=False)

        #     loss_active.backward()
        #     self.optimiser.step()
        #     for param_group in self.optimiser.param_groups:
        #         params = param_group["params"]
        #         for param in params:
        #             param.grad = None

            loss_approx = active_loss_approx[-1].detach().cpu().numpy()
            loss_approx_viz = imgviz.depth2rgb(loss_approx)
            loss_approx_viz = cv2.cvtColor(loss_approx_viz, cv2.COLOR_RGB2BGR)
            loss_approx_viz = cv2.resize(loss_approx_viz, (200, 200),
                                         interpolation=cv2.INTER_NEAREST)
            #cv2.imshow("loss_approx_viz", loss_approx_viz)
            #cv2.waitKey(1)

        step_time = end_timing(start, end)
        time_s = step_time / 1000.
        self.tot_step_time += (1 / self.frac_time_perception) * time_s
        self.steps_since_frame += 1

        return losses, step_time

    # Visualisation methods -----------------------------------

    def update_vis_vars(self):
        depth_batch_np = self.frames.depth_batch_np
        im_batch_np = self.frames.im_batch_np

        if self.gt_depth_vis is None:
            updates = depth_batch_np.shape[0]
        else:
            diff_size = depth_batch_np.shape[0] - \
                self.gt_depth_vis.shape[0]
            updates = diff_size + 1

        for i in range(updates, 0, -1):
            prev_depth_gt = depth_batch_np[-i]
            prev_im_gt = im_batch_np[-i]
            prev_depth_gt_resize = imgviz.resize(
                prev_depth_gt, width=self.W_vis,
                height=self.H_vis,
                interpolation="nearest")[None, ...]
            prev_im_gt_resize = imgviz.resize(
                prev_im_gt, width=self.W_vis,
                height=self.H_vis)[None, ...]

            replace = False
            if i == updates:
                replace = True

            self.gt_depth_vis = data_util.expand_data(
                self.gt_depth_vis,
                prev_depth_gt_resize,
                replace=replace)
            self.gt_im_vis = data_util.expand_data(
                self.gt_im_vis,
                prev_im_gt_resize,
                replace=replace)

    def latest_frame_vis(self, do_render=True, obs=[]):
        start, end = start_timing()
        # get latest frame from camera
        data = self.scene_dataset.process(obs)
        image = data['image']
        depth = data['depth']
        T_WC_np = data['T']

        w = self.W_vis_up * 2
        h = self.H_vis_up * 2
        image = cv2.resize(image, (w, h))
        depth = cv2.resize(depth, (w, h))
        depth_viz = imgviz.depth2rgb(
            depth, min_value=self.min_depth, max_value=self.max_depth)
        # depth_viz[depth == 0] = [0, 255, 0]

        rgbd_vis = np.hstack((image, depth_viz))

        if not do_render:
            return rgbd_vis, None, T_WC_np
        else:
            T_WC = torch.FloatTensor(T_WC_np).to(self.device)[None, ...]

            with torch.set_grad_enabled(False):
                # efficient depth and normals render
                # valid_depth = depth != 0.0
                # depth_sample = torch.FloatTensor(depth).to(self.device)[valid_depth]
                # max_depth = depth_sample + 0.5  # larger max depth for depth render
                # dirs_C = self.dirs_C_vis[0, valid_depth.flatten()]

                pc, z_vals = sample.sample_along_rays(
                    T_WC,
                    self.min_depth,
                    self.max_depth,
                    n_stratified_samples=20,
                    n_surf_samples=0,
                    dirs_C=self.dirs_C_vis,
                    gt_depth=None,  # depth_sample
                )

                sdf = self.sdf_map(pc)
                # sdf = fc_map.chunks(pc, self.chunk_size, self.sdf_map)
                depth_vals_vis = render.sdf_render_depth(z_vals, sdf)

                depth_up = torch.nn.functional.interpolate(
                    depth_vals_vis.view(1, 1, self.H_vis, self.W_vis),
                    size=[self.H_vis_up, self.W_vis_up],
                    mode='bilinear', align_corners=True
                )
                depth_up = depth_up.view(-1)

                pc_up, z_vals_up = sample.sample_along_rays(
                    T_WC,
                    depth_up - 0.1,
                    depth_up + 0.1,
                    n_stratified_samples=12,
                    n_surf_samples=12,
                    dirs_C=self.dirs_C_vis_up,
                )
                sdf_up = self.sdf_map(pc_up)
                depth_vals = render.sdf_render_depth(z_vals_up, sdf_up)

            surf_normals_C = render.render_normals(
                T_WC, depth_vals[None, ...], self.sdf_map, self.dirs_C_vis_up)

            # render_depth = torch.zeros(self.H_vis, self.W_vis)
            # render_depth[valid_depth] = depth_vals.detach().cpu()
            # render_depth = render_depth.numpy()
            render_depth = depth_vals.view(self.H_vis_up, self.W_vis_up).cpu().numpy()
            render_depth_viz = imgviz.depth2rgb(
                render_depth, min_value=self.min_depth, max_value=self.max_depth)

            surf_normals_C = (- surf_normals_C + 1.0) / 2.0
            surf_normals_C = torch.clip(surf_normals_C, 0., 1.)
            # normals_viz = torch.zeros(self.H_vis, self.W_vis, 3)
            # normals_viz[valid_depth] = surf_normals_C.detach().cpu()
            normals_viz = surf_normals_C.view(self.H_vis_up, self.W_vis_up, 3).detach().cpu()
            normals_viz = (normals_viz.numpy() * 255).astype(np.uint8)

            render_vis = np.hstack((normals_viz, render_depth_viz))
            w_up = int(render_vis.shape[1] * 2)
            h_up = int(render_vis.shape[0] * 2)
            render_vis = cv2.resize(render_vis, (w_up, h_up))

            elapsed = end_timing(start, end)
            #print("Time for depth and normal render", elapsed)
            return rgbd_vis, render_vis, T_WC_np

    def keyframe_vis(self, reduce_factor=2):
        start, end = start_timing()

        h, w = self.frames.im_batch_np.shape[1:3]
        h = int(h / reduce_factor)
        w = int(w / reduce_factor)

        kf_vis = []
        for i, kf in enumerate(self.frames.im_batch_np):
            kf = cv2.resize(kf, (w, h))
            kf = cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)

            pad_color = [255, 255, 255]
            if self.active_idxs is not None and self.active_pixels is not None:
                if i in self.active_idxs:
                    pad_color = [0, 0, 139]

                    # show sampled pixels
                    act_inds_mask = self.active_pixels['indices_b'] == i
                    h_inds = self.active_pixels['indices_h'][act_inds_mask]
                    w_inds = self.active_pixels['indices_w'][act_inds_mask]
                    mask = np.zeros([self.H, self.W])
                    mask[h_inds.cpu().numpy(), w_inds.cpu().numpy()] = 1
                    mask = ndimage.binary_dilation(mask, iterations=6)
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (w, h)).astype(np.bool)
                    kf[mask, :] = [0, 0, 139]

            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=pad_color)
            kf = cv2.copyMakeBorder(
                kf, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            kf_vis.append(kf)

        kf_vis = np.hstack(kf_vis)
        elapsed = end_timing(start, end)
        print("Time for kf vis", elapsed)
        return kf_vis

    def frames_vis(self):
        view_depths = self.render_depth_vis()
        view_normals = self.render_normals_vis(view_depths)
        view_depths = view_depths.cpu().numpy()
        view_normals = view_normals.detach().cpu().numpy()
        gt_depth_ims = self.gt_depth_vis
        im_batch_np = self.gt_im_vis

        views = []
        for batch_i in range(len(self.frames)):
            depth = view_depths[batch_i]
            depth_viz = imgviz.depth2rgb(
                depth, min_value=self.min_depth, max_value=self.max_depth)

            gt = gt_depth_ims[batch_i]
            gt_depth = imgviz.depth2rgb(
                gt, min_value=self.min_depth, max_value=self.max_depth)

            loss = np.abs(gt - depth)
            loss[gt == 0] = 0
            loss_viz = imgviz.depth2rgb(loss)

            normals = view_normals[batch_i]
            normals = (- normals + 1.0) / 2.0
            normals = np.clip(normals, 0., 1.)
            normals = (normals * 255).astype(np.uint8)

            visualisations = [gt_depth, depth_viz, loss_viz, normals]
            if im_batch_np is not None:
                visualisations.append(im_batch_np[batch_i])

            viz = np.vstack(visualisations)
            views.append(viz)

        viz = np.hstack(views)
        return viz

    def render_depth_vis(self):
        view_depths = []

        depth_gt = self.frames.depth_batch_np
        T_WC_batch = self.frames.T_WC_batch
        if self.frames.T_WC_track:
            T_WC_batch = self.frames.T_WC_track

        with torch.set_grad_enabled(False):
            for batch_i in range(len(self.frames)):  # loops through frames
                T_WC = T_WC_batch[batch_i].unsqueeze(0)

                depth_sample = depth_gt[batch_i]
                depth_sample = cv2.resize(
                    depth_sample, (self.W_vis, self.H_vis))
                depth_sample = torch.FloatTensor(depth_sample).to(self.device)

                # larger max depth for depth render
                max_depth = (depth_sample + 0.8).flatten()
                pc, z_vals = sample.sample_along_rays(
                    T_WC,
                    self.min_depth,
                    max_depth,
                    self.n_strat_samples,
                    n_surf_samples=0,
                    dirs_C=self.dirs_C_vis[0],
                    gt_depth=None,
                    grad=False,
                )

                sdf = self.sdf_map(pc)

                view_depth = render.sdf_render_depth(z_vals, sdf)
                view_depth = view_depth.view(self.H_vis, self.W_vis)
                view_depths.append(view_depth)

            view_depths = torch.stack(view_depths)
        return view_depths

    def render_normals_vis(self, view_depths):
        view_normals = []

        T_WC_batch = self.frames.T_WC_batch
        if self.frames.T_WC_track:
            T_WC_batch = self.frames.T_WC_track

        for batch_i in range(len(self.frames)):  # loops through frames
            T_WC = T_WC_batch[batch_i].unsqueeze(0)
            view_depth = view_depths[batch_i]

            surf_normals_C = render.render_normals(
                T_WC, view_depth, self.sdf_map, self.dirs_C_vis[0])
            view_normals.append(surf_normals_C)

        view_normals = torch.stack(view_normals)
        return view_normals

    def draw_3D(
        self,
        show_pc=False,
        show_mesh=False,
        draw_cameras=False,
        show_gt_mesh=False,
        camera_view=True,
    ):  
        start, end = start_timing()

        scene = trimesh.Scene()
        scene.set_camera()
        scene.camera.focal = (self.fx, self.fy)
        scene.camera.resolution = (self.W, self.H)

        T_WC_np = self.frames.T_WC_batch_np
        if self.frames.T_WC_track:
            T_WC_np = self.frames.T_WC_track.cpu().numpy()

        if draw_cameras:
            n_frames = len(self.frames)
            cam_scale = 0.25 if "franka" in self.dataset_format else 1.0
            draw3D.draw_cams(
                n_frames, T_WC_np, scene, color=(0.0, 1.0, 0.0, 1.0), cam_scale = cam_scale)

            if self.frames.T_WC_gt:  # show gt and input poses too
                draw3D.draw_cams(
                    n_frames, self.frames.T_WC_gt, scene,
                    color=(1.0, 0.0, 1.0, 0.8), cam_scale = cam_scale)
                draw3D.draw_cams(
                    n_frames, self.frames.T_WC_batch_np, scene,
                    color=(1., 0., 0., 0.8), cam_scale = cam_scale)

            trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
            if self.frames.T_WC_gt is not None:
                trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
            visualisation.draw3D.draw_trajectory(
                trajectory_gt, scene, color=(1.0, 0.0, 0.0)
            )

        if show_pc:
            if self.gt_depth_vis is None:
                self.update_vis_vars()  # called in self.mesh_rec
            pcs_cam = geometry.transform.backproject_pointclouds(
                self.gt_depth_vis, self.fx_vis, self.fy_vis,
                self.cx_vis, self.cy_vis)
            pc_w, colors = draw3D.draw_pc(
                n_frames,
                pcs_cam,
                T_WC_np,
                self.gt_im_vis,
            )
            pc = trimesh.PointCloud(pc_w, colors=colors)
            scene.add_geometry(pc, geom_name='depth_pc')

        if show_mesh:
            try:
                sdf_mesh = self.mesh_rec()
                scene.add_geometry(sdf_mesh, geom_name="rec_mesh")
            except ValueError: # ValueError: Surface level must be within volume data range.
                print("ValueError: Surface level must be within volume data range.")
                pass

        if show_gt_mesh:
            gt_mesh = trimesh.load(self.scene_file)
            gt_mesh.visual.material.image.putalpha(50)
            scene.add_geometry(gt_mesh)

        if not camera_view and self.scene_center is not None:
            if "realsense_franka" in self.dataset_format:
                cam_pos = self.scene_center + self.up * 1 + np.array([1, -1, 0.])
            else:
                cam_pos = self.scene_center + self.up * 12 + np.array([3., 0., 0.])
            R, t = geometry.transform.look_at(
                cam_pos, self.scene_center, -self.up)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            scene.camera_transform = geometry.transform.to_trimesh(T)
        else:
            view_idx = -1
            scene.camera_transform = geometry.transform.to_trimesh(
                T_WC_np[view_idx])
            scene.camera_transform = (
                scene.camera_transform
                @ trimesh.transformations.translation_matrix([0, 0, 0.1]))

        elapsed = end_timing(start, end)
        print(f'Time to draw scene: {elapsed}ms')
        return scene

    def draw_obj_3D(self, show_gt_mesh=True):
        if self.obj_bounds_file is not None:

            scene = trimesh.Scene()
            scene.set_camera()
            scene.camera.focal = (self.fx, self.fy)
            scene.camera.resolution = (self.W, self.H)

            if show_gt_mesh:
                gt_mesh = trimesh.load(self.scene_file)
                gt_mesh.visual.material.image.putalpha(50)

            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, self.up_ix,
                expand_m=0.2, expand_down=True)

            for i, bounds in enumerate(obj_bounds):

                x = torch.linspace(bounds[0, 0], bounds[1, 0], 128)
                y = torch.linspace(bounds[0, 1], bounds[1, 1], 128)
                z = torch.linspace(bounds[0, 2], bounds[1, 2], 128)
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3)
                pc = pc.view(-1, 3).to(self.device)

                with torch.set_grad_enabled(False):
                    sdf = fc_map.chunks(
                        pc, self.chunk_size, self.sdf_map,
                        # surf_dists=gt_dist,
                    )
                T = np.eye(4)
                T[:3, 3] = bounds[0] + 0.5 * (bounds[1] - bounds[0])
                sdf = sdf.view(128, 128, 128)
                obj_mesh = draw3D.draw_mesh(
                    sdf, 0.5 * (bounds[1] - bounds[0]), T)
                obj_mesh.visual.face_colors = [160, 160, 160, 160]
                scene.add_geometry(obj_mesh)

                if show_gt_mesh:
                    box = trimesh.primitives.Box(
                        extents=bounds[1] - bounds[0], transform=T)
                    crop = gt_mesh.slice_plane(
                        box.facets_origin, -box.facets_normal)
                    crop.visual.face_colors = [0, 160, 50, 160]
                    scene.add_geometry(crop)

            scene.set_camera()

            return scene
        return None

    def get_sdf_grid(self, k=5):
        with torch.set_grad_enabled(False):

            # gt_dist = sdf_util.eval_sdf_interp(
            #     self.gt_sdf_interp, self.grid_pc.cpu().numpy(),
            #     handle_oob='fill')
            # gt_dist = torch.FloatTensor(gt_dist).to(self.device)

            sdf = fc_map.chunks(
                self.grid_pc,
                self.chunk_size,
                self.sdf_map,
                # surf_dists=gt_dist,
            )

            # p_sdf = torch.stack([fc_map.chunks(
            #     self.grid_pc,
            #     self.chunk_size,
            #     self.sdf_map,
            #     # surf_dists=gt_dist,
            # ) for i in range(k)], dim=1)
            # sdf = p_sdf.mean(axis=1)
            #print("mean sdf")
            dim = self.grid_dim
            sdf = sdf.view(dim, dim, dim)

        return sdf

    def get_topdown_sdf(self, grid, k=5):
        pc = grid.view(-1, 3).to(self.device)
        with torch.set_grad_enabled(False):
            sdf = fc_map.chunks(
                pc,
                self.chunk_size,
                self.sdf_map,
                # surf_dists=gt_dist,
            )
            # p_sdf = torch.stack([fc_map.chunks(
            #     pc,
            #     self.chunk_size,
            #     self.sdf_map,
            #     # surf_dists=gt_dist,
            # ) for i in range(k)], dim=1)
            # sdf = p_sdf.mean(axis=1)
            #print("mean sdf topdown")

            sdf = sdf.view(grid.size()[:2])

        return sdf

    def get_sdf(self, pts, k=5):
        with torch.set_grad_enabled(False):
            p_sdf = torch.stack([fc_map.chunks(
                pts,
                self.chunk_size,
                self.sdf_map,
                # surf_dists=gt_dist,
            ) for i in range(k)], dim=1)
            sdf = p_sdf.mean(axis=1)
        return sdf

    def get_sdf_grid_pc(self, include_gt=False, mask_near_pc=False):
        sdf_grid = self.get_sdf_grid()
        grid_pc = self.grid_pc.reshape(
            self.grid_dim, self.grid_dim, self.grid_dim, 3)
        sdf_grid_pc = torch.cat((grid_pc, sdf_grid[..., None]), dim=-1)
        sdf_grid_pc = sdf_grid_pc.detach().cpu().numpy()

        if include_gt and self.gt_sdf_interp is not None:
            self.gt_sdf_interp.bounds_error = False
            self.gt_sdf_interp.fill_value = 0.0
            gt_sdf = self.gt_sdf_interp(self.grid_pc.cpu())
            gt_sdf = gt_sdf.reshape(
                self.grid_dim, self.grid_dim, self.grid_dim)
            sdf_grid_pc = np.concatenate(
                (sdf_grid_pc, gt_sdf[..., None]), axis=-1)
            self.gt_sdf_interp.bounds_error = True

        keep_mask = None
        if mask_near_pc:
            self.update_vis_vars()
            pcs_cam = geometry.transform.backproject_pointclouds(
                self.gt_depth_vis, self.fx_vis, self.fy_vis,
                self.cx_vis, self.cy_vis)
            pc, _ = draw3D.draw_pc(
                len(self.frames),
                pcs_cam,
                self.frames.T_WC_batch_np,
            )
            tree = KDTree(pc)
            sparse_grid = sdf_grid_pc[::10, ::10, ::10, :3]
            dists, _ = tree.query(sparse_grid.reshape(-1, 3), k=1)
            dists = dists.reshape(sparse_grid.shape[:-1])
            keep_mask = dists < self.crop_dist
            keep_mask = keep_mask.repeat(10, axis=0).repeat(10, axis=1).repeat(10, axis=2)

        return sdf_grid_pc, keep_mask

    def param_norm(self, params):
        return np.sqrt(sum([p.data.pow(2).sum().item() for p in params]))

    def unit_params(self, like):
        new_params = [Variable(p.data.new(*p.size()).normal_(), requires_grad=True) for p in like]
        norm = self.param_norm(new_params)
        for p in new_params:
            p.data.div_(norm)
        return new_params

    def vector_aligned_unit_params(self, like):
        new_params = [Variable(p.data.clone(), requires_grad=True) for p in like]
        norm = self.param_norm(new_params)
        for p in new_params:
            p.data.div_(norm)
        return new_params

    def axis_aligned_unit_params(self, like):
        new_params = [Variable(p.data.new(*p.size()).zero_(), requires_grad=True) for p in like]
        # Choose a random weight matrix
        rand_ps = [p for p in new_params if p.dim() == 2]
        rand_p = rand_ps[np.random.randint(len(rand_ps))]
        # Light one up at a random position
        y_idx = np.random.randint(rand_p.size(0))
        x_idx = np.random.randint(rand_p.size(1))
        rand_p.data[y_idx, x_idx] = 1
        return new_params

    def set_net_params(self, model, params, inplace=False):
        if inplace:
            model_cp = model
        else:
            model_cp = deepcopy(model)
        for p, new_p in zip(model_cp.parameters(), params):
            p.data.copy_(new_p.data)
        return model_cp

    def scale_params(self, params, scale, inplace=False):
        if not inplace:
            params = deepcopy(params)
        for p in params:
            p.data.mul_(scale)
        return params

    def sum_params(self, param1, param2, inplace=False):
        if not inplace:
            params = deepcopy(param1)
        else:
            params = param1
        for p, p2 in zip(params, param2):
            p.data.add_(p2.data)
        return params

    def subtract_params(self, param1, param2, inplace=False):
        if not inplace:
            params = deepcopy(param1)
        else:
            params = param1
        for p, p2 in zip(params, param2):
            p.data.sub_(p2.data)
        return params

    def scale_params_(self, params, scale):
        return self.scale_params(params, scale, True)

    def vis_perturb(self, pts, last_net_params, to_cpu=True):
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        ori_sdf = fc_map.chunks(
            pts,
            self.chunk_size,
            self.sdf_map,
            to_cpu=to_cpu
            # surf_dists=gt_dist,
        )

        ori_sdf = np.array(np.hstack(ori_sdf))
        with torch.set_grad_enabled(False):
            curr_net_params = deepcopy(self.sdf_map.parameters)
            param_diff = self.subtract_params(list(last_net_params()), list(curr_net_params()), inplace=False)
            model_diff = self.set_net_params(self.sdf_map, param_diff, inplace=False)
            samples = []
            for it in range(self.num_perturb):
                rand_unit_params = self.unit_params(model_diff.parameters())
                scaled_params = self.scale_params_(rand_unit_params, 1)
                perturbed_params = self.sum_params(list(curr_net_params()), scaled_params, inplace=False)
                new_model = self.set_net_params(self.sdf_map, perturbed_params, inplace=False)
                sdf_perturb = fc_map.chunks(
                    pts,
                    self.chunk_size,
                    new_model,
                    to_cpu=to_cpu
                    # surf_dists=gt_dist,
                )
                samples.append(np.hstack(sdf_perturb))
            samples = np.array(samples)
            samples -= ori_sdf
            diff = np.abs(samples)
            diff_mean = diff.mean(0)
            diff_var = samples.var(0)
            torch.cuda.empty_cache()
        return diff_mean, diff_var

    def vis_accuracy(self, pts, to_cpu=True):
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        ori_sdf = fc_map.chunks(
            pts,
            self.chunk_size,
            self.sdf_map,
            to_cpu=to_cpu
            # surf_dists=gt_dist,
        )

        ori_sdf = np.array(np.hstack(ori_sdf))
        return ori_sdf

    def eval_perturb(self, pts, last_net_params):
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        ori_sdf = fc_map.chunks(
            pts,
            self.chunk_size,
            self.sdf_map,
            # surf_dists=gt_dist,
        )
        gradient = fc_map.gradient(pts, ori_sdf).detach().cpu().numpy()
        with torch.set_grad_enabled(False):
            curr_net_params = deepcopy(self.sdf_map.parameters)
            param_diff = self.subtract_params(list(last_net_params()), list(curr_net_params()), inplace=False)
            model_diff = self.set_net_params(self.sdf_map, param_diff, inplace=False)
            samples = []
            for it in range(self.num_perturb):
                rand_unit_params = self.unit_params(model_diff.parameters())
                scaled_params = self.scale_params_(rand_unit_params, 1)
                perturbed_params = self.sum_params(list(curr_net_params()), scaled_params, inplace=False)
                new_model = self.set_net_params(self.sdf_map, perturbed_params, inplace=False)
                sdf_perturb = fc_map.chunks(
                    pts,
                    self.chunk_size,
                    new_model
                    # surf_dists=gt_dist,
                ).detach().cpu().numpy()
                samples.append(sdf_perturb)
            samples = np.array(samples)
            samples -= ori_sdf.detach().cpu().numpy()
            diff = np.abs(samples)
            diff_mean = diff.mean(0)
            diff_var = samples.var(0)
            torch.cuda.empty_cache()
        return gradient, diff_mean, diff_var

    def view_sdf(self):
        show_mesh = False if self.gt_scene else True
        scene = self.draw_3D(
            show_pc=True,
            show_mesh=show_mesh,
            draw_cameras=True,
            show_gt_mesh=self.gt_scene,
            camera_view=True,
        )
        sdf_grid_pc, _ = self.get_sdf_grid_pc(include_gt=False)
        sdf_grid_pc = np.transpose(sdf_grid_pc, (2, 1, 0, 3))
        # sdf_grid_pc = sdf_grid_pc[:, :, ::-1]  # for replica
        visualisation.sdf_viewer.SDFViewer(
            scene=scene, sdf_grid_pc=sdf_grid_pc,
            colormap=True, surface_cutoff=0.01
        )

    def mesh_rec(self, crop_mesh_with_pc=True, color_type=0):
        """
        Generate mesh reconstruction.
        """
        self.update_vis_vars()
        pcs_cam = geometry.transform.backproject_pointclouds(
            self.gt_depth_vis, self.fx_vis, self.fy_vis,
            self.cx_vis, self.cy_vis)
        pc, _ = draw3D.draw_pc(
            len(self.frames),
            pcs_cam,
            self.frames.T_WC_batch_np,
        )

        if self.gt_scene is False:
            pc_tm = trimesh.PointCloud(pc)
            self.set_scene_properties(pc_tm)

        sdf = self.get_sdf_grid()

        sdf_mesh = draw3D.draw_mesh(
            sdf,
            self.scene_scale_np,
            self.bounds_transform_np,
            color_by=color_type
        )

        if crop_mesh_with_pc:
            tree = KDTree(pc)
            dists, _ = tree.query(sdf_mesh.vertices, k=1)
            keep_ixs = dists < self.crop_dist
            face_mask = keep_ixs[sdf_mesh.faces].any(axis=1)
            sdf_mesh.update_faces(face_mask)
            sdf_mesh.remove_unreferenced_vertices()
            # sdf_mesh.visual.vertex_colors[~keep_ixs, 3] = 10

        if self.new_grid_dim is not None:
            self.grid_dim = self.new_grid_dim
            self.grid_pc = self.new_grid_pc
            self.new_grid_dim = None
            self.new_grid_pc = None

        return sdf_mesh

    def write_mesh(self, filename, im_pose=None):
        mesh = self.mesh_rec()

        data = trimesh.exchange.ply.export_ply(mesh)
        out = open(filename, "wb+")
        out.write(data)
        out.close()

        if im_pose is not None:
            scene = trimesh.Scene(mesh)
            im = draw3D.capture_scene_im(
                scene, im_pose, tm_pose=True)
            cv2.imwrite(filename[:-4] + ".png", im[..., :3][..., ::-1])

    def compute_slices(
        self, z_ixs=None, n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        # Compute points to query
        if z_ixs is None:
            z_ixs = torch.linspace(30, self.grid_dim - 30, n_slices)
            z_ixs = torch.round(z_ixs).long()
        z_ixs = z_ixs.to(self.device)

        pc = self.grid_pc.reshape(
            self.grid_dim, self.grid_dim, self.grid_dim, 3)
        pc = torch.index_select(pc, self.up_ix, z_ixs)

        if not self.up_aligned:
            indices = np.arange(len(z_ixs))[::-1]
            indices = torch.from_numpy(indices.copy()).to(self.device)
            pc = torch.index_select(pc, self.up_ix, indices)

        cmap = sdf_util.get_colormap(sdf_range=sdf_range)
        grid_shape = pc.shape[:-1]
        n_slices = grid_shape[self.up_ix]
        pc = pc.reshape(-1, 3)

        scales = torch.cat([
            self.scene_scale[:self.up_ix], self.scene_scale[self.up_ix + 1:]])
        im_size = 256 * scales / scales.min()
        im_size = im_size.int().cpu().numpy()

        slices = {}

        with torch.set_grad_enabled(False):
            sdf = fc_map.chunks(pc, self.chunk_size, self.sdf_map)
            sdf = sdf.detach().cpu().numpy()
        sdf_viz = cmap.to_rgba(sdf.flatten(), alpha=1., bytes=False)
        sdf_viz = (sdf_viz * 255).astype(np.uint8)[..., :3]
        sdf_viz = sdf_viz.reshape(*grid_shape, 3)
        sdf_viz = [
            cv2.resize(np.take(sdf_viz, i, self.up_ix), im_size[::-1])
            for i in range(n_slices)
        ]
        slices["pred_sdf"] = sdf_viz

        if include_chomp:
            cost = metrics.chomp_cost(sdf, epsilon=2.)
            cost_viz = imgviz.depth2rgb(
                cost.reshape(self.grid_dim, -1), min_value=0., max_value=1.5)
            cost_viz = cost_viz.reshape(*grid_shape, 3)
            cost_viz = [
                cv2.resize(np.take(cost_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["pred_cost"] = cost_viz

        pc = pc.reshape(*grid_shape, 3)
        pc = pc.detach().cpu().numpy()

        if include_gt:
            gt_sdf = sdf_util.eval_sdf_interp(
                self.gt_sdf_interp, pc, handle_oob='fill')
            gt_sdf_viz = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
            gt_sdf_viz = gt_sdf_viz.reshape(*grid_shape, 4)
            gt_sdf_viz = (gt_sdf_viz * 255).astype(np.uint8)[..., :3]
            gt_sdf_viz = [
                cv2.resize(np.take(gt_sdf_viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["gt_sdf"] = gt_sdf_viz

            if include_chomp:
                gt_costs = metrics.chomp_cost(gt_sdf, epsilon=2.)
                gt_cost_viz = imgviz.depth2rgb(
                    gt_costs.reshape(self.grid_dim, -1),
                    min_value=0., max_value=1.5)
                gt_cost_viz = gt_cost_viz.reshape(*grid_shape, 3)
                gt_cost_viz = [
                    cv2.resize(
                        np.take(gt_cost_viz, i, self.up_ix), im_size[::-1])
                    for i in range(n_slices)
                ]
                slices["gt_cost"] = gt_cost_viz

        if include_diff:
            sdf = sdf.reshape(*grid_shape)
            diff = np.abs(gt_sdf - sdf)
            diff = diff.reshape(self.grid_dim, -1)
            diff_viz = imgviz.depth2rgb(diff, min_value=0., max_value=0.5)
            diff_viz = diff_viz.reshape(-1, 3)
            viz = np.full(diff_viz.shape, 255, dtype=np.uint8)

            viz = viz.reshape(*grid_shape, 3)
            viz = [
                cv2.resize(np.take(viz, i, self.up_ix), im_size[::-1])
                for i in range(n_slices)
            ]
            slices["diff"] = viz

        if draw_cams:  # Compute camera markers
            cam_xyz = self.frames.T_WC_batch[:, :3, 3].cpu()
            cam_td = self.to_topdown(cam_xyz, im_size)

            cam_rots = self.frames.T_WC_batch[:, :3, :3].cpu().numpy()
            angs = []
            for rot in cam_rots:
                ang = np.arctan2(rot[0, 2], rot[0, 0])
                # y = - np.sign(range_dim0) * rot[axis_dim0, 2]
                # x = - np.sign(range_dim1) * rot[axis_dim1, 2]
                # ang = np.arctan2(x, y)
                angs.append(ang)

            # Add cam markers to predicted sdf slices
            for i, im in enumerate(slices["pred_sdf"]):
                if self.incremental:
                    trajectory_gt = self.frames.T_WC_batch_np[:, :3, 3]
                    if self.frames.T_WC_gt is not None:
                        trajectory_gt = self.frames.T_WC_gt[:, :3, 3]
                    traj_td = self.to_topdown(trajectory_gt, im_size)
                    for j in range(len(traj_td) - 1):
                        if not (traj_td[j] == traj_td[j + 1]).all():
                            im = im.astype(np.uint8) / 255
                            im = cv2.line(
                                im,
                                traj_td[j][::-1],
                                traj_td[j + 1][::-1],
                                [1., 0., 0.], 2)
                            im = (im * 255).astype(np.uint8)
                slices["pred_sdf"][i] = im

        return slices

    def write_slices(
        self, save_path, prefix="", n_slices=6,
        include_gt=False, include_diff=False, include_chomp=False,
        draw_cams=False, sdf_range=[-2, 2],
    ):
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=include_gt,
            include_diff=include_diff,
            include_chomp=include_chomp,
            draw_cams=draw_cams,
            sdf_range=sdf_range,
        )

        for s in range(n_slices):
            cv2.imwrite(
                os.path.join(save_path, prefix + f"pred_{s}.png"),
                slices["pred_sdf"][s][..., ::-1])
            if include_gt:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_{s}.png"),
                    slices["gt_sdf"][s][..., ::-1])
            if include_diff:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"diff_{s}.png"),
                    slices["diff"][s][..., ::-1])
            if include_chomp:
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"pred_cost_{s}.png"),
                    slices["pred_cost"][s][..., ::-1])
                cv2.imwrite(
                    os.path.join(save_path, prefix + f"gt_cost_{s}.png"),
                    slices["gt_cost"][s][..., ::-1])

    def slices_vis(self, n_slices=6):
        slices = self.compute_slices(
            z_ixs=None,
            n_slices=n_slices,
            include_gt=True,
            include_diff=True,
            include_chomp=False,
            draw_cams=True,
        )

        gt_sdf = np.hstack((slices["gt_sdf"]))
        pred_sdf = np.hstack((slices["pred_sdf"]))
        diff = np.hstack((slices["diff"]))

        viz = np.vstack((gt_sdf, pred_sdf, diff))
        return viz

    def to_topdown(self, pts, im_size):
        cam_homog = np.concatenate(
            [pts, np.ones([pts.shape[0], 1])], axis=-1)
        inv_bt = np.linalg.inv(self.bounds_transform_np)
        cam_td = np.matmul(cam_homog, inv_bt.T)
        cam_td = cam_td[:, :3] / self.scene_scale.cpu().numpy()
        cam_td = cam_td / 2 + 0.5  # [-1, 1] -> [0, 1]
        cam_td = np.concatenate((
            cam_td[:, :self.up_ix], cam_td[:, self.up_ix + 1:]), axis=1)
        cam_td = cam_td * im_size
        cam_td = cam_td.astype(int)

        return cam_td

    def obj_slices_vis(self, n_slices=6):
        if self.obj_bounds_file is not None:
            up_ix = 1
            obj_bounds = metrics.get_obj_eval_bounds(
                self.obj_bounds_file, up_ix)

            cmap = sdf_util.get_colormap(sdf_range=[-0.5, 0.5])
            all_slices = []

            for bounds in obj_bounds:
                dims = [256, 256, 256]
                dims[up_ix] = n_slices
                x = torch.linspace(bounds[0, 0], bounds[1, 0], dims[0])
                y = torch.linspace(bounds[0, 1], bounds[1, 1], dims[1])
                z = torch.linspace(bounds[0, 2], bounds[1, 2], dims[2])
                xx, yy, zz = torch.meshgrid(x, y, z)
                pc = torch.cat(
                    (xx[..., None], yy[..., None], zz[..., None]), dim=3
                ).to(self.device)

                sdf = self.sdf_map(pc)
                col = cmap.to_rgba(
                    sdf.detach().cpu().numpy().flatten(),
                    alpha=1., bytes=False)
                col = (col * 255).astype(np.uint8)[..., :3]
                col = col.reshape(*pc.shape[:-1], 3)
                col = np.hstack([col[:, i] for i in range(n_slices)])

                gt_sdf = sdf_util.eval_sdf_interp(
                    self.gt_sdf_interp, pc.cpu(), handle_oob='fill')
                gt_col = cmap.to_rgba(gt_sdf.flatten(), alpha=1., bytes=False)
                gt_col = gt_col.reshape(*pc.shape[:-1], 4)
                gt_col = (gt_col * 255).astype(np.uint8)[..., :3]
                gt_col = np.hstack([gt_col[:, i] for i in range(n_slices)])

                slices = np.vstack((col, gt_col))

                all_slices.append(slices)

            return np.vstack((all_slices))
        return None

    def sdf_fn(self, pts):
        with torch.set_grad_enabled(False):
            pts = torch.FloatTensor(pts).to(self.device)
            sdf = self.sdf_map(pts)
        return sdf.detach().cpu().numpy()

    def grad_fn(self, pts):
        pts = torch.FloatTensor(pts).to(self.device)
        pts.requires_grad_()
        sdf = self.sdf_map(pts)
        sdf_grad = fc_map.gradient(pts, sdf)

        return sdf_grad.detach().cpu().numpy()