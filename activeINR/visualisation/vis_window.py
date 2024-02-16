import numpy as np
import threading
import time
import imgviz
import cv2
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


class VisWindow:

    def __init__(self, trainer, explorer, mapper, font_id):
        self.trainer = trainer
        self.explorer = explorer
        self.mapper = mapper

        self.is_key_frame = False
        self.two_d_position = []

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            trainer.W, trainer.H, trainer.fx, trainer.fy, trainer.cx, trainer.cy)

        self.window = gui.Application.instance.create_window('Viewer', 2560, 1440)

        mode = "incremental"

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Application control
        button_play_pause = gui.ToggleSwitch('Resume/Pause')
        button_play_pause.set_on_clicked(self._on_switch)

        self.training_iters_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, em, 0))
        steps_label = gui.Label('Training iters per step')
        self.training_iters_slider = gui.Slider(gui.Slider.INT)
        self.training_iters_slider.set_limits(1, 50)
        self.training_iters_slider.int_value = 5
        self.training_iters_grid.add_child(steps_label)
        self.training_iters_grid.add_child(self.training_iters_slider)

        button_clear_kf = None
        button_clear_kf = gui.Button('Clear Keyframes')
        button_clear_kf.horizontal_padding_em = 0
        button_clear_kf.vertical_padding_em = 0.1
        button_clear_kf.set_on_clicked(self._clear_keyframes)

        self.button_compute_mesh = gui.Button('Recompute mesh')
        self.button_compute_mesh.set_on_clicked(self._recompute_mesh)
        self.button_compute_mesh.horizontal_padding_em = 0
        self.button_compute_mesh.vertical_padding_em = 0.1
        self.button_compute_mesh.enabled = True

        self.button_compute_slices = gui.Button('Recompute SDF slices')
        self.button_compute_slices.set_on_clicked(self._recompute_slices)
        self.button_compute_slices.horizontal_padding_em = 0
        self.button_compute_slices.vertical_padding_em = 0.1
        self.button_compute_slices.enabled = False

        self.button_compute_renders = gui.Button('Recompute renders')
        self.button_compute_renders.set_on_clicked(self._recompute_renders)
        self.button_compute_renders.horizontal_padding_em = 0
        self.button_compute_renders.vertical_padding_em = 0.1
        self.button_compute_renders.enabled = False

        ### Info panel
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = font_id

        ## Items in vis props
        self.vis_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
        height_label = gui.Label('Pruned height')
        self.height_slider = gui.Slider(gui.Slider.DOUBLE)
        self.height_slider.set_limits(-10.0, 10.0)
        self.height_slider.double_value = 0.8
        self.vis_prop_grid.add_child(height_label)
        self.vis_prop_grid.add_child(self.height_slider)

        mesh_label = gui.Label('Mesh reconstruction')
        self.mesh_box = gui.Checkbox('')
        self.mesh_box.checked = True
        self.mesh_box.set_on_checked(self._toggle_mesh)
        self.vis_prop_grid.add_child(mesh_label)
        self.vis_prop_grid.add_child(self.mesh_box)

        interval_label = gui.Label('    Meshing interval steps')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(5, 100)
        self.interval_slider.int_value = 20
        self.vis_prop_grid.add_child(interval_label)
        self.vis_prop_grid.add_child(self.interval_slider)

        voxel_dim_label = gui.Label('    Meshing voxel grid dim')
        self.voxel_dim_slider = gui.Slider(gui.Slider.INT)
        self.voxel_dim_slider.set_limits(20, 256)
        self.voxel_dim_slider.int_value = self.trainer.grid_dim
        self.voxel_dim_slider.set_on_value_changed(self._change_voxel_dim)
        self.vis_prop_grid.add_child(voxel_dim_label)
        self.vis_prop_grid.add_child(self.voxel_dim_slider)

        col_label = gui.Label('    Color types')
        self.col_slider = gui.Slider(gui.Slider.INT)
        self.col_slider.set_limits(0, 3)#white, height, normal, uncertainty
        self.col_slider.int_value = 3
        self.vis_prop_grid.add_child(col_label)
        self.vis_prop_grid.add_child(self.col_slider)

        crop_label = gui.Label('    Crop near point cloud')
        self.crop_box = gui.Checkbox('')
        if mode == 'batch':
            self.crop_box.checked = False
        else:
            self.crop_box.checked = True
        self.vis_prop_grid.add_child(crop_label)
        self.vis_prop_grid.add_child(self.crop_box)

        slices_label = gui.Label('SDF slices')
        self.slices_box = gui.Checkbox('')
        self.slices_box.checked = False
        self.slices_box.set_on_checked(self._toggle_slices)
        self.vis_prop_grid.add_child(slices_label)
        self.vis_prop_grid.add_child(self.slices_box)

        slices_interval_label = gui.Label('    Compute interval steps')
        self.slices_interval_slider = gui.Slider(gui.Slider.INT)
        self.slices_interval_slider.set_limits(20, 500)
        self.slices_interval_slider.int_value = 20
        self.vis_prop_grid.add_child(slices_interval_label)
        self.vis_prop_grid.add_child(self.slices_interval_slider)

        frontier_label = gui.Label('Frontiers')
        self.frontier_box = gui.Checkbox('')
        self.frontier_box.checked = True
        self.frontier_box.set_on_checked(self._toggle_frontier)
        self.vis_prop_grid.add_child(frontier_label)
        self.vis_prop_grid.add_child(self.frontier_box)

        vis_cluter_label = gui.Label('    Visualize frontier clusters')
        self.vis_cluster_box = gui.Checkbox('')
        self.vis_cluster_box.checked = True
        self.vis_prop_grid.add_child(vis_cluter_label)
        self.vis_prop_grid.add_child(self.vis_cluster_box)

        frontier_interval_label = gui.Label('    Compute interval steps')
        self.frontier_interval_slider = gui.Slider(gui.Slider.INT)
        self.frontier_interval_slider.set_limits(1, 20)
        self.frontier_interval_slider.int_value = 1
        self.vis_prop_grid.add_child(frontier_interval_label)
        self.vis_prop_grid.add_child(self.frontier_interval_slider)

        frontier_selection_label = gui.Label('    Selected frontier ratio')
        self.frontier_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
        self.frontier_ratio_slider.set_limits(0, 1.0)
        self.frontier_ratio_slider.double_value = 0.8
        self.vis_prop_grid.add_child(frontier_selection_label)
        self.vis_prop_grid.add_child(self.frontier_ratio_slider)

        cluster_pruning_label = gui.Label('    Pruned cluster ratio')
        self.cluster_ratio_slider = gui.Slider(gui.Slider.DOUBLE)
        self.cluster_ratio_slider.set_limits(0, 0.2)
        self.cluster_ratio_slider.double_value = 0.005
        self.vis_prop_grid.add_child(cluster_pruning_label)
        self.vis_prop_grid.add_child(self.cluster_ratio_slider)

        local_label = gui.Label('Local status')
        self.local_box = gui.Checkbox('')
        self.local_box.checked = True
        self.local_box.set_on_checked(self._toggle_local)
        self.vis_prop_grid.add_child(local_label)
        self.vis_prop_grid.add_child(self.local_box)

        curr_pcd_label = gui.Label('    Current observation')
        self.curr_pcd_box = gui.Checkbox('')
        self.curr_pcd_box.checked = True
        self.vis_prop_grid.add_child(curr_pcd_label)
        self.vis_prop_grid.add_child(self.curr_pcd_box)

        bbox_label = gui.Label('    Local horizon')
        self.bbox_box = gui.Checkbox('')
        self.bbox_box.checked = True
        self.bbox_box.set_on_checked(self._toggle_bbox)
        self.vis_prop_grid.add_child(bbox_label)
        self.vis_prop_grid.add_child(self.bbox_box)

        global_label = gui.Label('Global status')
        self.global_box = gui.Checkbox('')
        self.global_box.checked = True
        self.global_box.set_on_checked(self._toggle_global)
        self.vis_prop_grid.add_child(global_label)
        self.vis_prop_grid.add_child(self.global_box)

        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        origin_label = gui.Label('    Origin')
        self.origin_box = gui.Checkbox('')
        self.origin_box.checked = False
        self.vis_prop_grid.add_child(origin_label)
        self.vis_prop_grid.add_child(self.origin_box)

        traj_label = gui.Label('    Trajectory')
        self.traj_box = gui.Checkbox('')
        self.traj_box.checked = True
        self.traj_box.set_on_checked(self._toggle_trajectory)
        self.vis_prop_grid.add_child(traj_label)
        self.vis_prop_grid.add_child(self.traj_box)

        kf_label = gui.Label('    Keyframes')
        self.kf_box = gui.Checkbox('')
        self.kf_box.checked = True
        self.vis_prop_grid.add_child(kf_label)
        self.vis_prop_grid.add_child(self.kf_box)

        pc_label = gui.Label('    Stored point cloud')
        self.pc_box = gui.Checkbox('')
        self.pc_box.checked = False
        self.vis_prop_grid.add_child(pc_label)
        self.vis_prop_grid.add_child(self.pc_box)

        self.gt_mesh = None
        if self.trainer.gt_scene:
            gt_mesh_label = gui.Label('Ground truth mesh')
            self.gt_mesh_box = gui.Checkbox('')
            self.gt_mesh_box.checked = False
            self.vis_prop_grid.add_child(gt_mesh_label)
            self.vis_prop_grid.add_child(self.gt_mesh_box)

        set_enabled(self.vis_prop_grid, True)
        self.interval_slider.enabled = self.mesh_box.checked
        self.voxel_dim_slider.enabled = self.mesh_box.checked
        self.col_slider.enabled = self.mesh_box.checked
        self.slices_interval_slider.enabled = self.slices_box.checked
        self.frontier_interval_slider.enabled = self.frontier_box.checked
        self.frontier_ratio_slider.enabled = self.frontier_box.checked
        self.cluster_ratio_slider.enabled = self.frontier_box.checked
        self.bbox_box.enabled = self.local_box.checked

        self.panel.add_child(gui.Label(f'Operation mode: {mode}'))
        self.panel.add_child(button_play_pause)
        self.panel.add_child(self.training_iters_grid)
        if button_clear_kf is not None:
            self.panel.add_child(button_clear_kf)
        self.panel.add_child(self.button_compute_mesh)
        self.panel.add_child(self.button_compute_slices)
        self.panel.add_child(self.button_compute_renders)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(self.output_info)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('3D visualisation settings'))
        self.panel.add_child(self.vis_prop_grid)


        # 2D Visualization widget
        self.vis_panel = gui.Vert(spacing, margins)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), em, em)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_rgb_depth = gui.ImageWidget()
        self.render_normals_depth = gui.ImageWidget()
        tab1.add_child(gui.Label('Input rgb and depth'))
        tab1.add_child(self.input_rgb_depth)
        tab1.add_fixed(vspacing)

        black_vis = np.full(
            [self.trainer.H, 2 * self.trainer.W, 3], 0, dtype=np.uint8)
        self.no_render = o3d.geometry.Image(black_vis)
        render_label = gui.Label('Rendered normals and depth')
        self.render_box = gui.Checkbox('')
        self.render_box.checked = True
        self.render_box.set_on_checked(self._toggle_render)
        render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
        render_grid.add_child(render_label)
        render_grid.add_child(self.render_box)
        render_interval_label = gui.Label('    Render interval')
        self.render_interval_slider = gui.Slider(gui.Slider.INT)
        self.render_interval_slider.set_limits(5, 100)
        self.render_interval_slider.int_value = 20
        render_grid.add_child(render_interval_label)
        render_grid.add_child(self.render_interval_slider)
        tab1.add_child(render_grid)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.render_normals_depth)
        tab1.add_fixed(2*vspacing)
        self.topdown_map = gui.ImageWidget()
        tab1.add_child(gui.Label('Top down visualization'))
        tab1.add_child(self.topdown_map)
        tab1.add_fixed(2*vspacing)
        self.log_info = gui.Label('Log info')
        self.log_info.font_id = font_id
        tab1.add_child(self.log_info)
        tab1.add_fixed(2*vspacing)
        self.render_interval_slider.enabled = self.render_box.checked
        tabs.add_tab('Live visualization', tab1)

        ### Keyframes image tab
        tab2 = gui.Vert(0, tab_margins)
        self.n_panels = 10
        self.keyframe_panels = []
        for _ in range(self.n_panels):
            kf_panel = gui.ImageWidget()
            tab2.add_child(kf_panel)
            self.keyframe_panels.append(kf_panel)
        tabs.add_tab('Stored memories', tab2)

        self.vis_panel.add_child(tabs)
        # Scene widget
        self.widget3d = gui.SceneWidget()

        # timings panel
        self.timings_panel = gui.Vert(spacing, gui.Margins(em, 0.5 * em, em, em))
        self.timings = gui.Label('Compute balance in last 20s:')
        self.timings_panel.add_child(self.timings)

        # dialog panel
        self.dialog_panel = gui.Vert(spacing, gui.Margins(em, em, em, em))
        self.dialog = gui.Dialog("Tracking lost!")
        self.dialog.add_child(gui.Label('Tracking lost!'))
        self.dialog_panel.add_child(self.dialog)
        self.dialog_panel.visible = False

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.timings_panel)
        w.add_child(self.dialog_panel)
        w.add_child(self.vis_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])
        self.widget3d.scene.scene.set_sun_light([-0.2,1,0.2],[1,1,1], 70000)
        self.widget3d.scene.scene.enable_sun_light(True)

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = True
        self.is_running = True
        self.is_surface_updated = False
        if self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)

        self.kfs = []
        if mode == 'batch':
            h = int(trainer.frames.im_batch_np[-1].shape[0] / 6)
            w = int(trainer.frames.im_batch_np[-1].shape[1] / 6)
            self.kfs = [
                cv2.resize(kf, (w, h)) for kf in trainer.frames.im_batch_np
            ]
        self.max_points = 300000
        self.kf_panel_size = 3
        self.steps_before_meshing = 49
        self.clear_kf_frustums = False
        self.sdf_grid_pc = None
        self.slice_ix = 0
        self.slice_step = 1
        self.colormap_fn = None
        self.vis_times = []
        self.optim_times = []
        self.prepend_text = ""
        self.latest_mesh = None
        self.latest_pcd = None
        self.current_pcd = None
        self.local_bbox = None
        self.latest_frustums = []
        self.T_WC_latest = None
        self.frontier_col_margin = None
        self.trajectory = []

        self.lit_mat = rendering.MaterialRecord()
        self.lit_mat.shader = "defaultLit"
        self.unlit_mat = rendering.MaterialRecord()
        self.unlit_mat.shader = "unlitLine"
        self.unlit_mat.line_width = 5.0
        self.slim_unlit_mat = rendering.MaterialRecord()
        self.slim_unlit_mat.shader = "unlitLine"
        self.slim_unlit_mat.line_width = 2.0

        self.cam_scale = 0.1 if "franka" in trainer.dataset_format else 0.2

        self.l_policy = DdppoPolicy(path=self.explorer.config.local_policy_path)
        self.l_policy = self.l_policy.to("cuda")

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()

    def run_local_policy(self, depth, rho, phi, step, device):
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(device)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y, rect.get_right() - 2*panel_width, rect.height)

        timings_panel_width = 15 * em
        timings_panel_height = 3 * em
        self.timings_panel.frame = gui.Rect(
            rect.get_right() - timings_panel_width,
            rect.y,
            timings_panel_width,
            timings_panel_height
        )

        dialog_panel_width = 16 * em
        dialog_panel_height = 4 * em
        self.dialog_panel.frame = gui.Rect(
            rect.get_right() // 2 - dialog_panel_width + panel_width // 2,
            rect.get_bottom() // 2 - dialog_panel_height,
            dialog_panel_width,
            dialog_panel_height
        )
        self.vis_panel.frame = gui.Rect(self.widget3d.frame.get_right(), rect.y, panel_width, rect.height)


    # Toggle callback: application's main controller
    def _on_switch(self, is_on):
        if not self.is_started:
            gui.Application.instance.post_to_main_thread(
                self.window, self._on_start)
        self.is_running = not self.is_running
        self.button_compute_slices.enabled = not self.button_compute_slices.enabled
        self.button_compute_mesh.enabled = not self.button_compute_mesh.enabled
        self.button_compute_renders.enabled = not self.button_compute_renders.enabled

    def _recompute_mesh(self):
        self.reconstruct_mesh()
        self.output_info.text = "Recomputed mesh\n\n\n"

    def _recompute_slices(self):
        self.compute_sdf_slices()
        self.output_info.text = "Recomputed slices\n\n\n"

    def _recompute_renders(self):
        self.update_latest_frames(True)
        self.output_info.text = "Recomputed renders\n\n\n"

    def _clear_keyframes(self):
        self.is_started = False
        time.sleep(0.3)
        self.output_info.text = "Clearing keyframes"
        self.trainer.clear_keyframes()
        self.iter = 0
        info, new_kf, end = self.optim_iter(self.trainer, self.iter)
        self.iter += 1
        self.kfs = [new_kf]
        self.output_info.text = f"Iteration {self.iter}\n" + info
        self.clear_kf_frustums = True
        self.is_started = True

    def _toggle_mesh(self, is_on):
        if self.mesh_box.checked:
            self.interval_slider.enabled = True
            self.voxel_dim_slider.enabled = True
            self.col_slider.enabled = True
        else:
            self.widget3d.scene.remove_geometry("rec_mesh")
            self.interval_slider.enabled = False
            self.voxel_dim_slider.enabled = False
            self.col_slider.enabled = False

    def _toggle_slices(self, is_on):
        if self.slices_box.checked:
            self.slices_interval_slider.enabled = True
        else:
            self.widget3d.scene.remove_geometry("sdf_slice")
            self.slices_interval_slider.enabled = False

    def _toggle_frontier(self, is_on):
        if self.frontier_box.checked:
            self.frontier_interval_slider.enabled = True
            self.frontier_ratio_slider.enabled = True
            self.cluster_ratio_slider.enabled = True
            self.vis_cluster_box.enabled = True
            self.vis_cluster_box.checked = True
        else:
            self.frontier_interval_slider.enabled = False
            self.frontier_ratio_slider.enabled = False
            self.cluster_ratio_slider.enabled = False
            self.vis_cluster_box.enabled = False
            self.vis_cluster_box.checked = False

    def _toggle_local(self, is_on):
        if self.local_box.checked:
            self.bbox_box.enabled = True
            self.bbox_box.checked = True
            self.curr_pcd_box.enabled = True
            self.curr_pcd_box.checked = True
        else:
            if self.bbox_box.checked:
                self.widget3d.scene.remove_geometry("local_bbox")
            self.bbox_box.checked = False
            self.bbox_box.enabled = False
            if self.curr_pcd_box.checked:
                self.widget3d.scene.remove_geometry("local_pcd")
            self.curr_pcd_box.checked = False
            self.curr_pcd_box.enabled = False

    def _toggle_bbox(self, is_on):
        if self.bbox_box.checked is False:
            self.widget3d.scene.remove_geometry("local_bbox")

    def _toggle_trajectory(self, is_on):
        if self.traj_box.checked is False:
            self.widget3d.scene.remove_geometry("trajectory")

    def _toggle_global(self, is_on):
        if self.global_box.checked:
            self.origin_box.enabled = True
            self.origin_box.checked = True
            self.traj_box.enabled = True
            self.traj_box.checked = True
            self.kf_box.enabled = True
            self.kf_box.checked = True
            self.pc_box.enabled = True
            self.pc_box.checked = False

        else:
            self.origin_box.checked = False
            self.origin_box.enabled = False
            self.traj_box.checked = False
            self.traj_box.enabled = False
            self.widget3d.scene.remove_geometry("trajectory")
            self.kf_box.checked = False
            self.kf_box.enabled = False
            self.pc_box.checked = False
            self.pc_box.enabled = False

    def _toggle_render(self, is_on):
        if self.render_box.checked is False:
            self.render_normals_depth.update_image(self.no_render)
            self.render_interval_slider.enabled = False
        else:
            self.render_interval_slider.enabled = True

    def _change_voxel_dim(self, val):
        grid_dim = self.voxel_dim_slider.int_value
        grid_pc = geometry.transform.make_3D_grid(
            [-1.0, 1.0],
            grid_dim,
            self.trainer.device,
            transform=self.trainer.bounds_transform,
            scale=self.trainer.scene_scale,
        )
        self.trainer.new_grid_dim = grid_dim
        self.trainer.new_grid_pc = grid_pc.view(-1, 3).to(self.trainer.device)

    # On start: point cloud buffer and model initialization.
    def _on_start(self):

        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        big_pt_mat = rendering.MaterialRecord()
        big_pt_mat.shader = 'defaultUnlit'
        big_pt_mat.sRGB_color = True
        big_pt_mat.point_size = 5

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((self.max_points, 3), dtype=np.float32)))
        pcd_placeholder.point['colors'] = o3c.Tensor(
            np.zeros((self.max_points, 3), dtype=np.float32))
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)
        self.widget3d.scene.scene.add_geometry('slice_pc', pcd_placeholder, mat)
        self.widget3d.scene.scene.add_geometry('local_pcd', pcd_placeholder, mat)

        if self.trainer.gt_scene:
            if self.gt_mesh_box.checked:
                self.widget3d.scene.add_geometry(
                    'gt_mesh', self.gt_mesh, self.lit_mat)

        if self.origin_box.checked:
            self.widget3d.scene.add_geometry('origin', self.origin, self.unlit_mat)

        self.is_started = True

    def _on_close(self):
        self.is_done = True
        print('Finished.')
        return True

    def init_render(self):
        self.output_info.text = "\n\n\n"
        blank = np.full([self.trainer.H, self.trainer.W, 3], 255, dtype=np.uint8)

        blank_im = o3d.geometry.Image(np.hstack([blank] * 2))
        self.input_rgb_depth.update_image(blank_im)
        self.render_normals_depth.update_image(self.no_render)
        self.topdown_map.update_image(blank_im)

        kf_im = o3d.geometry.Image(np.hstack([blank] * self.kf_panel_size))
        for panel in self.keyframe_panels:
            panel.update_image(kf_im)

        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox(self.bound[0], self.bound[1])#[-5,-5,-5], [5,5,5])
        center_x = 0.5 * (self.bound[0][0]+self.bound[1][0])
        center_z = 0.5 * (self.bound[0][2] + self.bound[1][2])
        z_location = max(np.abs(self.bound[0][0]-self.bound[1][0]),np.abs(self.bound[0][2]-self.bound[1][2]))
        self.widget3d.setup_camera(60, bbox, [center_x, 0, center_z])
        self.widget3d.look_at([center_x, 0, center_z], [center_x, -z_location+1, center_z-5], [0, 0, 1])
        #self.widget3d.look_at([0, 0, 1.5], [0, -16, 1.5], [0, 0, 1])#center, eye, up

    def toggle_content(self, name, geometry, mat, show):
        if (self.widget3d.scene.has_geometry(name) is False) and show:
            self.widget3d.scene.add_geometry(name, geometry, mat)
        elif self.widget3d.scene.has_geometry(name) and (show is False):
            self.widget3d.scene.remove_geometry(name)

    def update_render(
        self,
        topdown_map,
        latest_frame,
        render_frame,
        keyframes_vis,
        trajectory,
        rec_mesh,
        slice_pcd,
        pcd,
        local_pcd,
        local_bbox,
        #frontiers,
        latest_frustum,
        kf_frustums,
        target_frustums,
        best_frustum
    ):
        self.input_rgb_depth.update_image(latest_frame)
        self.topdown_map.update_image(topdown_map)
        if render_frame is not None:
            self.render_normals_depth.update_image(render_frame)

        for im, kf_panel in zip(keyframes_vis, self.keyframe_panels):
            kf_panel.update_image(im)

        self.widget3d.scene.remove_geometry("gt_mesh")
        #self.widget3d.scene.add_geometry("gt_mesh", self.gt_mesh, self.lit_mat)

        self.widget3d.scene.scene.update_geometry(
            "points", pcd, rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.scene.update_geometry(
            "slice_pc", slice_pcd, rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.scene.update_geometry(
            "local_pcd", local_pcd, rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)

        if rec_mesh is not None and self.mesh_box.checked:
            self.widget3d.scene.remove_geometry("rec_mesh")
            self.widget3d.scene.add_geometry("rec_mesh", rec_mesh, self.lit_mat)

        if trajectory is not None:
            self.widget3d.scene.remove_geometry("trajectory")
            self.widget3d.scene.add_geometry(
                "trajectory", trajectory, self.slim_unlit_mat)

        if local_bbox is not None:
            self.widget3d.scene.remove_geometry("local_bbox")
            self.widget3d.scene.add_geometry(
            "local_bbox", local_bbox, self.unlit_mat)

        if latest_frustum is not None:
            self.widget3d.scene.remove_geometry("latest_frustum")
            self.widget3d.scene.add_geometry(
                "latest_frustum", latest_frustum, self.unlit_mat)

        if self.clear_kf_frustums:
            for i in range(50):
                self.widget3d.scene.remove_geometry(f"frustum_{i}")
            self.clear_kf_frustums = False

        for i, frustum in enumerate(kf_frustums):
            self.widget3d.scene.remove_geometry(f"frustum_{i}")
            if frustum is not None:
                self.widget3d.scene.add_geometry(f"frustum_{i}", frustum, self.slim_unlit_mat)

        for i, target_frustum in enumerate(target_frustums):
            self.widget3d.scene.remove_geometry(f"tfrustum_{i}")
            if ((target_frustum is not None) and self.frontier_box.checked):
                self.widget3d.scene.add_geometry(f"tfrustum_{i}", target_frustum, self.slim_unlit_mat)

        self.widget3d.scene.remove_geometry("best frustum")
        if ((best_frustum is not None) and self.frontier_box.checked):
            self.widget3d.scene.add_geometry("best frustum", best_frustum, self.slim_unlit_mat)

        if self.trainer.gt_scene:
            self.toggle_content('gt_mesh', self.gt_mesh, self.lit_mat, self.gt_mesh_box.checked)
        self.toggle_content('origin', self.origin, self.unlit_mat, self.origin_box.checked)

    def get_arrow_pts(self, xy, ori, phi=np.pi/7, size=16):
        ori = np.pi + ori
        xy = xy - 0.5 * np.array([np.cos(ori)*size, np.sin(ori)*size])
        arrow_top = xy + np.array([np.cos(ori)*size, np.sin(ori)*size])
        arrow_bottom_size = np.tan(phi)*size
        arrow_bottom_left = xy - np.array([np.sin(ori) * arrow_bottom_size, -np.cos(ori) * arrow_bottom_size])
        arrow_bottom_right = xy + np.array([np.sin(ori) * arrow_bottom_size, -np.cos(ori) * arrow_bottom_size])
        pts = np.array([arrow_top, arrow_bottom_left, arrow_bottom_right])
        return pts


    def get_topdown_vis(self):
        top_down_sdf = self.trainer.get_topdown_sdf(self.topdown_grid)
        top_down_sdf_clamp = torch.clamp(top_down_sdf, -1, 1)
        top_down_sdf_clamp_np = np.flipud(top_down_sdf_clamp.cpu().detach().numpy().T)
        vis = cv2.resize(top_down_sdf_clamp_np, (top_down_sdf_clamp_np.shape[1], top_down_sdf_clamp_np.shape[0]))
        vis = (-vis * 127.5 + 127.5).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TWILIGHT_SHIFTED)

        curr_pose = self.T_WC_latest
        curr_pos = np.array(curr_pose[:3, 3])
        curr_rot = np.array(curr_pose[:3,:3])
        curr_x = geometry.transform.get_2d_coord(curr_pos[0], self.bound[0][0], self.meters_per_pixel)
        curr_y = -geometry.transform.get_2d_coord(curr_pos[2], self.bound[1][2], self.meters_per_pixel)
        curr_ori = geometry.transform.get_orientation(curr_rot)
        pts = self.get_arrow_pts(np.array([curr_x, curr_y]), curr_ori)
        
        for candidate in self.explorer.target_poses:
            pos = np.array(candidate[:3, 3])
            candidate_x = geometry.transform.get_2d_coord(pos[0], self.bound[0][0], self.meters_per_pixel)
            candidate_y = -geometry.transform.get_2d_coord(pos[2], self.bound[1][2], self.meters_per_pixel)
            cv2.circle(vis, np.int32([candidate_x,candidate_y]), 6, (88, 214, 141), -1)

        s = 0
        for selection in self.explorer.selected_frontiers:
            pos = np.array(selection[:3, 3])
            candidate_x = geometry.transform.get_2d_coord(pos[0], self.bound[0][0], self.meters_per_pixel)
            candidate_y = -geometry.transform.get_2d_coord(pos[2], self.bound[1][2], self.meters_per_pixel)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=10, clip=True)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.tab10)
            if s == 2:
                s = 9
            color = (255 * np.array(mapper.to_rgba(s)[0:3])).astype(np.int32)
            color = (int(color[0]), int(color[1]), int(color[2]))
            if s != self.selected_id:
                cv2.circle(vis, np.int32([candidate_x, candidate_y]), 6, (color), -1)
            s += 1

        pos_selected = np.array(self.explorer.selected_frontiers[self.selected_id][:3, 3])
        target_x = geometry.transform.get_2d_coord(pos_selected[0], self.bound[0][0], self.meters_per_pixel)
        target_y = -geometry.transform.get_2d_coord(pos_selected[2], self.bound[1][2], self.meters_per_pixel)
        cv2.circle(vis, np.int32([target_x, target_y]), 6, (231, 76, 60), -1)
        cv2.polylines(vis, np.int32([pts]), True, (int(255 * 0.961), int(255 * 0.475), 0), 3)
        #cv2.fillPoly(vis, np.int32([pts]), (int(255 * 0.961), int(255 * 0.475), 0))
        cv2.circle(vis, np.int32([pts[0,0], pts[0,1]]), 5, (int(255 * 0.961), int(255 * 0.475),0), -1)

        self.two_d_position.append([curr_x, curr_y])
        output_dir = self.trainer.log_dir + self.trainer.dataset_format \
                   + "/" + self.explorer.config.scene_id
        if not os.path.exists(output_dir + "/top_down/"): os.makedirs(output_dir + "/top_down/")
        filename = output_dir + "/top_down/step_" + str(self.step) + ".png"
        if self.step % 1 == 0:
            save = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, save)

        self.topdown_vis = o3d.geometry.Image(vis.astype(np.uint8))

    # Visualization
    def vis_engine_run(self, obs):
        # keyframe vis
        kf_vis = []
        c = 0
        ims = []
        for im in self.kfs:
            im = cv2.copyMakeBorder(
                im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            ims.append(im)
            c += 1
            if c == self.kf_panel_size:
                kf_im = o3d.geometry.Image(np.hstack(ims))
                kf_vis.append(kf_im)
                ims = []
                c = 0
        blank = np.full(im.shape, 255, dtype=np.uint8)
        if len(ims) != 0:
            for _ in range(c, 3):
                ims.append(blank)
            kf_im = o3d.geometry.Image(np.hstack(ims))
            kf_vis.append(kf_im)

        # latest frame vis (rgbd and render) --------------------------------
        do_render = False
        if (self.render_box.checked and
            self.is_key_frame
            #self.step % self.render_interval_slider.int_value == 0
        ):
            do_render = True
        self.update_latest_frames(do_render, obs)

        # 3D vis --------------------------------------
        # sdf slices
        if (
            self.slices_box.checked
        ):
            self.compute_sdf_slices()

        slice_pcd = self.next_slice_pc()

        # point cloud from depth
        pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
        if self.pc_box.checked:
            self.compute_depth_pcd()
            pcd = self.latest_pcd

        local_pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
        local_bbox = None
        bound = None
        if self.local_box.checked:
            curr_pcd, min_bound, max_bound = self.compute_current_pcd(obs)
            if self.curr_pcd_box.checked:
                local_pcd = curr_pcd
            if self.bbox_box.checked:
                local_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                local_bbox.color = (0, 1, 0)
            bound = [min_bound, max_bound]
        # trajectory
        T_WC_np = self.trainer.scene_dataset.process(obs)['T']
        latest_location = T_WC_np[:3, 3].copy()

        if not self.trajectory:
            self.trajectory.append(latest_location)
        elif not np.array_equal(self.trajectory[-1], latest_location):
            self.trajectory.append(latest_location)
        if ((self.traj_box.checked) and (len(self.trajectory) > 1)):
            traj = self.update_traj()
        else:
            traj = None

        # keyframes
        if self.kf_box.checked:
            self.update_kf_frustums()
            kf_frustums = self.latest_frustums
        else:
            kf_frustums = [None] * len(self.latest_frustums)

        # latest frame
        latest_frustum = None

        latest_frustum = o3d.geometry.LineSet.create_camera_visualization(
            self.intrinsic.width,
            self.intrinsic.height,
            self.intrinsic.intrinsic_matrix,
            np.linalg.inv(self.T_WC_latest),
            scale=self.cam_scale,
        )
        latest_frustum.paint_uniform_color([0.961, 0.475, 0.000])
        self.latest_frustum = latest_frustum

        return bound, kf_vis, traj, slice_pcd, \
               pcd, local_pcd, local_bbox, kf_frustums

    def vis_engine_stop(self, obs):
        kf_vis = []
        slice_pcd = self.next_slice_pc()

        pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
        if self.pc_box.checked:
            if self.latest_pcd is None:
                self.compute_depth_pcd()
            pcd = self.latest_pcd

        # local geometry
        local_pcd = o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))
        local_bbox = None
        if self.local_box.checked:
            curr_pcd, min_bound, max_bound = self.compute_current_pcd(obs)
            if self.curr_pcd_box.checked:
                local_pcd = curr_pcd
            if self.bbox_box.checked:
                local_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                local_bbox.color = (0, 1, 0)

        # trajectory
        if ((self.traj_box.checked) and (len(self.trajectory) > 1)):
            traj = self.update_traj()
        else:
            traj = None

        if self.kf_box.checked:
            kf_frustums = self.latest_frustums
        else:
            kf_frustums = [None] * len(self.latest_frustums)
        return kf_vis, traj, slice_pcd, \
                pcd, local_pcd, local_bbox, kf_frustums

    def get_topdown_dimension(self, env):
        self.height = - env.get_agent_state().position[1] # originally OpenGL coordinate
        mesh_bound = self.gt_mesh.get_axis_aligned_bounding_box()
        self.bound = [mesh_bound.get_min_bound(), mesh_bound.get_max_bound()]
        world_width = max([self.bound[0][0], self.bound[1][0]]) - min([self.bound[0][0], self.bound[1][0]])
        world_height = max([self.bound[0][2], self.bound[1][2]]) - min([self.bound[0][2], self.bound[1][2]])
        self.meters_per_pixel = float(world_height / self.trainer.H)
        x = torch.linspace(self.bound[0][0], self.bound[1][0], int(world_width / self.meters_per_pixel)).to(
            self.trainer.device)
        z = torch.linspace(self.bound[0][2], self.bound[1][2], self.trainer.H).to(self.trainer.device)
        xx, zz = torch.meshgrid([x, z])
        yy = torch.ones(xx.size()).to(self.trainer.device) * self.height
        self.topdown_grid = torch.cat(
            (xx[..., None], yy[..., None], zz[..., None]), dim=2)

    # Major loop
    def update_main(self):
        self.data_source = HabitatDataScene([self.trainer.H, self.trainer.W, self.trainer.hfov], self.explorer.config,
                                            config_file=self.explorer.config_file,
                                            scene_id=self.explorer.config.scene_id)
        print("========================== ", self.trainer.dataset_format, " - ", self.explorer.config.scene_id)

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

        # set 2D vis grid
        self.get_topdown_dimension(env)

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render()
        )
        self.iter = self.trainer.iter
        self.step = 0

        self.last_keyframe_id = 0
        self.l_policy.reset()

        t_start = time.time()

        output_dir = self.trainer.log_dir + self.trainer.dataset_format \
                     + "/" + self.explorer.config.scene_id + "/results"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        f_action = open(output_dir + "/action.txt", "w")
        f_error = open(output_dir + "/error.txt", "w")
        f_keyframes = open(output_dir + "/keyframes.txt", "w")

        while not self.is_done:
            t0 = time.time()
            last_net_params = deepcopy(self.trainer.sdf_map.parameters)

            gt_vertices = np.array(self.gt_mesh.vertices)
            filtered_mask = gt_vertices[:, 1] < (self.bound[0][1] + self.height_slider.double_value)
            self.gt_mesh.remove_vertices_by_mask(filtered_mask)

            if self.is_running:
                # training steps ------------------------------
                sim_obs = env.get_sensor_observations()
                data_source = env._sensor_suite.get_observations(sim_obs)
                curr_pose = env.get_agent_state()
                pose_position = curr_pose.position
                pose_rotation = curr_pose.rotation

                if self.step > 600:#600:#1100:
                    self.crop_box.checked = False
                if self.step > 900:#900:#1950:
                    self.voxel_dim_slider.int_value = 160
                if self.step > 950:#950:#1980:
                    self.gt_mesh_box.checked = True

                self.trainer.frame_id = self.step
                image = data_source['rgb'][:, :, :3].detach().cpu().numpy()
                depth = data_source['depth'].detach().cpu().numpy()
                depth = np.squeeze(depth)
                info, new_kf, end = self.mapper(self.trainer, self.step, self.training_iters_slider.int_value, [image, depth, pose_position, pose_rotation])
                self.is_key_frame = (new_kf is not None)
                if new_kf is not None:
                    self.kfs.append(new_kf)
                    self.last_keyframe_id = self.step
                self.iter += self.training_iters_slider.int_value
                f_keyframes.write(f"{len(self.kfs)}\n")

                curr_bound, kf_vis, traj, slice_pcd, \
                pcd, local_pcd, local_bbox, kf_frustums = \
                    self.vis_engine_run([image, depth, pose_position, pose_rotation])
                # reconstructed mesh from marching cubes on zero level set
                interval = self.step - self.last_keyframe_id
                assert((interval % self.interval_slider.int_value) < self.interval_slider.int_value)
                rec_mesh_status = False
                if (
                    (np.any([self.step == 0, self.is_key_frame, interval % self.interval_slider.int_value == 0]))
                ):
                    self.reconstruct_mesh()
                    rec_mesh = self.latest_mesh
                    rec_mesh_status = True
                # Frontier extraction
                if (self.step%50 == 0):
                    df, color = self.explorer.vis_prediction_error( \
                            self.trainer, input_mesh=self.gt_mesh)
                    write_mean = np.mean(df)
                    write_ratio = np.array(df)[df<0.05].shape[0] / np.array(df.squeeze()).shape[0]
                    f_error.write(f"{write_mean}  {write_ratio}\n")
                    self.gt_mesh.vertex_colors = o3d.utility.Vector3dVector(color)

                frontier_changes = False
                if (
                    rec_mesh_status
                ):
                    assert rec_mesh is not None
                    color_frontiers, self.n_clusters = self.explorer.compute_frontiers(\
                                            self.trainer,last_net_params, input_mesh=rec_mesh, \
                                            ratio=self.frontier_ratio_slider.double_value, \
                                            visualize_clusters=self.vis_cluster_box.checked,\
                                            removed_cluster_ratio=self.cluster_ratio_slider.double_value,\
                                            agent_Y=self.height)

                    self.output_info.text = "Computing Frontiers\n"
                    selected_frontiers = self.explorer.select_frontiers(trainer=self.trainer, agent_height=self.data_source.agent_height, curr_bound = curr_bound)

                    # Visualization
                    target_frustums = self.update_target_frustums(self.explorer.target_poses)
                    frontier_changes = True
                if (((self.explorer.rho < 0.8 ) or (self.explorer.frontier_last_step > 100)) and frontier_changes):
                    self.explorer.frontier_last_step = 0
                    self.explorer.selected_frontiers = selected_frontiers
                    
                    self.selected_id = 3 # 0-area, 1-\sigma_max, 2-\sigma_mean, 3-number, 4-distance
                    self.target_pose = self.explorer.selected_frontiers[self.selected_id]
                    best_frustum = o3d.geometry.LineSet.create_camera_visualization(
                            self.intrinsic.width,
                            self.intrinsic.height,
                            self.intrinsic.intrinsic_matrix,
                            np.linalg.inv(self.target_pose),
                            scale=self.cam_scale,
                    )
                    best_frustum.paint_uniform_color([231./255, 76./255, 60./255])
                self.explorer.frontier_last_step += 1

                if self.col_slider.int_value == 3:
                    rec_mesh.vertex_colors = o3d.utility.Vector3dVector(color_frontiers)
                
                # DD-PPO =============================================
                #act = self.explorer.run_manual_policy()
                target_location = self.target_pose[:3, 3]
                current_location = self.trainer.frames.T_WC_batch_np[-1][:3, 3]
                curr_x = geometry.transform.get_2d_coord(current_location[0], self.bound[0][0], self.meters_per_pixel)
                curr_y = -geometry.transform.get_2d_coord(current_location[2], self.bound[1][2], self.meters_per_pixel)
                tar_x = geometry.transform.get_2d_coord(target_location[0], self.bound[0][0], self.meters_per_pixel)
                tar_y = -geometry.transform.get_2d_coord(target_location[2], self.bound[1][2], self.meters_per_pixel)
                phi = torch.atan2(torch.tensor(curr_x - tar_x), torch.tensor(curr_y-tar_y))
                curr_rot = self.trainer.frames.T_WC_batch_np[-1][:3, :3]
                curr_ori = geometry.transform.get_orientation(curr_rot)
                first_ori = geometry.transform.get_orientation(self.trainer.frames.T_WC_batch_np[0])
                rel_ori = curr_ori - first_ori

                phi += rel_ori
                if phi > 2 * np.pi:
                    phi -= 2 * np.pi
                elif phi < -2*np.pi:
                    phi += 2*np.pi

                if phi > np.pi:
                    phi -= 2*np.pi
                elif phi < -np.pi:
                    phi += 2*np.pi

                rho = np.linalg.norm(target_location - current_location)
                self.explorer.rho = rho
                self.prepend_text = 'Remaining distance/angle:\n{}/{}\n\n'\
                    .format('%.3f'%rho, '%.3f'%(phi/np.pi*180))
                with torch.no_grad():
                    act = self.run_local_policy(depth=data_source['depth'].reshape(256,256,1), rho=rho, phi=phi, \
                                                    step=self.step, device=self.trainer.device)
                # 2D visualization
                self.get_topdown_vis()

                if end:
                    self.prepend_text = "SEQUENCE ENDED - CONTINUING TRAINING\n"
                self.output_info.text = self.prepend_text + \
                                        f"Step {self.step} -- Iteration {self.iter}\n" + info

                t1 = time.time()
                self.optim_times.append(t1 - t0)

            else:
                while (not self.is_running):
                    info, new_kf, end = self.mapper(self.trainer, self.step, self.training_iters_slider.int_value)
                    self.iter += self.training_iters_slider.int_value
                    kf_vis, traj, slice_pcd, \
                    pcd, local_pcd, local_bbox, kf_frustums = \
                        self.vis_engine_stop([image, depth, pose_position, pose_rotation])
                    rec_mesh = None
                    if self.mesh_box.checked:
                        rec_mesh = self.latest_mesh
                    self.output_info.text = self.prepend_text + \
                                            f"Step {self.step} -- Iteration {self.iter}\n" + info

                    time.sleep(0.05)

            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_render(
                    self.topdown_vis,
                    self.latest_frame,
                    self.render_frame,
                    kf_vis,
                    traj,
                    rec_mesh,
                    slice_pcd,
                    pcd,
                    local_pcd,
                    local_bbox,
                    self.latest_frustum,
                    kf_frustums,
                    target_frustums,
                    best_frustum
                )
            )

            if not end:
                log_info = "Scene id: {}-{} \n\n".format(self.trainer.dataset_format,self.explorer.config.scene_id)
                log_info += 'Frame {}/{}\n\n'.format(min(self.trainer.get_latest_frame_id(),len(self.trainer.scene_dataset)), len(self.trainer.scene_dataset))
                log_info += 'Last keyframe id: {}\n'.format(self.last_keyframe_id)
                T_WC_np = self.trainer.scene_dataset.process([image, depth, pose_position, pose_rotation])['T']
                log_info += 'Transformation:\n{}\n'.format(
                    np.array2string(T_WC_np,
                                    precision=3,
                                    max_line_width=40,
                                    suppress_small=True))
                log_info += '\nLatest keyframe criteria: {}/{}\n'.format(
                    '%.3f' % self.trainer.below_th_prop,
                    self.trainer.kf_pixel_ratio)
                log_info += 'Number of frontier clusters: {}\n'.format(self.n_clusters)

            data_source = None

            if self.is_running:
                self.vis_times.append(time.time() - t1)

                t_vis = np.sum(self.vis_times)
                t_optim = np.sum(self.optim_times)
                t_tot = t_vis + t_optim
                prop_vis = int(np.round(100 * t_vis / t_tot))
                prop_optim = int(np.round(100 * t_optim / t_tot))
                while t_tot > 20:
                    self.vis_times.pop(0)
                    self.optim_times.pop(0)
                    t_tot = np.sum(self.vis_times) + np.sum(self.optim_times)

                self.timings.text = "Compute balance in last 20s:\n" +\
                    f"training {prop_optim}% : visualisation {prop_vis}%"

                self.step += 1

            action_space = ["STOP","MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
            if (act in [1,2,3]):
                env.step(act)
            else:
                print("Action out of the action space: ")
                act = np.random.randint(1, 4)
                env.step(act)
                #act = 0
            f_action.write(str(act))

            log_info += 'Action: {}\n'.format(action_space[act])
            self.log_info.text = log_info

            if self.step == self.trainer.n_steps:
                print("Finished!")
                f_action.close()
                f_error.close()
                f_keyframes.close()
                t_end = time.time()
                save_ckpts = self.trainer.save_checkpoints
                save_mesh = self.trainer.save_meshes
                with open(output_dir + "/config.json", "w") as outfile:
                    json.dump(self.trainer.config, outfile, indent=4)
                f = open(output_dir + "/runtime.txt", "w")
                f.write(str(t_end-t_start))
                f.close()
                if save_ckpts:
                    print("Save checkpoints.")
                    if not os.path.exists(output_dir + "/checkpoints"): os.makedirs(output_dir + "/checkpoints")
                    torch.save(
                        {
                            "step": self.step,
                            "model_state_dict":
                                self.trainer.sdf_map.state_dict(),
                            "optimizer_state_dict":
                                self.trainer.optimiser.state_dict()
                        },

                        output_dir + "/checkpoints/step_" + str(self.step) + ".pth"
                    )
                if save_mesh:
                    print("Save mesh.")
                    mesh_path = output_dir + '/mesh.ply'
                    o3d.io.write_triangle_mesh(mesh_path, rec_mesh)

                gui.Application.instance.quit()
                break
        
        print("==============End==================")
        time.sleep(0.5)

    def reconstruct_mesh(self):
        self.output_info.text = "Computing mesh reconstruction with marching cubes\n\n\n"
        rec_trimesh = self.trainer.mesh_rec(self.crop_box.checked, self.col_slider.int_value)
        rec_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(rec_trimesh.vertices),
            triangles=o3d.utility.Vector3iVector(rec_trimesh.faces))
        rec_mesh.vertex_colors=o3d.utility.Vector3dVector(rec_trimesh.visual.vertex_colors[:,:3]/255.)
        rec_mesh.compute_vertex_normals()
        if self.col_slider.int_value == 2:
            rec_mesh.vertex_colors = rec_mesh.vertex_normals
        vertices = np.array(rec_mesh.vertices)
        filtered_mask = vertices[:, 1] < (self.bound[0][1] + self.height_slider.double_value)
        rec_mesh.remove_vertices_by_mask(filtered_mask)

        self.latest_mesh = rec_mesh

    def next_slice_pc(self):
        if self.slices_box.checked and self.sdf_grid_pc is not None:
            slice_pc = self.sdf_grid_pc[:, :, self.slice_ix].reshape(-1, 4)
            slice_pcd = o3d.t.geometry.PointCloud(o3c.Tensor(slice_pc[:, :3]))
            slice_cols = self.colormap_fn.to_rgba(slice_pc[:, 3], bytes=False)
            slice_cols = slice_cols[:, :3].astype(np.float32)
            slice_pcd.point['colors'] = o3c.Tensor(slice_cols)
            # next slice
            if self.slice_ix == self.sdf_grid_pc.shape[2] - 1:
                self.slice_step = -1
            if self.slice_ix == 0:
                self.slice_step = 1
            self.slice_ix += self.slice_step
            return slice_pcd
        else:
            return o3d.t.geometry.PointCloud(np.zeros((1, 3), dtype=np.float32))

    def compute_sdf_slices(self):
        self.output_info.text = "Computing SDF slices\n\n\n"
        sdf_grid_pc, _ = self.trainer.get_sdf_grid_pc(mask_near_pc=self.crop_box.checked)
        sdf_grid_pc = np.transpose(sdf_grid_pc, (2, 1, 0, 3))
        self.sdf_grid_pc = sdf_grid_pc
        sdf_range = [self.sdf_grid_pc[..., 3].min(), self.sdf_grid_pc[..., 3].max()]

        self.colormap_fn = sdf_util.get_colormap(sdf_range, 0.02)

        fig, ax = plt.subplots(figsize=(5, 2), tight_layout=True)
        plt.colorbar(self.colormap_fn, ax=ax, orientation='horizontal')
        ax.remove()
        fig.set_tight_layout(True)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data[120:]
        plt.cla()
        plt.close()

    def compute_depth_pcd(self):
        T_WC_batch = self.trainer.frames.T_WC_batch_np
        self.trainer.update_vis_vars()
        pcs_cam = geometry.transform.backproject_pointclouds(
            self.trainer.gt_depth_vis, self.trainer.fx_vis, self.trainer.fy_vis,
            self.trainer.cx_vis, self.trainer.cy_vis)
        pcs_cam = np.einsum('Bij,Bkj->Bki', T_WC_batch[:, :3, :3], pcs_cam)
        pcs_world = pcs_cam + T_WC_batch[:, None, :3, 3]
        pcs_world = pcs_world.reshape(-1, 3).astype(np.float32)
        cols = self.trainer.gt_im_vis.reshape(-1, 3)
        cols = cols.astype(np.float32) / 255
        if len(pcs_world) > self.max_points:
            ixs = np.random.choice(
                np.arange(len(pcs_world)), self.max_points, replace=False)
            pcs_world = pcs_world[ixs].astype(np.float32)
            cols = cols[ixs].astype(np.float32)
        mask = pcs_world[:, 1] > (self.bound[0][1] + self.height_slider.double_value)
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(pcs_world[mask]))
        pcd.point['colors'] = o3c.Tensor(cols[mask])
        self.latest_pcd = pcd

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
        col_curr = img_resize.reshape(-1, 3)
        col_curr = col_curr.astype(np.float32) / 255
        mask = pcd_curr[:, 1] > (self.bound[0][1] + self.height_slider.double_value)
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(pcd_curr[mask].astype(np.float32)))
        pcd.point['colors'] = o3c.Tensor(col_curr[mask])
        return pcd, \
               np.array([np.min(pcd_curr[:, 0]), np.min(pcd_curr[:, 1]), np.min(pcd_curr[:, 2])]),\
               np.array([np.max(pcd_curr[:, 0]), np.max(pcd_curr[:, 1]), np.max(pcd_curr[:, 2])])

    def update_kf_frustums(self):
        kf_frustums = []
        T_WC_batch = self.trainer.frames.T_WC_batch_np
        for T_WC in T_WC_batch[:-1]:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.intrinsic.width,
                self.intrinsic.height,
                self.intrinsic.intrinsic_matrix,
                np.linalg.inv(T_WC),
                scale=self.cam_scale,
            )
            frustum.paint_uniform_color([0.000, 0.475, 0.900])
            kf_frustums.append(frustum)
        self.latest_frustums = kf_frustums

    def update_target_frustums(self, target_poses):
        target_frustums = []
        for T_WC in target_poses:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.intrinsic.width,
                self.intrinsic.height,
                self.intrinsic.intrinsic_matrix,
                np.linalg.inv(T_WC),
                scale=self.cam_scale,
            )
            frustum.paint_uniform_color([88./255, 214./255, 141./255])
            target_frustums.append(frustum)
        return target_frustums

    def update_traj(self):
        points = []
        lines = []
        for i in range(len(self.trajectory)):
            points.append([self.trajectory[i]])
            if i < len(self.trajectory)-1:
                lines.append([[i, i+1]])
        points = np.array(points).astype(np.float32)
        points = points.reshape(-1, 3)
        lines = np.array(lines)
        lines = lines.reshape(-1, 2)
        traj = o3d.geometry.LineSet()
        traj.points=o3d.utility.Vector3dVector(points)
        traj.lines=o3d.utility.Vector2iVector(lines)
        traj.paint_uniform_color([0, 0, 1])
        return traj

    def update_latest_frames(self, do_render, obs):
        rgbd_vis, render_vis, T_WC = self.trainer.latest_frame_vis(do_render, obs)
        self.T_WC_latest = T_WC
        latest_frame = o3d.geometry.Image(rgbd_vis.astype(np.uint8))
        render_frame = None
        if render_vis is not None:
            render_frame = o3d.geometry.Image(render_vis.astype(np.uint8))
        self.latest_frame = latest_frame
        self.render_frame = render_frame
