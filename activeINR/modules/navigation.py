import torch
import numpy as np
from math import atan
import json
import cv2
import random
import math
import quaternion
import argparse

import open3d as o3d
import matplotlib.cm
import copy


class Explorer():
    def __init__(self, device, config_file, scene_id = None):
        super(Explorer, self).__init__()
        with open(config_file) as json_file:
            options = json.load(json_file)
        #print("options:", options)
        self.rho = 0
        self.frontier_last_step = 0
        self.config = argparse.ArgumentParser()
        self.config.device = device
        if scene_id is not None:
            options["dataset"]["scenes_list"] = scene_id
        self.set_params(options)

        # build summary dir
        self.config_file = self.config.config
        # self.config_file = self.config.config_noisy

        self.step_count = 0

    def set_params(self, options):
        # Dataset
        # require dataset format, depth scale and camera params
        self.config.dataset_format = options["dataset"]["format"]
        self.config.scene_id = options["dataset"]["scenes_list"]
        self.config.config = options["dataset"]["config"]
        self.config.config_noisy = options["dataset"]["noisy_config"]
        self.config.root = options["dataset"]["root"]
        self.config.noisy_pose = options["dataset"]["noisy_pose"]
        self.config.local_policy_path = options["planner"]["model_dir"] + options["planner"]["file_name"]

    def run_manual_policy(self):
        # import random
        # action_id = random.randint(1, 3)
        act = None

        FORWARD_KEY = "w"
        LEFT_KEY = "a"
        RIGHT_KEY = "d"
        DONE_KEY = "q"

        k = cv2.waitKey()
        if k == ord(FORWARD_KEY):
            act = 1
        elif k == ord(LEFT_KEY):
            act = 2
        elif k == ord(RIGHT_KEY):
            act = 3
        return act

    def vis_neural_variability(self, trainer, last_net_params,\
                          input_mesh=None, ratio=0.1, visualize_clusters = True, \
                          removed_cluster_ratio = 0.05, ray_length = 0.5, agent_Y = None):
        mesh_enabled = True
        surface_points = np.asarray(input_mesh.vertices)
        # if surface_points is None:
        #     T_WC_batch = self.trainer.frames.T_WC_batch_np
        #     self.trainer.update_vis_vars()
        #     pcs_cam = geometry.transform.backproject_pointclouds(
        #         self.trainer.gt_depth_vis, self.trainer.fx_vis, self.trainer.fy_vis,
        #         self.trainer.cx_vis, self.trainer.cy_vis)
        #     pcs_cam = np.einsum('Bij,Bkj->Bki', T_WC_batch[:, :3, :3], pcs_cam)
        #     pcs_world = pcs_cam + T_WC_batch[:, None, :3, 3]
        #     pcs_world = pcs_world.reshape(-1, 3).astype(np.float32)
        #     tree = KDTree(pcs_world)
        #     mask = pcs_world[:, 1] > (self.bound[0][1] + self.height_slider.double_value)
        #     surface_points = pcs_world[mask]
        #     mesh_enabled = False`
        mean, var = trainer.vis_perturb(surface_points, last_net_params, True)
        mean = np.log(mean)
        frontier_col_margin = [1.5 * np.min(mean), np.max(mean)]
        mean[mean<frontier_col_margin[0]] = frontier_col_margin[0]
        mean[mean>frontier_col_margin[1]] = frontier_col_margin[1]
        rescaled_mean = (mean - frontier_col_margin[0]) / (frontier_col_margin[1] - frontier_col_margin[0])
        #rescaled_mean =  (mean - -7.1885166 ) / (-7.129596 - -7.1885166 )#5.229596
        rescaled_mean *= rescaled_mean
        rescaled_mean *= rescaled_mean
        rescaled_mean *= rescaled_mean
        #print(rescaled_mean.min(), rescaled_mean.max())

        if mesh_enabled:
            col = np.ones((rescaled_mean.size, 3), dtype=np.float32)
            col[:,1] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
            col[:,2] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        else:
            col = np.zeros((rescaled_mean.size, 3), dtype=np.float32)
            col[:, 0] = rescaled_mean.copy()

        norm = matplotlib.colors.Normalize(vmin=min(rescaled_mean), vmax=max(rescaled_mean), clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.coolwarm)
        col = mapper.to_rgba(rescaled_mean)[:, 0:3]

        #torch.cuda.empty_cache()
        # col = np.ones((rescaled_mean.size, 3), dtype=np.float32)
        # col[:,1] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        # col[:,2] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        #col[:,3] = rescaled_mean
        # import open3d as o3d
        # import open3d.core as o3c
        # pcd = o3d.t.geometry.PointCloud(o3c.Tensor(surface_points))
        # pcd.point['colors'] = o3c.Tensor(col)
        return col

    def vis_prediction_error(self, trainer, \
                          input_mesh=None):
        mesh_enabled = True
        surface_points = np.asarray(input_mesh.vertices)
        # if surface_points is None:
        #     T_WC_batch = self.trainer.frames.T_WC_batch_np
        #     self.trainer.update_vis_vars()
        #     pcs_cam = geometry.transform.backproject_pointclouds(
        #         self.trainer.gt_depth_vis, self.trainer.fx_vis, self.trainer.fy_vis,
        #         self.trainer.cx_vis, self.trainer.cy_vis)
        #     pcs_cam = np.einsum('Bij,Bkj->Bki', T_WC_batch[:, :3, :3], pcs_cam)
        #     pcs_world = pcs_cam + T_WC_batch[:, None, :3, 3]
        #     pcs_world = pcs_world.reshape(-1, 3).astype(np.float32)
        #     tree = KDTree(pcs_world)
        #     mask = pcs_world[:, 1] > (self.bound[0][1] + self.height_slider.double_value)
        #     surface_points = pcs_world[mask]
        #     mesh_enabled = False`
        sdf = trainer.vis_accuracy(surface_points, True)
        df = np.abs(sdf)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Greens)
        col = mapper.to_rgba(df)[:, 0:3]

        #torch.cuda.empty_cache()
        # col = np.ones((rescaled_mean.size, 3), dtype=np.float32)
        # col[:,1] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        # col[:,2] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        #col[:,3] = rescaled_mean
        # import open3d as o3d
        # import open3d.core as o3c
        # pcd = o3d.t.geometry.PointCloud(o3c.Tensor(surface_points))
        # pcd.point['colors'] = o3c.Tensor(col)
        return df, col

    def compute_frontiers(self, trainer, last_net_params,\
                          input_mesh=None, ratio=1.0, visualize_clusters = True, \
                          removed_cluster_ratio = 0.05, ray_length = 0.5, agent_Y = None):# ratio=0.1, removed_cluster_ratio=0.05
        mesh_enabled = True
        surface_points = np.asarray(input_mesh.vertices)
        # if surface_points is None:
        #     T_WC_batch = self.trainer.frames.T_WC_batch_np
        #     self.trainer.update_vis_vars()
        #     pcs_cam = geometry.transform.backproject_pointclouds(
        #         self.trainer.gt_depth_vis, self.trainer.fx_vis, self.trainer.fy_vis,
        #         self.trainer.cx_vis, self.trainer.cy_vis)
        #     pcs_cam = np.einsum('Bij,Bkj->Bki', T_WC_batch[:, :3, :3], pcs_cam)
        #     pcs_world = pcs_cam + T_WC_batch[:, None, :3, 3]
        #     pcs_world = pcs_world.reshape(-1, 3).astype(np.float32)
        #     tree = KDTree(pcs_world)
        #     mask = pcs_world[:, 1] > (self.bound[0][1] + self.height_slider.double_value)
        #     surface_points = pcs_world[mask]
        #     mesh_enabled = False`
        #gradient, mean, score = trainer.eval_BALD(surface_points)
        gradient, mean, var = trainer.eval_perturb(surface_points, last_net_params)
        #sorted_score = np.sort(score)
        sorted_mean = np.sort(mean)
        threshold = sorted_mean[int(sorted_mean.shape[0] * (1. - ratio))]
        filtered_mask = mean <= threshold
        #threshold = sorted_score[int(sorted_score.shape[0] * (1.-ratio))]
        #filtered_mask = score <= threshold
        #print("sorted_score",sorted_score)
        #print("threshold",threshold)

        mesh = copy.deepcopy(input_mesh)
        #print(np.array(mesh.vertices).shape, np.array(mesh.triangles).shape, gradient.shape)
        mesh.remove_vertices_by_mask(filtered_mask)
        gradient = gradient[np.invert(filtered_mask)]
        uncertainty = copy.deepcopy(mean)#score)
        uncertainty = uncertainty[np.invert(filtered_mask)]
        #print(np.array(mesh.vertices).shape, np.array(mesh.triangles).shape, gradient.shape)
        mesh.paint_uniform_color([1, 0.706, 0])

        # Frontier clustering
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangle_n_triangles = cluster_n_triangles[triangle_clusters]
        cluster_area = np.asarray(cluster_area)
        removed_cluster_size = int(removed_cluster_ratio * np.array(mesh.triangles).shape[0])
        #print(removed_cluster_size)
        #print(cluster_area)

        cluster_area = cluster_area[cluster_n_triangles>= removed_cluster_size]
        cluster_n_triangles = cluster_n_triangles[cluster_n_triangles >= removed_cluster_size]

        #print(triangle_clusters, cluster_n_triangles, cluster_area)

        num_clusters = sum(cluster_n_triangles >= removed_cluster_size)
        #assert visualize_clusters == True
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.tab20)
        if len(cluster_n_triangles[cluster_n_triangles >= removed_cluster_size]) >= 1:
            candidate_cluster = cluster_n_triangles[cluster_n_triangles >= removed_cluster_size]
        else:
            candidate_cluster = cluster_n_triangles
            print("less than one cluster")
            print(len(candidate_cluster))
        i = 0
        self.target_poses = []
        # self.mesh_frontiers = o3d.geometry.TriangleMesh()
        self.cluster_area = []
        self.cluster_uncertainty_max = []
        self.cluster_uncertainty_mean = []
        self.cluster_samples = []
        # self.cluster_distance = []
        for n_triangles in candidate_cluster:
            mesh_i = copy.deepcopy(mesh)
            removed_mask_i = triangle_n_triangles != n_triangles
            color = mapper.to_rgba(i)[0:3]
            mesh_i.paint_uniform_color(color)
            # self.mesh_frontiers += mesh_i

            mesh_i.remove_triangles_by_mask(removed_mask_i)
            vertices_id = np.unique(np.array(mesh_i.triangles))
            self.cluster_samples.append(cluster_n_triangles[i])
            self.cluster_area.append(cluster_area[i])
            vertices_i = np.array(mesh_i.vertices)[vertices_id]
            gradients_i = gradient[vertices_id]
            uncertainty_i = uncertainty[vertices_id]
            self.cluster_uncertainty_max.append(np.max(uncertainty_i))
            self.cluster_uncertainty_mean.append(np.mean(uncertainty_i))
            samples_i = vertices_i + ray_length * gradients_i
            direction_i = - gradients_i.mean(0)
            position_i = samples_i.mean(0)

            pitch = atan(direction_i[0] / direction_i[2])
            rot_i = quaternion.as_rotation_matrix( \
                quaternion.from_euler_angles(np.array([0, pitch, 0])))
            pose_i = np.eye(4)
            pose_i[:3, :3] = rot_i
            if agent_Y is not None:
                position_i[1] = agent_Y
            pose_i[:3, 3] = position_i
            self.target_poses.append(pose_i)
            # distance_i = np.linalg.norm(position_i - trainer.frames.T_WC_batch_np[-1][:3,3].copy())
            # self.cluster_distance.append(distance_i)
            i += 1

        # else:
        #     self.mesh_frontiers = copy.deepcopy(mesh)
        #     removed_mask = triangle_n_triangles < removed_cluster_size
        #     self.mesh_frontiers.remove_triangles_by_mask(removed_mask)

        # print(self.cluster_area,self.cluster_uncertainty_max,\
        #       self.cluster_uncertainty_mean,self.cluster_samples,\
        #       self.cluster_distance)
        mean = np.log(mean)
        frontier_col_margin = [np.min(mean), np.max(mean)]
        mean[mean < frontier_col_margin[0]] = frontier_col_margin[0]
        mean[mean > frontier_col_margin[1]] = frontier_col_margin[1]
        rescaled_mean = (mean - frontier_col_margin[0]) / (frontier_col_margin[1] - frontier_col_margin[0])
        rescaled_mean *= rescaled_mean
        rescaled_mean *= rescaled_mean
        rescaled_mean *= rescaled_mean
        # score = np.log(score+1e-10)
        # frontier_col_margin = [np.min(score), np.max(score)]
        # score[score<frontier_col_margin[0]] = frontier_col_margin[0]
        # score[score>frontier_col_margin[1]] = frontier_col_margin[1]
        # rescaled_score =  (score - frontier_col_margin[0]) / (frontier_col_margin[1] - frontier_col_margin[0])
        # rescaled_score *= rescaled_score
        # rescaled_score *= rescaled_score
        # rescaled_score *= rescaled_score
        #print(rescaled_mean.min(), rescaled_mean.max())
        norm = matplotlib.colors.Normalize(vmin=min(rescaled_mean), vmax=max(rescaled_mean), clip=True)
        #norm = matplotlib.colors.Normalize(vmin=min(rescaled_score), vmax=max(rescaled_score), clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.coolwarm)
        col = mapper.to_rgba(rescaled_mean)[:, 0:3]
        #col = mapper.to_rgba(rescaled_score)[:, 0:3]

        #torch.cuda.empty_cache()
        # col = np.ones((rescaled_mean.size, 3), dtype=np.float32)
        # col[:,1] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        # col[:,2] = np.ones((rescaled_mean.size), dtype=np.float32) - rescaled_mean
        #col[:,3] = rescaled_mean
        # import open3d as o3d
        # import open3d.core as o3c
        # pcd = o3d.t.geometry.PointCloud(o3c.Tensor(surface_points))
        # pcd.point['colors'] = o3c.Tensor(col)
        return col, num_clusters

    def select_frontiers(self, trainer, check_num=10, \
                         agent_height = 1.5, discard_occupied = True, curr_bound=None):
        agent_height -= 0.4 # assume 0.4m guarantees traversability
        frontier_pool = np.array(self.target_poses)
        num_targets = frontier_pool.shape[0]
        #print(frontier_pool.shape)
        current_location = trainer.frames.T_WC_batch_np[-1][:3, 3]
        target_locations = frontier_pool[:, :3, 3]
        target_check = target_locations.repeat(check_num, axis=0) #reshape(-1,check_num, 3)
        samples_dY = np.linspace(0, agent_height, num=check_num)
        samples_dY = samples_dY.repeat(num_targets).reshape(-1,num_targets).T.reshape(-1,1)
        samples = target_check.copy()
        samples[:,1] += samples_dY.squeeze()
        samples_torch = torch.FloatTensor(samples).cuda().to(self.config.device)
        sdf_samples = trainer.get_sdf(samples_torch).detach().cpu().numpy().reshape(-1,check_num)
        cluster_occupied = np.all(sdf_samples>0, axis=1)
        #print("Occupied clusters: ", np.array(np.where(cluster_occupied == False)).squeeze())
        if discard_occupied and cluster_occupied.any():
            frontier_pool = frontier_pool[cluster_occupied]
            self.cluster_area = np.array(self.cluster_area)[cluster_occupied]
            self.cluster_uncertainty_max = np.array(self.cluster_uncertainty_max)[cluster_occupied]
            self.cluster_uncertainty_mean = np.array(self.cluster_uncertainty_mean)[cluster_occupied]
            self.cluster_samples = np.array(self.cluster_samples)[cluster_occupied]
            target_locations = target_locations[cluster_occupied]

        self.cluster_distance = np.linalg.norm(target_locations - current_location, axis=1)

        far_from_view = True
        further_than_1m = self.cluster_distance > 1

        if far_from_view and further_than_1m.any():
            frontier_pool = frontier_pool[further_than_1m]
            self.cluster_area = np.array(self.cluster_area)[further_than_1m]
            self.cluster_uncertainty_max = np.array(self.cluster_uncertainty_max)[further_than_1m]
            self.cluster_uncertainty_mean = np.array(self.cluster_uncertainty_mean)[further_than_1m]
            self.cluster_samples = np.array(self.cluster_samples)[further_than_1m]
            self.cluster_distance = np.array(self.cluster_distance)[further_than_1m]
            target_locations = target_locations[further_than_1m]

        if curr_bound is not None:
            min_bound, max_bound = curr_bound[0], curr_bound[1]
            within_bbox = np.logical_and((target_locations>min_bound).all(axis=1),(target_locations<max_bound).all(axis=1))
            if within_bbox.any():
                frontier_pool = frontier_pool[within_bbox]
                self.cluster_area = np.array(self.cluster_area)[within_bbox]
                self.cluster_uncertainty_max = np.array(self.cluster_uncertainty_max)[within_bbox]
                self.cluster_uncertainty_mean = np.array(self.cluster_uncertainty_mean)[within_bbox]
                self.cluster_samples = np.array(self.cluster_samples)[within_bbox]
                self.cluster_distance = np.array(self.cluster_distance)[within_bbox]
                #target_locations = target_locations[within_bbox]



        return [frontier_pool[np.argmax(self.cluster_area)], \
               frontier_pool[np.argmax(self.cluster_uncertainty_max)], frontier_pool[np.argmax(self.cluster_uncertainty_mean)], \
               frontier_pool[np.argmax(self.cluster_samples)], frontier_pool[np.argmin(self.cluster_distance)]]

    def add_pose_noise(self, rel_pose, action_id):
        if action_id == 1:
            x_err, y_err, o_err = self.test_ds.sensor_noise_fwd.sample()[0][0]
        elif action_id == 2:
            x_err, y_err, o_err = self.test_ds.sensor_noise_left.sample()[0][0]
        elif action_id == 3:
            x_err, y_err, o_err = self.test_ds.sensor_noise_right.sample()[0][0]
        else:
            x_err, y_err, o_err = 0., 0., 0.
        rel_pose[0,0] += x_err*self.options.noise_level
        rel_pose[0,1] += y_err*self.options.noise_level
        rel_pose[0,2] += torch.tensor(np.deg2rad(o_err*self.options.noise_level))
        return rel_pose


    def dot(self, v1, v2):
        return v1[0]*v2[0]+v1[1]*v2[1]

    def get_angle(self, line1, line2):
        v1 = [(line1[0][0]-line1[1][0]), (line1[0][1]-line1[1][1])]
        v2 = [(line2[0][0]-line2[1][0]), (line2[0][1]-line2[1][1])]
        dot_prod = self.dot(v1, v2)
        mag1 = self.dot(v1, v1)**0.5 + 1e-5
        mag2 = self.dot(v2, v2)**0.5 + 1e-5
        cos_ = dot_prod/mag1/mag2 
        angle = math.acos(dot_prod/mag2/mag1)
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            ang_deg = 360 - ang_deg

        return ang_deg


    def eval_path(self, ensemble, path, prev_path):
        reach_per_model = []
        for k in range(ensemble.shape[0]):
            model = ensemble[k].squeeze(0)
            reachability = []    
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]

                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1

                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])
                    reachability.append(model[1,x,y]) # probability of occupancy
            reach_per_model.append(max(reachability))
        avg = torch.mean(torch.tensor(reach_per_model))
        std = torch.sqrt(torch.var(torch.tensor(reach_per_model)))
        path_len = len(path) / 100 # normalize by a pseudo max length
        #print(path_len)
        result = avg - self.options.a_1*std + self.options.a_2*path_len
        
        if prev_path:
            angle = (self.get_angle((path[0], path[min(self.options.reach_horizon,len(path))-1]), (prev_path[0], prev_path[min(self.options.reach_horizon,len(prev_path))-1]))) / 360.0
            result += self.options.a_3 * angle

        return result


    def eval_path_expl(self, ensemble, paths):
        # evaluate each path based on its average occupancy uncertainty
        #N, B, C, H, W = ensemble.shape # number of models, batch, classes, height, width
        ### Estimate the variance only of the occupied class (1) for each location # 1 x B x object_classes x grid_dim x grid_dim
        ensemble_occupancy_var = torch.var(ensemble[:,:,1,:,:], dim=0, keepdim=True).squeeze(0) # 1 x H x W
        path_sum_var = []
        for k in range(len(paths)):
            path = paths[k]
            path_var = []
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]
                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1
                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])          
                    path_var.append(ensemble_occupancy_var[0,x,y])
            path_sum_var.append( np.sum(np.asarray(path_var)) )
        return path_sum_var


    def get_rrt_goal(self, pose_coords, goal, grid, ensemble, prev_path):
        probability_map, indexes = torch.max(grid,dim=1)
        probability_map = probability_map[0]
        indexes = indexes[0]
        binarymap = (indexes == 1)
        start = [int(pose_coords[0][0][1]), int(pose_coords[0][0][0])]
        finish = [int(goal[0][0][1]), int(goal[0][0][0])]
        rrt_star = RRTStar(start=start, 
                           obstacle_list=None, 
                           goal=finish, 
                           rand_area=[0,binarymap.shape[0]], 
                           max_iter=self.options.rrt_max_iters,
                           expand_dis=self.options.expand_dis,
                           goal_sample_rate=self.options.goal_sample_rate,
                           connect_circle_dist=self.options.connect_circle_dist,
                           occupancy_map=binarymap)
        best_path = None
        
        path_dict = {'paths':[], 'value':[]} # visualizing all the paths
        if self.options.exploration:
            paths = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line, exploration=self.options.exploration, horizon=self.options.reach_horizon)
            ## evaluate each path on the exploration objective
            path_sum_var = self.eval_path_expl(ensemble, paths)
            path_dict['paths'] = paths
            path_dict['value'] = path_sum_var

            best_path_var = 0 # we need to select the path with maximum overall uncertainty
            for i in range(len(paths)):
                if path_sum_var[i] > best_path_var:
                    best_path_var = path_sum_var[i]
                    best_path = paths[i]

        else:
            best_path_reachability = float('inf')        
            for i in range(self.options.rrt_num_path):
                path = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line)
                if path:
                    if self.options.rrt_path_metric == "reachability":
                        reachability = self.eval_path(ensemble, path, prev_path)
                    elif self.options.rrt_path_metric == "shortest":
                        reachability = len(path)
                    path_dict['paths'].append(path)
                    path_dict['value'].append(reachability)
                    
                    if reachability < best_path_reachability:
                        best_path_reachability = reachability
                        best_path = path

        if best_path:
            best_path.reverse()
            last_node = min(len(best_path)-1, self.options.reach_horizon)
            return torch.tensor([[[int(best_path[last_node][1]), int(best_path[last_node][0])]]]).cuda(), best_path, path_dict
        return None, None, None

    def run_map_predictor(self, step_ego_grid_crops):

        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops.unsqueeze(0)}
        input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

        model_pred_output = {}
        ensemble_spatial_maps = []
        for n in range(self.options.ensemble_size):
            model_pred_output[n] = self.models_dict[n]['predictor_model'](input_batch)
            ensemble_spatial_maps.append(model_pred_output[n]['pred_maps_spatial'].clone())
        ensemble_spatial_maps = torch.stack(ensemble_spatial_maps) # N x B x T x C x cH x cW

        ### Estimate average predictions from the ensemble
        mean_ensemble_spatial = torch.mean(ensemble_spatial_maps, dim=0) # B x T x C x cH x cW
        return mean_ensemble_spatial, ensemble_spatial_maps


    def save_test_summaries(self, output):
        prefix = 'test/' + self.scene_id + '/'
        for k in output['metrics']:
            self.summary_writer.add_scalar(prefix + k, output['metrics'][k], self.step_count)
