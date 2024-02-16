import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("/hdd/matterport/v1/tasks/17DRP5sb8fy/17DRP5sb8fy_semantic.ply")
mesh.get_max_bound()[2]-0.8
mesh.get_min_bound()
filtered_mask = (np.array(mesh.vertices)[:,2] > (mesh.get_max_bound()[2]-0.2))
print(filtered_mask)
mesh.remove_vertices_by_mask(filtered_mask)
o3d.visualization.draw_geometries([mesh])
pcd = mesh.sample_points_uniformly(number_of_points=500)
o3d.visualization.draw_geometries([pcd])