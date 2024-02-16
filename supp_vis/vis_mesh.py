import open3d as o3d
import trimesh
import numpy as np
from PIL import Image
import os
import json

fo = open("/hdd/gibson_habitat/single_floor_split.txt", "r")
with open('/hdd/gibson_habitat/gibson.json', 'r') as f:
    data = json.load(f)
for line in fo.readlines():
    scene_id = line.strip()
    mesh_est_file = "/media/yan/Passport/CVPR2023/logs/single_floor/gibson/" + scene_id + "/step_0/mesh.ply"
    mesh_gt_file = "/hdd/gibson_v2_selected/" + scene_id + "/" + scene_id + "_mesh.ply"
    texture_file = "/hdd/gibson_v2_selected/" + scene_id + "/" + scene_id + "_mesh_texture.small.jpg"
    if os.path.isfile(mesh_est_file) and os.path.isfile(texture_file):
        for i in range(len(data)):
            if data[i]['id'] == scene_id:
                area = data[i]['stats']['area']
                floors = data[i]['stats']['floor']
                room = data[i]['stats']['room']
        print("============= ", scene_id, ", area: ", area, ", room number: ", room)

        im =  Image.open(texture_file)
        tex = trimesh.visual.TextureVisuals(image=im)

        mesh_est = o3d.io.read_triangle_mesh(mesh_est_file)
        T_rot = np.eye(4)
        T_rot[1, 1], T_rot[2, 2] = 0, 0
        T_rot[1, 2], T_rot[2, 1] = -1, 1
        T_rot[1, 3] = 1.25
        T_wc = np.linalg.inv(T_rot)
        mesh_est.transform(T_wc)

        # tri_mesh_est = trimesh.load(mesh_est_file, force='mesh')
        # tri_mesh_est.apply_transform(T_wc)
        # tri_mesh_est.visual = tri_mesh_est.visual.to_texture()
        # tri_mesh_est.visual.texture = tex
        # print(tri_mesh_est.visual)
        # tri_mesh_est.show()

        mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_file)
        tri_mesh_gt = trimesh.load(mesh_gt_file)
        tri_mesh_gt.visual.texture=tex
        tri_mesh_gt.visual = tri_mesh_gt.visual.to_color()
        mesh_gt.vertices = o3d.utility.Vector3dVector(tri_mesh_gt.vertices)
        mesh_gt.triangles = o3d.utility.Vector3iVector(tri_mesh_gt.faces)
        mesh_gt.vertex_colors = o3d.utility.Vector3dVector(tri_mesh_gt.visual.vertex_colors[:,:3]/255.)
        #o3d.visualization.draw_geometries([mesh_est, mesh_gt])
        mesh_tree = o3d.geometry.KDTreeFlann(mesh_gt)
        for i in range(np.array(mesh_est.vertices).shape[0]):
            [k, idx, _] = mesh_tree.search_knn_vector_3d(mesh_est.vertices[i], 1)
            mesh_est.vertex_colors[i] = mesh_gt.vertex_colors[idx[0]]
        o3d.visualization.draw_geometries([mesh_gt])
        o3d.visualization.draw_geometries([mesh_est])
    else:
        print(scene_id, "does not has .ply file")

#scene_id = "Greigsville"
