from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os, sys
# import pika
from scipy.spatial.transform import Rotation as R

# import needed only when running with ROS
try:
    from activeINR.ros_utils import node
except ImportError:
    print('Did not import ROS node.')

# Consume RGBD + pose data from ROS node
class HabitatSimulator(Dataset):
    def __init__(
        self,
        traj_file=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        max_length = 1000
    ):
        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.noisy_depth = noisy_depth
        crop = False
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.max_length = max_length

    def __len__(self):
        return self.max_length

    def process(self, obs):
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

        if self.rgb_transform:
            image = self.rgb_transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        sample = {
            "image": image,
            "depth": depth,
            "T": Twc_rl,
        }

        # from PIL import Image
        # rgb_img = Image.fromarray(sample["image"], mode="RGB")
        # depth_obs = sample["depth"] / np.amax(sample["depth"])
        # depth_img = Image.fromarray((np.squeeze(depth_obs) * 255).astype(np.uint8), mode="L")
        # arr = [rgb_img, depth_img]
        # n = len(arr)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 8))
        # for i, data in enumerate(arr):
        #     ax = plt.subplot(1, n, i + 1)
        #     ax.axis('off')
        #     plt.imshow(data)
        # plt.show()

        return sample

class ReplicaDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=".jpg",
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext
        self.noisy_depth = noisy_depth

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = f"{idx:06}"  # int variable
        if self.noisy_depth:
            depth_file = os.path.join(self.root_dir, "ndepth" + s + ".png")
        else:
            depth_file = os.path.join(self.root_dir, "depth" + s + ".png")
        rgb_file = os.path.join(self.root_dir, "frame" + s + self.col_ext)

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class ScanNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file,
        rgb_transform=None,
        depth_transform=None,
        col_ext=None,
        noisy_depth=None,
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "frames", "color/")
        self.depth_dir = os.path.join(root_dir, "frames", "depth/")
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.depth_dir + str(idx) + ".png"
        rgb_file = self.rgb_dir + str(idx) + self.col_ext

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample

# class for franka tabletop data with realsense + calibrated end-effector poses 
class RealsenseFrankaOffline(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file,
        rgb_transform=None,
        depth_transform=None,
        col_ext=None,
        noisy_depth=None,
        distortion_coeffs=None,
        camera_matrix=None,
    ):
        abspath = os.path.abspath(sys.argv[0])
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.depth_dir = os.path.join(root_dir, "depth")
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file)
            self.Ts = self.Ts[:, 1:].reshape(-1, 4, 4)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = os.path.join(self.depth_dir, str(idx).zfill(5) + ".npy")
        rgb_file = os.path.join(self.rgb_dir, str(idx).zfill(5) + self.col_ext)

        depth = np.load(depth_file)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample

class SceneCache(Dataset):
    def __init__(
        self,
        dataset_format,
        root_dir,
        traj_file,
        keep_ixs=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=".jpg",
        distortion_coeffs=None,
        camera_matrix=None,
    ):

        self.dataset_format = dataset_format
        self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.samples = []

        if keep_ixs is not None:
            keep_ixs.sort()
        self.keep_ixs = keep_ixs

        print("Loading scene cache dataset for evaluation...")
        for idx in range(self.Ts.shape[0]):
            if keep_ixs is not None:
                if idx not in keep_ixs:
                    continue

            if dataset_format == "replicaCAD":
                s = f"{idx:06}"  # int variable
                if noisy_depth:
                    depth_file = self.root_dir + "/ndepth" + s + ".png"
                else:
                    depth_file = self.root_dir + "/depth" + s + ".png"
                rgb_file = self.root_dir + "/frame" + s + col_ext
            elif dataset_format == "ScanNet":
                depth_file = root_dir + "/frames/depth/" + str(idx) + ".png"
                rgb_file = root_dir + "/frames/color/" + str(idx) + col_ext

            depth = cv2.imread(depth_file, -1)
            image = cv2.imread(rgb_file)

            if self.rgb_transform:
                image = self.rgb_transform(image)

            if self.depth_transform:
                depth = self.depth_transform(depth)

            self.samples.append((image, depth, self.Ts[idx]))

        self.samples = np.array(self.samples)
        print("Len cached dataset", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_all(self):
        idx = np.arange(self.__len__())

        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))

        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }

        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.keep_ixs is not None:
            idx = [x for x in idx if x in self.keep_ixs]
            idx = np.array([np.where(self.keep_ixs == x)[0][0] for x in idx])

        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))

        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }

        return sample


# Consume RGBD + pose data from ROS node
class ROSSubscriber(Dataset):
    def __init__(
        self,
        extrinsic_calib=None,
        root_dir=None,
        traj_file=None,
        keep_ixs=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        col_ext=None,
        distortion_coeffs=None,
        camera_matrix=None,
    ):
        crop = False
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.distortion_coeffs = np.array(distortion_coeffs)
        self.camera_matrix = camera_matrix

        torch.multiprocessing.set_start_method('spawn', force=True)
        self.queue = torch.multiprocessing.Queue(maxsize=1)

        if extrinsic_calib is not None:
            process = torch.multiprocessing.Process(
                target=node.activeINRFrankaNode,
                args=(self.queue, crop, extrinsic_calib),
            ) # subscribe to franka poses 
        else:
            process = torch.multiprocessing.Process(
                target=node.activeINRNode,
                args=(self.queue, crop),
            ) # subscribe to ORB-SLAM backend

        process.start()

    def __len__(self):
        return 1000000000

    def __getitem__(self, idx):
        data = None
        while data is None:
            data = node.get_latest_frame(self.queue)

            if data is not None:
                image, depth, Twc = data

                if self.rgb_transform:
                    image = self.rgb_transform(image)
                if self.depth_transform:
                    depth = self.depth_transform(depth)

                    # undistort depth, using nn rather than linear interpolation
                    img_size = (depth.shape[1], depth.shape[0])
                    map1, map2 = cv2.initUndistortRectifyMap(
                        self.camera_matrix, self.distortion_coeffs, np.eye(3),
                        self.camera_matrix, img_size, cv2.CV_32FC1)
                    depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)

                sample = {
                    "image": image,
                    "depth": depth,
                    "T": Twc,
                }
                return sample