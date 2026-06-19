import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, DA=False, flip_sign=False, rot=False, drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points
        self.reset()

    def reset(self):
        self.points_to_drop = None
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.remissions = np.zeros((0, 1), dtype=np.float32)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        self.proj_x = np.zeros((0, 1), dtype=np.int32)
        self.proj_y = np.zeros((0, 1), dtype=np.int32)
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

    def open_scan(self, filename):
        self.reset()
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        if self.drop_points:
            n_drop = int(len(points) * self.drop_points)
            if n_drop > 0:
                self.points_to_drop = np.random.randint(0, len(points) - 1, n_drop)
                points = np.delete(points, self.points_to_drop, axis=0)
                remissions = np.delete(remissions, self.points_to_drop)
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        self.points = points
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            self.points[:, 0] += random.uniform(-5, 5)
            self.points[:, 1] += random.uniform(-3, 3)
            self.points[:, 2] += random.uniform(-1, 0)
        if self.rot:
            from scipy.spatial.transform import Rotation as R
            euler_angle = np.random.normal(0, 90, 1)[0]
            r = R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix()[0]
            self.points = self.points.dot(r.T)
        self.remissions = remissions if remissions is not None else np.zeros((points.shape[0]), dtype=np.float32)
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)
        depth = np.linalg.norm(self.points, 2, axis=1)
        scan_x, scan_y, scan_z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov
        proj_x = np.floor(proj_x * self.proj_W)
        proj_x = np.minimum(self.proj_W - 1, np.maximum(0, proj_x)).astype(np.int32)
        self.proj_x = np.copy(proj_x)
        proj_y = np.floor(proj_y * self.proj_H)
        proj_y = np.minimum(self.proj_H - 1, np.maximum(0, proj_y)).astype(np.int32)
        self.proj_y = np.copy(proj_y)
        self.unproj_range = np.copy(depth)
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth, indices, points, remission, proj_y, proj_x = depth[order], indices[order], self.points[order], self.remissions[order], proj_y[order], proj_x[order]
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

class SemLaserScan(LaserScan):
    EXTENSIONS_LABEL = ['.label']
    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, DA=False, flip_sign=False, rot=False, drop_points=False):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down, DA, flip_sign, rot, drop_points)
        self.reset()
        if sem_color_dict:
            max_sem_key = max(sem_color_dict.keys()) + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            self.sem_color_lut = np.random.uniform(low=0.0, high=1.0, size=(300, 3))
            self.sem_color_lut[0] = 0.1

    def reset(self):
        super(SemLaserScan, self).reset()
        self.sem_label = np.zeros((0, 1), dtype=np.int32)
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

    def open_label(self, filename):
        label = np.fromfile(filename, dtype=np.int32).reshape((-1))
        if self.points_to_drop is not None:
            label = np.delete(label, self.points_to_drop)
        self.set_label(label)

    def set_label(self, label):
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF
        else:
            raise ValueError("Scan and Label don't contain same number of points")
        if self.project:
            self.do_label_projection()

    def do_label_projection(self):
        mask = self.proj_idx >= 0
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]

class SemanticKitti(Dataset):
    def __init__(self, root, sequences, labels, color_map, learning_map, learning_map_inv, sensor, max_points=150000, gt=True, transform=False):
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        self.scan_files = []
        self.label_files = []

        for seq_idx in self.sequences:
            seq = '{0:02d}'.format(int(seq_idx))
            if not os.path.exists(os.path.join(self.root, seq)):
                seq = '{0:04d}'.format(int(seq_idx))
            
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            
            scans = sorted([os.path.join(scan_path, f) for f in os.listdir(scan_path) if f.endswith('.bin')])
            self.scan_files.extend(scans)
            if self.gt:
                labels = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.label')])
                self.label_files.extend(labels)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        DA, flip_sign, rot, drop_points = False, False, False, False
        if self.transform and random.random() > 0.5:
            DA, flip_sign, rot, drop_points = random.random()>0.5, random.random()>0.5, random.random()>0.5, random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(self.color_map, project=True, H=self.sensor_img_H, W=self.sensor_img_W, fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down, DA=DA, flip_sign=flip_sign, rot=rot, drop_points=drop_points)
            scan.open_scan(scan_file)
            scan.open_label(self.label_files[index])
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)
        else:
            scan = LaserScan(project=True, H=self.sensor_img_H, W=self.sensor_img_W, fov_up=self.sensor_fov_up, fov_down=self.sensor_fov_down, DA=DA, flip_sign=flip_sign, rot=rot, drop_points=drop_points)
            scan.open_scan(scan_file)

        unproj_n_points = scan.points.shape[0]
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_range = torch.from_numpy(scan.proj_range)
        proj_xyz = torch.from_numpy(scan.proj_xyz)
        proj_remission = torch.from_numpy(scan.proj_remission)
        
        proj = torch.cat([proj_range.unsqueeze(0), proj_xyz.permute(2, 0, 1), proj_remission.unsqueeze(0)])
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()
        
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone() * proj_mask
        else:
            proj_labels = torch.zeros_like(proj_mask)

        return proj, proj_mask, proj_labels, unproj_n_points

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        maxkey = max(mapdict.keys())
        lut = np.zeros(maxkey + 100, dtype=np.int32)
        for key, data in mapdict.items():
            lut[key] = data
        return lut[label]

class Parser:
    def __init__(self, root, train_sequences, valid_sequences, labels, color_map, learning_map, learning_map_inv, sensor, max_points, batch_size, workers, gt=True, shuffle_train=True):
        self.nclasses = len(learning_map_inv)
        
        train_ds = SemanticKitti(root, train_sequences, labels, color_map, learning_map, learning_map_inv, sensor, max_points, gt=gt, transform=True)
        valid_ds = SemanticKitti(root, valid_sequences, labels, color_map, learning_map, learning_map_inv, sensor, max_points, gt=gt, transform=False)
        
        self.trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=workers, pin_memory=True, drop_last=True)
        self.validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    def get_train_set(self): return self.trainloader
    def get_valid_set(self): return self.validloader
    def get_n_classes(self): return self.nclasses