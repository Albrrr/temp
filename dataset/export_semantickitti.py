import os

import fire
import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_io import load_bin_file

class KittiConverter:
    def __init__(self, nusc_dir: str, nusc_skitti_dir: str, lidar_name: str = 'LIDAR_TOP', nusc_version: str = 'v1.0-mini',):
        self.nusc_skitti_dir = os.path.expanduser(nusc_skitti_dir)
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version

        if not os.path.isdir(self.nusc_skitti_dir):
            os.makedirs(self.nusc_skitti_dir)

        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_dir)

    def nuscenes_gt_to_semantickitti(self):
        """
        Converts nuScenes GT panoptic annotations to SemanticKITTI format.
        """
        nu_to_kitti_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        nu_to_kitti_lidar_inv = nu_to_kitti_lidar.inverse

        seqs_folder = os.path.join(self.nusc_skitti_dir, 'sequences')

        for scene_idx, scene in enumerate(self.nusc.scene):
            print(f'Converting scene {scene_idx} out of {len(self.nusc.scene)}: {scene["name"]}')

            name_idx = int(scene['name'][6:])

            seq_folder = os.path.join(seqs_folder, f'{name_idx:04d}')
            if not os.path.exists(seq_folder):
                os.makedirs(seq_folder)

            velo_folder = os.path.join(seq_folder, 'velodyne')
            label_folder = os.path.join(seq_folder, 'labels')
            if not os.path.exists(velo_folder):
                os.makedirs(velo_folder)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            calib_file = os.path.join(seq_folder, 'calib.txt')
            pose_file = os.path.join(seq_folder, 'poses.txt')
            times_file = os.path.join(seq_folder, 'times.txt')
            calib_f = open(calib_file, 'w')
            pose_f = open(pose_file, 'w')
            times_f = open(times_file, 'w')

            sample_token = scene['first_sample_token']

            sample = self.nusc.get('sample', sample_token)
            lidar_data_token = sample['data'][self.lidar_name]
            lidar_data = self.nusc.get('sample_data', lidar_data_token)
            cali_lidar = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            ego_to_lidar = transform_matrix(cali_lidar['translation'], Quaternion(cali_lidar['rotation']), inverse=False)

            ego_to_lidar_kitti = np.dot(ego_to_lidar, nu_to_kitti_lidar.transformation_matrix )
            ego_to_lidar_kitti_flat = ego_to_lidar_kitti[:3].reshape(-1)

            calib_f.write('P0: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P1: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P2: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P3: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('Tr: ' + ' '.join([str(x) for x in ego_to_lidar_kitti_flat]) + '\n')
            calib_f.close()

            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

            ego_pose_kitti_first = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                

            token_idx = 0
            while True:
                sample = self.nusc.get('sample', sample_token)

                lidar_data_token = sample['data'][self.lidar_name]

                lidar_panoptic = self.nusc.get('lidarseg', lidar_data_token)

                lidar_panoptic_anno = load_bin_file(os.path.join(self.nusc.dataroot, lidar_panoptic['filename']), type='lidarseg')
                semantic_anno = np.uint32(lidar_panoptic_anno)
                print("semantic_anno unique:", np.unique(semantic_anno))
                semantic_anno.tofile(os.path.join(label_folder, f'{token_idx:06}.label'))

                lidar_data = self.nusc.get('sample_data', lidar_data_token)
 
                lidar_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
                lidar_pc.rotate(nu_to_kitti_lidar_inv.rotation_matrix)

                lidar_pc.points[:4, :].T.astype(np.float32).tofile(os.path.join(velo_folder, f'{token_idx:06}.bin'))

                ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

                ego_pose_kitti = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                ego_pose_kitti_first_to_curr = np.dot(np.linalg.inv(ego_pose_kitti_first), ego_pose_kitti)
                ego_pose_kitti_flat = ego_pose_kitti_first_to_curr[:3].reshape(-1)
                pose_f.write(' '.join([str(x) for x in ego_pose_kitti_flat]) + '\n')

                time_second = lidar_data['timestamp']/1e6
                if token_idx == 0:
                    time_start = time_second
                times_f.write('{:.6e}\n'.format(time_second-time_start))

                token_idx += 1
                if sample['next'] == '':
                    break
                else:
                    sample_token = sample['next']
                    
            pose_f.close()
            times_f.close()
            print(f'Finish processing scene {scene_idx} with {token_idx} samples.', flush=True)
            
        print('Finish processing all scenes.')
        return

if __name__ == '__main__':
    fire.Fire(KittiConverter)