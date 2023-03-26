import os
import torch
import numpy as np
from data.base_dataset import BaseDataset
from PIL import Image
import cv2
from torchvision import transforms as T
from models.utils import *
from utils.colmap import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import json
import imageio
from scipy.spatial.transform import Rotation as R

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


class AvtDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--num_masks', type=int, default=-1, help="Number of gt masks used for training, -1 for using all available masks.")
        parser.set_defaults(white_bkgd=False, noise_std=1.)
        return parser

    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode
        self.root_dir = opt.dataset_root
        self.split = mode
        assert self.split in ['train', 'val', 'test', 'test_train', 'test_val']
        self.img_wh = opt.img_wh
        self.val_num = 1
        self.patch_size = opt.patch_size
        self.white_back = opt.white_bkgd
        
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
            
        for frame in meta['frames'][::1]:
            file_path = frame['file_path']
            if '.png' not in file_path:
                file_path = file_path + '.png'
            fname = os.path.join(self.root_dir, file_path)
            img = imageio.imread(fname)
            imgs.append(img[..., :3])
            T_cam_to_world = np.array(frame['transform_matrix'])
            if True:
                T_cam_face_to_world = T_cam_to_world

                T_img_to_cam_face = np.eye(4) 
                T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
                T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face

            poses.append(T_cam_to_world)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        
        H, W = imgs[0].shape[:2]

        fx = meta['fx']
        fy = meta['fy']
        cx = meta['cx']
        cy = meta['cy']

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

        focal = fx

        self.focal = focal
        self.imgs = imgs
        self.train_idxs = [1,2]
        self.val_idxs = [4]

        poses = poses[:,:3,:] # (N_images, 3, 4) cam2world matrices
        self.poses = poses

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, self.opt.use_pixel_centers) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            # split data to patches instead of individual pixels
            self.n_img_patches = (self.img_wh[0] - self.patch_size + 1) * (self.img_wh[1] - self.patch_size + 1)
            self.n_patches = self.n_img_patches * (len(self.train_idxs) - 1)
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_masks_valid = []
            count = 0            
            for i, image_path in enumerate(imgs):
                if i not in self.train_idxs: # exclude the val images
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = imgs[i]
                assert img.shape[0]*self.img_wh[0] == img.shape[1]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = self.transform(img)
                img = img.reshape((-1, 3))
                self.all_rgbs += [img]

                # load gt masks if exist
                mask = torch.zeros_like(img)[:,[0]]
                self.all_masks += [mask]
                self.all_masks_valid += [torch.zeros_like(mask)]

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                self.focal, 1.0, rays_o, rays_d)
                                    # near plane is always at 1.0
                                    # near and far in NDC are always 0 and 1
                                    # See https://github.com/bmild/nerf/issues/34

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             rays_d],
                                             1)] # (h*w, 11)
                                 
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_masks = torch.cat(self.all_masks, 0)
            self.all_masks_valid = torch.cat(self.all_masks_valid, 0)          
        
        elif self.split == 'val':
            print('val image is', self.val_idxs[0])
            self.val_idx = self.val_idxs[0]

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train') or self.split.endswith('val'): # test on training set
                self.poses_test = self.poses
            else:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays) // self.patch_size**2
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_train':
            return len(self.train_idxs)
        if self.split == 'test_val':
            return len(self.val_idxs)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            if self.patch_size == 1:
                sample = {'rays': self.all_rays[idx],
                        'rgbs': self.all_rgbs[idx],
                        'masks': self.all_masks[idx],
                        'masks_valid': self.all_masks_valid[idx]}
            else:
                i_patch = torch.randint(high=self.n_patches, size=(1,))[0].item()
                i_img, i_patch = i_patch // self.n_img_patches, i_patch % self.n_img_patches
                row, col = i_patch // (self.img_wh[0] - self.patch_size + 1), i_patch % (self.img_wh[0] - self.patch_size + 1)
                start_idx = i_img * self.img_wh[0] * self.img_wh[1] + row * self.img_wh[0] + col
                idxs = start_idx + torch.cat([torch.arange(self.patch_size) + i * self.img_wh[0] for i in range(self.patch_size)])
                sample = {
                    'rays': self.all_rays[idxs],
                    'rgbs': self.all_rgbs[idxs],
                    'masks': self.all_masks[idxs],
                    'masks_valid': self.all_masks_valid[idxs]
                }
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses[self.train_idxs[idx]])
            elif self.split == 'test_val':
                c2w = torch.FloatTensor(self.poses[self.val_idxs[idx]])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            
            viewdir = rays_d

            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d)
            viewdir = rays_d

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1]),
                              viewdir],
                              1) # (h*w, 11)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split in ['val', 'test_train', 'test_val']:
                if self.split == 'val':
                    idx = self.val_idx
                if self.split == 'test_train':
                    idx = self.train_idxs[idx]
                if self.split == 'test_val':
                    idx = self.val_idxs[idx]
                img =  self.imgs[idx]
                img = self.transform(img)
                img = img.reshape((-1, 3))
                sample['rgbs'] = img

                mask = torch.zeros_like(img)[:,[0]]
                sample['masks'] = mask
                sample['masks_valid'] = torch.zeros_like(mask)

        return sample
