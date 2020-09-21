from __future__ import print_function

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

import os
import numpy as np
import scipy.spatial as sp

from Pointfilter_Utils import pca_alignment


##################################New Dataloader Class###########################

def my_collate(batch):

    batch = list(filter(lambda x : x is not None, batch))

    return default_collate(batch)

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.05, points_per_patch=500,
                 seed=None, train_state='train', shape_name=None):

        self.root = root
        self.shapes_list_file = shapes_list_file

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.train_state = train_state

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 10 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []

        self.shape_names = []
        if self.train_state == 'evaluation' and shape_name is not None:
            noise_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
            noise_kdtree = sp.cKDTree(noise_pts)
            self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})
            self.shape_patch_count.append(noise_pts.shape[0])
            bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)
        elif self.train_state == 'train':
            with open(os.path.join(self.root, self.shapes_list_file)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % shape_name)
                if shape_ind % 6 == 0:
                    gt_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    gt_normal = np.load(os.path.join(self.root, shape_name + '_normal.npy'))
                    gt_kdtree = sp.cKDTree(gt_pts)
                    self.gt_shapes.append({'gt_pts': gt_pts, 'gt_normal': gt_normal, 'gt_kdtree': gt_kdtree})
                    self.noise_shapes.append({'noise_pts': gt_pts, 'noise_kdtree': gt_kdtree})
                    noise_pts = gt_pts
                else:
                    noise_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    noise_kdtree = sp.cKDTree(noise_pts)
                    self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})

                self.shape_patch_count.append(noise_pts.shape[0])
                bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
                self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def patch_sampling(self, patch_pts):

        if patch_pts.shape[0] > self.points_per_patch:

            sample_index = np.random.choice(range(patch_pts.shape[0]), self.points_per_patch, replace=False)

        else:

            sample_index = np.random.choice(range(patch_pts.shape[0]), self.points_per_patch)

        return sample_index

    def __getitem__(self, index):

        # find shape that contains the point with given global index

        shape_ind, patch_ind = self.shape_index(index)
        noise_shape = self.noise_shapes[shape_ind]
        patch_radius = self.patch_radius_absolute[shape_ind]

        # For noise_patch
        noise_patch_idx = noise_shape['noise_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind], patch_radius)

        if len(noise_patch_idx) < 3:
            return None

        noise_patch_pts  = noise_shape['noise_pts'][noise_patch_idx] - noise_shape['noise_pts'][patch_ind]
        noise_patch_pts, noise_patch_inv = pca_alignment(noise_patch_pts)
        noise_patch_pts /= patch_radius

        noise_sample_idx = self.patch_sampling(noise_patch_pts)
        noise_patch_pts  = noise_patch_pts[noise_sample_idx]

        support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
        support_radius = np.expand_dims(support_radius, axis=0)

        if self.train_state == 'evaluation':

            return torch.from_numpy(noise_patch_pts), torch.from_numpy(noise_patch_inv), \
                   noise_shape['noise_pts'][patch_ind]

        # For gt_patch
        gt_shape = self.gt_shapes[shape_ind // 6]
        gt_patch_idx = gt_shape['gt_kdtree'].query_ball_point(noise_shape['noise_pts'][patch_ind], patch_radius)

        if len(gt_patch_idx) < 3:
            return None

        gt_patch_pts    = gt_shape['gt_pts'][gt_patch_idx]
        gt_patch_pts   -= noise_shape['noise_pts'][patch_ind]
        gt_patch_pts   /= patch_radius
        gt_patch_pts    = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_pts.T)).T

        gt_patch_normal = gt_shape['gt_normal'][gt_patch_idx]
        gt_patch_normal = np.array(np.linalg.inv(noise_patch_inv) * np.matrix(gt_patch_normal.T)).T

        gt_sample_idx   = self.patch_sampling(gt_patch_pts)
        gt_patch_pts    = gt_patch_pts[gt_sample_idx]
        gt_patch_normal = gt_patch_normal[gt_sample_idx]

        return torch.from_numpy(noise_patch_pts), torch.from_numpy(gt_patch_pts), \
               torch.from_numpy(gt_patch_normal), torch.from_numpy(support_radius)

    def __len__(self):
        return sum(self.shape_patch_count)


    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if (index >= shape_patch_offset) and (index < shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind


