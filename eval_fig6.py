import scipy.spatial as sp
import numpy as np
import torch
import os
from Customer_Module.chamfer_distance.dist_chamfer import chamferDist
from plyfile import PlyData, PlyElement
nnd = chamferDist()

def npy2ply(filename, save_filename):
    pts = np.load(filename)
    vertex = [tuple(item) for item in pts]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(save_filename)

def Eval_With_Charmfer_Distance():
    print('Errors under Chamfer Distance')
    for shape_id, shape_name in enumerate(shape_names):
        gt_pts = np.load(os.path.join('./Dataset/Test', shape_name[:-6] + '.npy'))
        pred_pts = np.load(os.path.join('./Dataset/Results', shape_name + '_pred_iter_2.npy'))
        with torch.no_grad():
            gt_pts_cuda = torch.from_numpy(np.expand_dims(gt_pts, axis=0)).cuda().float()
            pred_pts_cuda = torch.from_numpy(np.expand_dims(pred_pts, axis=0)).cuda().float()
            dist1, dist2 = nnd(pred_pts_cuda, gt_pts_cuda)
            chamfer_errors = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
            print('%12s  %.3f' % (models_name[shape_id], round(chamfer_errors.item() * 100000, 3)))

def Eval_With_Mean_Square_Error():
    print('Errors under Mean Square Error')
    for shape_id, shape_name in enumerate(shape_names):
        gt_pts = np.load(os.path.join('./Dataset/Test', shape_name[:-6] + '.npy'))
        gt_pts_tree = sp.cKDTree(gt_pts)
        pred_pts = np.load(os.path.join('./Dataset/Results', shape_name + '_pred_iter_2.npy'))
        pred_dist, _ = gt_pts_tree.query(pred_pts, 10)
        print('%12s  %.3f' % (models_name[shape_id], round(pred_dist.mean() * 1000, 3)))

def File_Conversion():
    for shape_id, shape_name in enumerate(shape_names):
        npy2ply(os.path.join('./Dataset/Results', shape_name + '_pred_iter_2.npy'),
                os.path.join('./Dataset/Results', shape_name + '_pred_iter_2.ply'))

if __name__ == '__main__':

    with open(os.path.join('./Dataset/Test', 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    models_name = ['Boxunion',
                   'Cube',
                   'Fandisk',
                   'Tetrahedron']

    File_Conversion()

    Eval_With_Charmfer_Distance()
    Eval_With_Mean_Square_Error()
