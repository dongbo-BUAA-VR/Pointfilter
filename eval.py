import os
import sys
import torch
import numpy as np
from Pointfilter_Network_Architecture import pointfilternet
from Pointfilter_DataLoader import PointcloudPatchDataset
from Pointfilter_Utils import parse_arguments


def eval(opt):

    with open(os.path.join(opt.testset, 'test.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    if not os.path.exists(parameters.save_dir):
        os.makedirs(parameters.save_dir)

    for shape_id, shape_name in enumerate(shape_names):
        print(shape_name)
        original_noise_pts = np.load(os.path.join(opt.testset, shape_name + '.npy'))
        np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_0.npy'), original_noise_pts.astype('float32'))

        for eval_index in range(opt.eval_iter_nums):
            print(eval_index)
            test_dataset = PointcloudPatchDataset(
                root=opt.save_dir,
                shape_name=shape_name + '_pred_iter_' + str(eval_index),
                patch_radius=opt.patch_radius,
                train_state='evaluation')
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers))

            pointfilter_eval = pointfilternet().cuda()
            model_filename = os.path.join(parameters.eval_dir, 'model_full_ae.pth')
            checkpoint = torch.load(model_filename)
            pointfilter_eval.load_state_dict(checkpoint['state_dict'])
            pointfilter_eval.cuda()
            pointfilter_eval.eval()

            patch_radius = test_dataset.patch_radius_absolute
            pred_pts = np.empty((0, 3), dtype='float32')
            for batch_ind, data_tuple in enumerate(test_dataloader):

                noise_patch, noise_inv, noise_disp = data_tuple
                noise_patch = noise_patch.float().cuda()
                noise_inv = noise_inv.float().cuda()

                noise_patch = noise_patch.transpose(2, 1).contiguous()
                predict = pointfilter_eval(noise_patch)
                predict = predict.unsqueeze(2)
                predict = torch.bmm(noise_inv, predict)

                pred_pts = np.append(pred_pts,
                                     np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                     axis=0)
            np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1) + '.npy'),
                    pred_pts.astype('float32'))



if __name__ == '__main__':

    parameters = parse_arguments()
    parameters.testset = './Dataset/Test'
    parameters.eval_dir = './Summary/pre_train_model/'
    parameters.batchSize = 64
    parameters.workers = 4
    parameters.save_dir = './Dataset/Results/'
    parameters.eval_iter_nums = 5
    parameters.patch_radius = 0.05
    eval(parameters)





