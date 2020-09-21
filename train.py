
# coding=utf-8

from __future__ import print_function
from tensorboardX import SummaryWriter
from Pointfilter_Network_Architecture import pointfilternet
from Pointfilter_DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from Pointfilter_Utils  import parse_arguments, adjust_learning_rate, compute_bilateral_loss_with_repulsion

import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True

def train(opt):
    if not os.path.exists(opt.summary_dir):
        os.makedirs(opt.summary_dir)
    if not os.path.exists(opt.network_model_dir):
        os.makedirs(opt.network_model_dir)
    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    denoisenet = pointfilternet().cuda()
    optimizer = optim.SGD(
        denoisenet.parameters(),
        lr=opt.lr,
        momentum=opt.momentum)
    train_writer = SummaryWriter(opt.summary_dir)
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            denoisenet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    train_dataset = PointcloudPatchDataset(
        root=opt.trainset,
        shapes_list_file='train.txt',
        patch_radius=0.05,
        seed=opt.manualSeed,
        train_state='train')
    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=8000,
        seed=opt.manualSeed,
        identical_epochs=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        shuffle=(train_datasampler is None),
        collate_fn=my_collate,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        pin_memory=True)
    num_batch = len(train_dataloader)
    for epoch in range(opt.start_epoch, opt.nepoch):
        adjust_learning_rate(optimizer, epoch, opt)
        print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
        for batch_ind, data_tuple in enumerate(train_dataloader):
            denoisenet.train()
            optimizer.zero_grad()
            noise_patch, gt_patch, gt_normal, support_radius = data_tuple
            noise_patch = noise_patch.float().cuda(non_blocking=True)
            gt_patch = gt_patch.float().cuda(non_blocking=True)
            gt_normal = gt_normal.float().cuda(non_blocking=True)
            support_radius = opt.support_multiple * support_radius
            support_radius = support_radius.float().cuda(non_blocking=True)
            support_angle =  (opt.support_angle / 360) * 2 * np.pi
            noise_patch = noise_patch.transpose(2, 1).contiguous()
            pred_pts = denoisenet(noise_patch)
            loss = 100 * compute_bilateral_loss_with_repulsion(pred_pts, gt_patch, gt_normal,
                                                               support_radius, support_angle, opt.repulsion_alpha)
            loss.backward()
            optimizer.step()
            print('[%d: %d/%d] train loss: %f\n' % (epoch, batch_ind, num_batch, loss.item()))
            train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)
        checpoint_state = {
            'epoch': epoch + 1,
            'state_dict': denoisenet.state_dict(),
            'optimizer': optimizer.state_dict()}

        if epoch == (opt.nepoch - 1):

            torch.save(checpoint_state, '%s/model_full_ae.pth' % opt.network_model_dir)

        if epoch % opt.model_interval == 0:

            torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (opt.network_model_dir, epoch))

if __name__ == '__main__':
    parameters = parse_arguments()
    parameters.trainset = './Dataset/Train'
    parameters.summary_dir = './Summary/Train/logs'
    parameters.network_model_dir = './Summary/Train'
    parameters.batchSize = 64
    parameters.lr = 1e-4
    parameters.workers = 4
    parameters.nepoch = 50
    print(parameters)
    train(parameters)