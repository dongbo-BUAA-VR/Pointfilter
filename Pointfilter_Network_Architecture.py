from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class pointfilter_encoder(nn.Module):
    def __init__(self, input_dim=3, patch_nums=500, sym_op='max'):
        super(pointfilter_encoder, self).__init__()
        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.activate = nn.ReLU()

    def forward(self, x):

        x = self.activate(self.bn1(self.conv1(x)))
        net1 = x  # 64

        x = self.activate(self.bn2(self.conv2(x)))
        net2 = x  # 128

        x = self.activate(self.bn3(self.conv3(x)))
        net3 = x  # 256

        x = self.activate(self.bn4(self.conv4(x)))
        net4 = x  # 512

        x = self.activate(self.bn5(self.conv5(x)))
        net5 = x  # 1024

        if self.sym_op == 'sum':
            x = torch.sum(x, dim=-1)
        else:
            x, index = torch.max(x, dim=-1)

        return x#, index


class pointfilter_decoder(nn.Module):
    def __init__(self):
        super(pointfilter_decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout_1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout_2(x)
        x = torch.tanh(self.fc3(x))

        return x

class pointfilternet(nn.Module):
    def __init__(self, input_dim=3, patch_nums=500, sym_op='max'):
        super(pointfilternet, self).__init__()

        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.encoder = pointfilter_encoder(self.input_dim, self.patch_nums, self.sym_op)
        self.decoder = pointfilter_decoder()

    def forward(self, x):
        x = self.encoder(x)

        #encoder_feature = x

        x = self.decoder(x)

        return x#, encoder_feature


if __name__ == '__main__':


    model = pointfilternet().cuda()

    print(model)

