import torch
import torch.nn as nn
import torch.nn.functional as F

import simulatorHC.utils.rot as rot

class poseRegressionNet(nn.Module):
    # input: 2D-2D correspondences
    # output: Rotation, translation and scale
    def __init__(self,inplanes=7, planes = 96, final_feat=5):
        super(poseRegressionNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(planes*4)

        self.conv2 = nn.Conv1d(in_channels=planes*4,out_channels=planes*8,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(planes*8)

        self.conv3 = nn.Conv1d(in_channels=planes*8,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm1d(planes*16)


        self.conv31 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn31 = nn.BatchNorm1d(planes*16)
        self.conv32 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn32 = nn.BatchNorm1d(planes*16)
        self.conv33 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn33 = nn.BatchNorm1d(planes*16)
        self.conv34 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn34 = nn.BatchNorm1d(planes*16)
        self.conv35 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=1,padding=1)
        self.bn35 = nn.BatchNorm1d(planes*16)


        self.conv4 = nn.Conv1d(in_channels=planes*16,out_channels=planes*16,kernel_size=3,stride=2,padding=1)
        self.bn4 = nn.BatchNorm1d(planes*16)

        self.conv5 = nn.Conv1d(in_channels=planes*16,out_channels=planes*8,kernel_size=3,stride=2,padding=1)
        self.bn5 = nn.BatchNorm1d(planes*8)

        self.conv6 = nn.Conv1d(in_channels=planes*8,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn6 = nn.BatchNorm1d(planes*4)

        self.rot_conv1 = nn.Conv1d(in_channels=planes*4,out_channels=planes*2,kernel_size=1,stride=2)
        self.rot_bn1 = nn.BatchNorm1d(planes*2)
        self.rot_conv2 = nn.Conv1d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn2 = nn.BatchNorm1d(planes)
        
        self.tr_conv1 = nn.Conv1d(in_channels=planes*4,out_channels=planes*2,kernel_size=1,stride=2)
        self.tr_bn1 = nn.BatchNorm1d(planes*2)
        self.tr_conv2 = nn.Conv1d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn2 = nn.BatchNorm1d(planes)
        
        self.s_conv1 = nn.Conv1d(in_channels=planes*4,out_channels=planes*2,kernel_size=1,stride=2)
        self.s_bn1 = nn.BatchNorm1d(planes*2)
        self.s_conv2 = nn.Conv1d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.s_bn2 = nn.BatchNorm1d(planes)

        self.rot_drop = nn.Dropout1d(p=0.2)
        self.rot_pool = nn.AdaptiveAvgPool1d(output_size=final_feat)
        
        self.tr_drop = nn.Dropout1d(p=0.2)
        self.tr_pool = nn.AdaptiveAvgPool1d(output_size=final_feat)
        
        self.s_drop = nn.Dropout1d(p=0.2)
        self.s_pool = nn.AdaptiveAvgPool1d(output_size=final_feat)

        self.fct = nn.Linear(planes*final_feat,3) 
        self.fc6DR = nn.Linear(planes*final_feat,6)
        self.fcs = nn.Linear(planes*final_feat,1)

        nn.init.xavier_normal_(self.fct.weight,0.1)
        nn.init.xavier_normal_(self.fc6DR.weight,0.1)
        nn.init.xavier_normal_(self.fcs.weight,0.5)

    def forward(self,x:torch.Tensor):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv31(x)
        x = F.relu(self.bn31(x))
        x = self.conv32(x)
        x = F.relu(self.bn32(x))
        x = self.conv33(x)
        x = F.relu(self.bn33(x))
        x = self.conv34(x)
        x = F.relu(self.bn34(x))
        x = self.conv35(x)
        x = F.relu(self.bn35(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))

        x_tr = self.tr_conv1(x)
        x_tr = F.relu(self.tr_bn1(x_tr))
        x_tr = self.tr_conv2(x_tr)
        x_tr = F.relu(self.tr_bn2(x_tr))
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)
        x_tr = self.fct(x_tr.view(x_tr.shape[0],-1))

        x_rot = self.rot_conv1(x)
        x_rot = F.relu(self.rot_bn1(x_rot))
        x_rot = self.rot_conv2(x_rot)
        x_rot = F.relu(self.rot_bn2(x_rot))
        x_rot = self.rot_drop(x_rot)
        x_rot = self.rot_pool(x_rot)
        x_rot = self.fc6DR(x_rot.view(x_rot.shape[0],-1))
        R = rot.compute_rotation_matrix_from_ortho6d(x_rot)

        x_s = self.s_conv1(x)
        x_s = F.relu(self.s_bn1(x_s))
        x_s = self.s_conv2(x_s)
        x_s = F.relu(self.s_bn2(x_s))
        x_s = self.s_drop(x_s)
        x_s = self.s_pool(x_s)
        x_s = self.fcs(x_s.view(x_s.shape[0],-1))

        return R, x_tr, x_s

class regressor(nn.Module):
    
    def __init__(self, nPts = 8):
        super(regressor, self).__init__()

        self.poseRegressionLayer = poseRegressionNet(inplanes=nPts)


    def forward(self, image_rays1, image_rays2, cameraoffsets1, cameraoffsets2):

        # concatenate 3D-2D correspondeces
        x = torch.cat((image_rays1, image_rays2, cameraoffsets1, cameraoffsets2), dim = 2)

        # regression layers
        # stack of q t and scale
        R_regress, t_regress, s_regress = self.poseRegressionLayer(x)

        q_regress = rot.batch_rotm2quat(R_regress)

        return torch.cat([q_regress, t_regress, s_regress], dim=1), R_regress
