import torch
import torch.nn as nn
import torch.nn.functional as F


class poseRegressionNet(nn.Module):
    # input: 3D-2D correspondences
    # output: quaternion and translation
    def __init__(self,inplanes=768,planes=96,final_feat=5):
        super(poseRegressionNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm1d(planes*4)
        self.conv2 = nn.Conv1d(in_channels=planes*4,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm1d(planes*4)
        self.conv3 = nn.Conv1d(in_channels=planes*4,out_channels=planes*2,kernel_size=2,stride=2)
        self.bn3 = nn.BatchNorm1d(planes*2)
        self.tr_conv = nn.Conv1d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm1d(planes)
        self.rot_conv = nn.Conv1d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm1d(planes)
        self.tr_drop = nn.Dropout1d(p=0.2)
        self.rot_drop = nn.Dropout1d(p=0.2)
        self.tr_pool = nn.AdaptiveAvgPool1d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool1d(output_size=final_feat)
        self.fc1 = nn.Linear(planes*final_feat,3)  # 96*10
        self.fc2 = nn.Linear(planes*final_feat,4)  # 96*10
        nn.init.xavier_normal_(self.fc1.weight,0.1)
        nn.init.xavier_normal_(self.fc2.weight,0.1)

    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x_tr = self.tr_conv(x)
        x_tr = F.relu(self.tr_bn(x_tr))
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))

        x_rot = self.rot_conv(x)
        x_rot = F.relu(self.rot_bn(x_rot))
        x_rot = self.rot_drop(x_rot)  
        x_rot = self.rot_pool(x_rot)  # (19.6)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        x_rot = F.normalize(x_rot, p=2, dim=1)

        result = torch.cat((x_rot, x_tr), dim=1)
        
        return result

class regressor(nn.Module):
    
    def __init__(self, nPts = 16):
        super(regressor, self).__init__()
        self.poseRegressionLayer = poseRegressionNet(nPts)

    def forward(self, Pts3D, Pts2D, cameraoffsets):

        # concatenate 3D-2D correspondeces
        x = torch.cat((Pts3D, Pts2D, cameraoffsets), axis = 2)

        # regression layers
        # stack of angles of R and t
        x = self.poseRegressionLayer(x) 

        return x
