import torch
import torch.nn.functional as F

def getterrors(t_pred, t_gt):
    angle_errors = 2 * torch.asin(((F.normalize(t_gt, p=2, dim=1)-F.normalize(t_pred, p=2, dim=1)).norm(dim=1))/2)
    angle_errors = angle_errors.rad2deg()

    scale_errors = ((torch.norm(t_pred, p=2, dim=1)/torch.norm(t_gt, p=2, dim=1)) - 1).abs()

    return angle_errors, scale_errors


def rotm_error_batch(R1, R2):

    sintheta = torch.linalg.norm(R1-R2, ord='fro', dim=(1, 2))/(2*torch.sqrt(torch.tensor(2)))
    
    # Clamping for numerical stability
    sintheta = torch.clamp(sintheta, -1, 1)
    
    angle_rad = 2 * torch.asin(sintheta)
    # print('angle_rad:', angle_rad)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg