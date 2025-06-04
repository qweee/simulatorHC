import torch
import simulatorHC.utils.rot as rot
import simulatorHC.utils.metric as metric
import simulatorHC.GeneralizedRelativePoseAndScale.HC_GRPS as HC_GRPS
import time


def getPosefromx(x):
    q = x[:, :4]
    t = x[:, 4:7]
    scale = x[:, 7]
    return q, t, scale

def getPosefromx2(x):
    q = x[:, :4]
    R = rot.getR_batch(q)
    t = x[:, 4:7]
    scale = x[:, 7]
    return R, q, t, scale

def getdepths(R, t, s, fs, fs_prime, vs, vs_prime):

    # Compute depths
    B = torch.zeros(fs.shape[0], fs.shape[1], 2, 3, device=R.device)
    B[:, :, 0, :] = fs
    B[:, :, 1, :] = -torch.einsum('bij,bkj->bki', R, fs_prime)
    B = B.transpose(-2,-1)
    c = -vs + s.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bij,bkj->bki', R, vs_prime) + t.unsqueeze(1)
    depths = torch.linalg.pinv(B) @ c.unsqueeze(-1)

    return depths

def getReprojectedf_6D_batch(R, t, s, fs, fs_prime, vs, vs_prime):

    batch_size, nPts, _ = fs.shape
    q = rot.batch_rotm2quat(R)
    s = s.view(-1)

    # Compute depths
    B = torch.zeros(batch_size, nPts, 2, 3, device=q.device)
    B[:, :, 0, :] = fs
    B[:, :, 1, :] = -torch.einsum('bij,bkj->bki', R, fs_prime)
    B = B.transpose(-2,-1)
    c = -vs + s.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bij,bkj->bki', R, vs_prime) + t.unsqueeze(1)
    depths = torch.linalg.pinv(B) @ c.unsqueeze(-1)
    depths2 = depths[:,:,1,0]
    
    # Compute reprojected fs
    fs_hat = torch.einsum('bij,bkj->bki', R, depths2.unsqueeze(-1) * fs_prime + s.unsqueeze(-1).unsqueeze(-1) * vs_prime) + t.unsqueeze(1) - vs
    fs_hat = fs_hat/depths[:,:,0,:]

    return fs_hat, q, depths2

def HCrefine(Net,imagePts_unit1, imagePts_unit2, CameraOffsets1, CameraOffsets2, R_gt, q_gt, t_gt, scale, depths, model_params):

    BatchSize = R_gt.shape[0]

    t1 = time.time()
    with torch.no_grad():
        x_hat, R_regress = Net(imagePts_unit1[:,:model_params["nPts_train"],:], imagePts_unit2[:,:model_params["nPts_train"],:],
                                CameraOffsets1[:,:model_params["nPts_train"],:], CameraOffsets2[:,:model_params["nPts_train"],:])

    q_regress, t_regress, s_regress = getPosefromx(x_hat)
    fs_hat, _, __ = getReprojectedf_6D_batch(R_regress, t_regress, s_regress, imagePts_unit1, imagePts_unit2, CameraOffsets1, CameraOffsets2)

    x_initial = torch.cat([q_regress, t_regress, s_regress.unsqueeze(-1)], dim=1)
    x_hc = torch.zeros_like(x_initial)

    for b in range(x_initial.shape[0]):
        x_hc_ = HC_GRPS.HomotopyContinuation_GRPS(x_initial[b].unsqueeze(-1), imagePts_unit1[b].T, imagePts_unit2[b].T, 
                                    CameraOffsets1[b].T, CameraOffsets2[b].T, fs_hat[b].T)
        x_hc[b] = x_hc_.float().squeeze(-1)
    R_hc, q_hc, t_hc, s_hc = getPosefromx2(x_hc)

    t2 = time.time()
    avg_time = ((t2-t1)/BatchSize)*1e3
    print(f'average time: {((t2-t1)/BatchSize)*1e3 :2f} msec')

    # Compute depths
    B = torch.zeros(BatchSize, model_params["nPts"], 2, 3, device=R_hc.device)
    B[:, :, 0, :] = imagePts_unit1
    B[:, :, 1, :] = -torch.einsum('bij,bkj->bki', R_hc, imagePts_unit2)
    B = B.transpose(-2,-1)
    c = -CameraOffsets1 + s_hc.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bij,bkj->bki', R_hc, CameraOffsets2) + t_hc.unsqueeze(1)
    depths_hc = torch.linalg.pinv(B) @ c.unsqueeze(-1)
    reproj_hc = torch.einsum('bij,bkj->bki', R_hc, depths_hc[:,:,1,:] * imagePts_unit2 + \
                    s_hc.unsqueeze(-1).unsqueeze(-1) * CameraOffsets2) + t_hc.unsqueeze(1) - \
                    (depths_hc[:,:,0,:] * imagePts_unit1 + CameraOffsets1)
    reproj_errors_hc = reproj_hc.norm(dim=2).sum(dim=1)


    depths_hc = depths_hc.to(depths.device)
    depth_errors = (depths_hc[:,:,0].squeeze(-1) - depths[:,:model_params["nPts"]]).abs()/depths[:,:model_params["nPts"]] + (depths_hc[:,:,1].squeeze(-1) - depths[:,model_params["nPts"]:]).abs()/depths[:,model_params["nPts"]:]

    q_sign = torch.sign(q_hc.view(-1,1,4) @ q_gt.unsqueeze(-1)).squeeze(-1)
    q_errors = torch.norm(q_sign * q_hc-q_gt, dim=1)
    R_errors = metric.rotm_error_batch(R_gt, R_hc)
    t_errors = torch.norm(t_hc-t_gt, dim=1)/torch.norm(t_gt, dim=1)
    s_errors = torch.abs(s_hc - scale)/torch.abs(s_hc)
    d_errors = depth_errors.sum(dim=1)/model_params["nPts"]

    R_success_rate = (R_errors <= 1).sum()/BatchSize*100
    q_success_rate = (q_errors <= 0.01).sum()/BatchSize*100
    t_success_rate = (t_errors <= 0.05).sum()/BatchSize*100
    s_success_rate = (s_errors <= 0.05).sum()/BatchSize*100
    d_success_rate = (d_errors <= 0.05).sum()/BatchSize*100

    return q_success_rate.cpu().detach().numpy(), R_success_rate.cpu().detach().numpy(), \
        t_success_rate.cpu().detach().numpy(), s_success_rate.cpu().detach().numpy(), d_success_rate.cpu().detach().numpy(), \
        avg_time