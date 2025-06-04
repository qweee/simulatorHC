
import time
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

import simulatorHC.utils.rot as rot
import simulatorHC.utils.metric as metric
import simulatorHC.GeneralizedRelativePoseAndScale.regressor as RegressorModule
from simulatorHC.utils.genData import Synthetic2D2DCorrNoncentral
import simulatorHC.GeneralizedRelativePoseAndScale.helper as helper
import simulatorHC.GeneralizedRelativePoseAndScale.HC_GRPS as HC_GRPS


def HCrefine(Net, imagePts_unit1, imagePts_unit2, CameraOffsets1, CameraOffsets2, 
             R_gt, q_gt, t_gt, scale, nPts, nCameras, noise_level):

    BatchSize = R_gt.shape[0]

    t1 = time.time()
    with torch.no_grad():
        x_hat, R_regress = Net(imagePts_unit1[:,:nPts,:], imagePts_unit2[:,:nPts,:],
                                CameraOffsets1[:,:nPts,:], CameraOffsets2[:,:nPts,:])

    q_regress, t_regress, s_regress = helper.getPosefromx(x_hat)

    fs_hat, _, __ = helper.getReprojectedf_6D_batch(R_regress, t_regress, s_regress, imagePts_unit1, imagePts_unit2, CameraOffsets1, CameraOffsets2)

    x_initial = torch.cat([q_regress, t_regress, s_regress.unsqueeze(-1)], dim=1)
    x_hc = torch.zeros_like(x_initial)

    for b in range(x_initial.shape[0]):
        x_hc_, hc_time = HC_GRPS.HomotopyContinuation_GRPS(x_initial[b].unsqueeze(-1), imagePts_unit1[b].T, imagePts_unit2[b].T, 
                                CameraOffsets1[b].T, CameraOffsets2[b].T, fs_hat[b].T, 0.05, 10, nPts)
        x_hc[b] = x_hc_.float().squeeze(-1)
    R_hc, q_hc, t_hc, s_hc = helper.getPosefromx2(x_hc)

    t2 = time.time()
    avg_time = ((t2-t1)/BatchSize)*1e3

    q_sign = torch.sign(q_hc.view(-1,1,4) @ q_gt.unsqueeze(-1)).squeeze(-1)
    q_errors = torch.norm(q_sign * q_hc-q_gt, dim=1)
    R_errors = metric.rotm_error_batch(R_gt, R_hc)
    t_errors = torch.norm(t_hc-t_gt, dim=1)/torch.norm(t_gt, dim=1)
    s_errors = torch.abs(s_hc - scale)/torch.abs(s_hc)

    # save results
    results_dict = {
        'q_errors': q_errors.view(-1).cpu().detach().numpy(),
        'R_errors': R_errors.view(-1).cpu().detach().numpy(),
        't_errors': t_errors.view(-1).cpu().detach().numpy(),
        's_errors': s_errors.view(-1).cpu().detach().numpy(),
    }

    np.save('output/GeneralizedRelativePoseAndScale/results'+str(nCameras)+'Cams'+str(nPts)+'Pts'+'Noise_'+str(noise_level)+'.npy', results_dict)

    R_success_rate = (R_errors <= 1).sum()/BatchSize*100
    t_success_rate = (t_errors <= 0.05).sum()/BatchSize*100
    s_success_rate = (s_errors <= 0.05).sum()/BatchSize*100

    t_ang_errors, t_s_errors = metric.getterrors(t_hc, t_gt)

    t_ang_success_rate = (t_ang_errors <= 2).sum()/BatchSize*100
    t_s_success_rate = (t_s_errors <= 0.05).sum()/BatchSize*100

    print("--------------------------------")
    print(f"{nCameras} Cameras, {nPts} Points")
    print("evaluating on GPU with 1000 trials")
    print(f'Rotation success rate: {R_success_rate:.2f} %')
    print(f'Translation success rate: {t_success_rate:.2f} %')
    print(f'Scale success rate: {s_success_rate:.2f} %')
    print(f'Translation angular error success rate: {t_ang_success_rate:.2f} %')
    print(f'Translation scale error success rate: {t_s_success_rate:.2f} %')
    print(f'Average time: {avg_time:.2f} msec')
    print("--------------------------------")

    return 0

def main(nCameras, nPts, noise_level=0.0):

    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')

    nTrials = 1000

    dataset = Synthetic2D2DCorrNoncentral(num_samples=nTrials, nPts=nPts, \
                                                nCamera=nCameras, noise_std=noise_level)
    dataloader = DataLoader(dataset, batch_size=nTrials)
    dataiter = iter(dataloader)
    batch = next(dataiter)

    worldPts, imagePts_unit1, imagePts_unit2, R_gt, t_gt, depths, scale, \
        CameraOffsets1, CameraOffsets2 = batch
    worldPts = worldPts.to(device)
    imagePts_unit1 = imagePts_unit1.to(device)
    imagePts_unit2 = imagePts_unit2.to(device)
    R_gt = R_gt.to(device)
    q_gt = rot.batch_rotm2quat(R_gt).to(device)
    t_gt = t_gt.to(device)
    CameraOffsets1 = CameraOffsets1.to(device)
    CameraOffsets2 = CameraOffsets2.to(device)
    scale = scale.view(-1).to(device)
    depths = depths.to(device)

    Net = RegressorModule.regressor(nPts)
    model = Net.to(device)

    state_dict = torch.load('output/GeneralizedRelativePoseAndScale/weights/regressor'+str(nCameras)+'Cams'+str(nPts)+'Pts.pth')
    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode

    HCrefine(Net, imagePts_unit1, imagePts_unit2, CameraOffsets1, CameraOffsets2, 
             R_gt, q_gt, t_gt, scale, nPts, nCameras, noise_level)
        
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set some parameters')

    parser.add_argument('--nCameras', type=int, default=7, help='Number of cameras')
    parser.add_argument('--nPts', type=int, default=8, help='Number of correspondences')
    parser.add_argument('--noise_level', type=float, default=0.0, help='noise level in pixel')

    # Parse arguments
    args = parser.parse_args()

    main(args.nCameras, args.nPts, args.noise_level)


    


