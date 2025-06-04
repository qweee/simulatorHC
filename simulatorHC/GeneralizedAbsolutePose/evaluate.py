import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

import simulatorHC.utils.metric as metric
import simulatorHC.utils.rot as rot
from simulatorHC.utils.genData import Synthetic3D2DCorrNoncentralPnP
import simulatorHC.GeneralizedAbsolutePose.regressor as model
import simulatorHC.GeneralizedAbsolutePose.helper as helper
import simulatorHC.GeneralizedAbsolutePose.HC_UPnP as HC_UPnP

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def reproj(worldPts, imagePts_unit, CameraOffsets, R, t):

    body_pts = torch.bmm(worldPts, R.transpose(1,2)) + t.view(-1,1,3) - CameraOffsets
    errors = (body_pts/body_pts.norm(dim=2, keepdim=True)-imagePts_unit).norm(dim=2).sum(dim=1)

    return errors

def main():

    nTrials = 1000

    # Define the model
    Net = model.regressor(nPts=16) # nPts is the number of correspondences
 
    device = 'cpu'
    Net = Net.to(device)

    # Load the saved state dict into the model
    Net.load_state_dict(torch.load('output/GeneralizedAbsolutePose/weights/regressor.pth'))
    Net.eval()  # Set the model to evaluation mode

    dataset = Synthetic3D2DCorrNoncentralPnP(num_samples=nTrials, nPts=16, nCamera=4, noise_std=2)
    dataloader = DataLoader(dataset, batch_size=nTrials)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    worldPts, imagePts_unit, R_gt, q_gt, t_gt, cameraoffsets = batch

    worldPts = worldPts.to(device)
    imagePts_unit = imagePts_unit.to(device)
    R_gt = R_gt.to(device)
    q_gt = q_gt.to(device)
    t_gt = t_gt.to(device)
    cameraoffsets = cameraoffsets.to(device)

    t1 = time.time()
    with torch.no_grad():
        x_regress = Net(worldPts, imagePts_unit, cameraoffsets)

    # get start system by simulator
    Pts2D_reproj = helper.reproj2D_batched_noncentral(worldPts, x_regress, cameraoffsets)

    coeffsG = helper.UnifiedPnPCoeff(Pts2D_reproj, worldPts, cameraoffsets)
    coeffsF, Is, Js = helper.UnifiedPnPCoeffall_IJ(imagePts_unit, worldPts, cameraoffsets)

    # solved by HC cpp implementation
    q_hc = torch.zeros(nTrials, 4, device=device)

    for i in range(nTrials):
        q_hc_, hc_time = HC_UPnP.HomotopyContinuation(
            np.array(x_regress[i][:4].cpu().data, dtype=np.float64),
            np.array(coeffsF[i].cpu().data, dtype=np.float64),
            np.array(coeffsG[i].cpu().data, dtype=np.float64),
        )
        q_hc.data[i] = torch.from_numpy(q_hc_).data

    t2 = time.time()
    avg_time = ((t2-t1)/nTrials)*1e3

    # get R,t
    q_hc = q_hc / torch.norm(q_hc, dim=1, keepdim=True)
    R_hc = rot.getR_batch(q_hc)

    s_batch = helper.upnp_fill_s_batched(q_hc)
    t_hc = torch.bmm(Is.cpu(), s_batch.cpu().unsqueeze(-1)).squeeze(-1) - Js.cpu()

    q_sign = torch.sign(q_hc.view(-1,1,4) @ q_gt.unsqueeze(-1)).squeeze(-1)
    q_errors = torch.norm(q_sign * q_hc-q_gt, dim=1)
    R_errors = metric.rotm_error_batch(R_gt, R_hc)
    t_errors = torch.norm(t_hc.to(device)-t_gt, dim=1)/torch.norm(t_gt, dim=1)

    reproj_gt = reproj(worldPts, imagePts_unit, cameraoffsets, R_gt, t_gt)
    reproj_errors = reproj(worldPts, imagePts_unit, cameraoffsets, R_hc, t_hc)

    # save results 
    results_dict = {
        'q_errors': q_errors.view(-1).cpu().detach().numpy(),
        'R_errors': R_errors.view(-1).cpu().detach().numpy(), 
        't_errors': t_errors.view(-1).cpu().detach().numpy(),
        'reproj_gt': reproj_gt.view(-1).cpu().detach().numpy(),
        'reproj_errors': reproj_errors.view(-1).cpu().detach().numpy()
    }
    np.save('output/GeneralizedAbsolutePose/results.npy', results_dict)

    print("--------------------------------")
    print("evaluating on CPU with 1000 trials")
    print(f'Mean rotation error: {R_errors.mean().item():.4f} deg')
    print(f'Mean translation error: {(t_errors.mean().item()*100):.4f} %')
    print(f'Mean reprojection error: {reproj_errors.mean().item():.4f}')
    print(f'average time: {avg_time :2f} msec')
    print("--------------------------------")

    return 0
    
    
if __name__ == "__main__":

    main()
