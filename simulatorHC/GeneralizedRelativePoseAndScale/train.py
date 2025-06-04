import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import simulatorHC.utils as utils
import simulatorHC.utils.rot as rot
import simulatorHC.GeneralizedRelativePoseAndScale.regressor as RegressorModule
from simulatorHC.utils.genData import Synthetic2D2DCorrNoncentral
from simulatorHC.GeneralizedRelativePoseAndScale import helper

import time
from itertools import chain
from pathlib import Path

import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set some parameters')

    parser.add_argument('--BatchSize', type=int, default=32, help='Number of Batch Size')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--nCameras', type=int, default=7, help='Number of cameras')
    parser.add_argument('--nPts', type=int, default=8, help='Number of correspondences')
    parser.add_argument('--TrainSize', type=int, default=64000, help='Number of training samples')
    parser.add_argument('--ValSize', type=int, default=32000, help='Number of validation samples')
    parser.add_argument('--noise_level', type=float, default=0.0, help='noise level in pixel')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--factor', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--patience', type=int, default=10, help='learning rate patience')
    parser.add_argument('--cooldown', type=int, default=30, help='learning rate cooldown')

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    BatchSize = args.BatchSize
    noise_level = args.noise_level
    nCameras = args.nCameras
    nPts = args.nPts

    t_start = time.time()

    Net = RegressorModule.regressor(nPts=nPts)

    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
    Net = Net.to(device)

    criterion = utils.criterion.Lossqts()

    optimizer = optim.Adam(chain(Net.parameters(), criterion.parameters()), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.factor, verbose=True, cooldown=args.cooldown)

    num_epochs = args.epoch

    dataset1 = Synthetic2D2DCorrNoncentral(num_samples=args.TrainSize, nPts=nPts, nCamera=nCameras, noise_std=noise_level)
    train_dataloader = DataLoader(dataset1, batch_size=BatchSize)
    dataset2 = Synthetic2D2DCorrNoncentral(num_samples=args.ValSize, nPts=nPts, nCamera=nCameras, noise_std=noise_level)
    val_dataloader = DataLoader(dataset2, batch_size=BatchSize)

    save_dir = Path('output/GeneralizedRelativePoseAndScale/weights')
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print(f"start training on {device}")
    for epoch in tqdm(range(num_epochs)):

        Net.train()  # Set the model to training mode
        total_train_loss = 0.0

        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            worldPts, imagePts_unit1, imagePts_unit2, R_gt, t_gt, depths, scale, CameraOffsets1, CameraOffsets2 = batch
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

            x_hat, R_hat = Net(imagePts_unit1,imagePts_unit2, CameraOffsets1, CameraOffsets2)
        
            q_hat, t_hat, s_hat = helper.getPosefromx(x_hat)

            loss = criterion(q_hat, q_gt, t_hat, t_gt, s_hat, scale)
            
            if torch.isnan(loss).any():
                print('got nan')

            loss.backward()
            
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataset1)

        Net.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:

                worldPts, imagePts_unit1, imagePts_unit2, R_gt, t_gt, depths, scale, CameraOffsets1, CameraOffsets2 = batch
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

                x_hat, R_hat = Net(imagePts_unit1,imagePts_unit2, CameraOffsets1, CameraOffsets2)
                
                q_hat, t_hat, s_hat = helper.getPosefromx(x_hat)
                
                loss = criterion(q_hat, q_gt, t_hat, t_gt, s_hat, scale)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataset2)
        
        scheduler.step(avg_val_loss) 

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            file_name = save_dir / f"regressor{nCameras}Cams{nPts}Pts.pth"
            torch.save(Net.state_dict(), file_name)
                
                                                                                                                                                                                                                                                                                                                
    t_end = time.time()
    print(f"total time:  {t_end-t_start} sec")
