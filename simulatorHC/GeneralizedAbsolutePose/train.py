import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from itertools import chain
import time
from pathlib import Path

import simulatorHC.GeneralizedAbsolutePose.regressor as RegressorModule
import simulatorHC.utils as utils
from simulatorHC.utils.genData import Synthetic3D2DCorrNoncentralPnP

import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for training')

    parser.add_argument('--BatchSize', type=int, default=32, help='Number of Batch Size')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--noise_level', type=float, default=2.0, help='noise level in pixel')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--factor', type=float, default=0.5, help='lr decay factor')
    parser.add_argument('--patience', type=int, default=10, help='lr patience')
    parser.add_argument('--nPts', type=int, default=16, help='number of correspondences')
    parser.add_argument('--nCams', type=int, default=4, help='number of cameras')

    # Parse arguments
    args = parser.parse_args()
    noise_level = args.noise_level

    t_start = time.time()

    # Define the model
    Net = RegressorModule.regressor(nPts=16) # nPts is the number of correspondences

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Net = Net.to(device)

    # Define the loss function
    criterion = utils.criterion.Lossqt(initial_weight_value=0.5)

    # Define the optimizer
    optimizer = optim.Adam(chain(Net.parameters(), criterion.parameters()), lr=args.lr)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.factor, verbose=True)

    # Define the number of epochs
    num_epochs = args.epoch

    # Define the training and validation datasets
    dataset1 = Synthetic3D2DCorrNoncentralPnP(num_samples=2400, nPts=args.nPts, nCamera=args.nCams, noise_std=noise_level)
    train_dataloader = DataLoader(dataset1, batch_size=args.BatchSize)
    dataset2 = Synthetic3D2DCorrNoncentralPnP(num_samples=1200, nPts=args.nPts, nCamera=args.nCams, noise_std=noise_level)
    val_dataloader = DataLoader(dataset2, batch_size=args.BatchSize)

    # start training
    save_dir = Path('output/GeneralizedAbsolutePose/weights')
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    print(f"start training on {device}")
    for epoch in tqdm(range(num_epochs)):
        Net.train()  # Set the model to training mode
        total_train_loss = 0.0

        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            worldPts, imagePts_unit, R_gt, q_gt, t_gt, CameraOffsets = batch
            worldPts = worldPts.to(device)
            imagePts_unit = imagePts_unit.to(device)
            
            q_gt = q_gt.to(device)
            t_gt = t_gt.to(device)
            CameraOffsets = CameraOffsets.to(device)
            x_regress = Net(worldPts, imagePts_unit, CameraOffsets)
            
            q_pred = x_regress[:,:4]

            loss = criterion(q_pred, q_gt)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataset1)

        Net.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:

                worldPts, imagePts_unit, R_gt, q_gt, t_gt, CameraOffsets = batch
                worldPts = worldPts.to(device)
                imagePts_unit = imagePts_unit.to(device)
                R_gt = R_gt.to(device)
                q_gt = q_gt.to(device)
                t_gt = t_gt.to(device)
                CameraOffsets = CameraOffsets.to(device)
                x_regress = Net(worldPts, imagePts_unit, CameraOffsets)

                q_pred = x_regress[:,:4]

                loss = criterion(q_pred, q_gt)
                
                if torch.isnan(loss).any():
                    print('got nan')

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataset2)

        # Update learning rate if necessary
        scheduler.step(avg_val_loss) 

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(Net.state_dict(), save_dir/'regressor.pth')

                                                                                                                                                                                                                                                                        
    t_end = time.time()
    print(f"training time:  {t_end-t_start} sec")
