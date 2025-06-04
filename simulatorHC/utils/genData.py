import torch
from torch.utils.data import Dataset

def generate_Rq():
    q = torch.rand(4)
    q = q/torch.norm(q)
    q1,q2,q3,q4 = q
    R_gt = torch.tensor([[q1*q1+q2*q2-q3*q3-q4*q4, 2*q2*q3-2*q1*q4, 2*q2*q4+2*q1*q3],
                [2*q2*q3+2*q1*q4, q1*q1-q2*q2+q3*q3-q4*q4, 2*q3*q4-2*q1*q2],
                [2*q2*q4-2*q1*q3, 2*q3*q4+2*q1*q2, q1*q1-q2*q2-q3*q3+q4*q4]])
    return R_gt, q

def generateR_torch_boundR(scale=0.5):
    # Generate random Euler angles
    yaw = (torch.rand(1) - 0.5) * 2 * torch.pi * scale
    pitch = (torch.rand(1) - 0.5) * 2 * torch.pi * scale
    roll = (torch.rand(1) - 0.5) * 2 * torch.pi * scale
    
    # Yaw rotation matrix (Z-axis)
    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Pitch rotation matrix (Y-axis)
    Ry = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])
    
    # Roll rotation matrix (X-axis)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])
    
    # Combine the rotations
    R = Rz @ Ry @ Rx  # Note: The order of multiplication matters
    return R


def add_noise_batch(noise_level, clean_points):
    
    # Preparing the normal_vector1 for batch operation
    normal_vector1 = torch.zeros_like(clean_points)
    
    # Conditions are now based on the maximum along the last dimension (axis=1)
    max_indices = torch.argmax(torch.abs(clean_points), dim=1)
    
    for i, max_index in enumerate(max_indices):
        if max_index == 0:  # Similar condition as the single point case, but for the whole batch
            normal_vector1[i, 1] = 1.0
            normal_vector1[i, 0] = -clean_points[i, 1] / clean_points[i, 0]
        elif max_index == 1:
            normal_vector1[i, 2] = 1.0
            normal_vector1[i, 1] = -clean_points[i, 2] / clean_points[i, 1]
        else:
            normal_vector1[i, 0] = 1.0
            normal_vector1[i, 2] = -clean_points[i, 0] / clean_points[i, 2]
    
    # Normalizing each vector in the batch
    normal_vector1 = normal_vector1 / torch.norm(normal_vector1, dim=1, keepdim=True)
    normal_vector2 = torch.cross(clean_points, normal_vector1, dim=1)
    
    # Generating noise for the entire batch
    noise_x = noise_level * (torch.rand(clean_points.shape[0], 1) - 0.5) * 2.0 / 1.4142
    noise_y = noise_level * (torch.rand(clean_points.shape[0], 1) - 0.5) * 2.0 / 1.4142
    
    # Adding noise to the clean points
    noisy_points = 800 * clean_points + noise_x * normal_vector1 + noise_y * normal_vector2
    noisy_points = noisy_points / torch.norm(noisy_points, dim=1, keepdim=True)
    
    return noisy_points


def generate_noncentral3D2D_data_torch_noisy(N, R, t, CameraOffsets, noise_std):
    # the output is transpose for the sake of batch

    # generate bearing vectors
    Pts3D = 2.0 * (torch.rand(3, N) - 0.5)
    directions = Pts3D / torch.norm(Pts3D, dim=0)
    worldPts = (8.0 - 4.0) * Pts3D + 4.0 * directions

    body_points = R @ worldPts + t.unsqueeze(1)

    body_points -= CameraOffsets

    imagePts_unit = body_points.T / torch.norm(body_points.T, dim=1, keepdim=True)

    if noise_std != 0.0:
        imagePts_unit = add_noise_batch(noise_std, imagePts_unit)
    return worldPts.T, imagePts_unit

class Synthetic3D2DCorrNoncentralPnP(Dataset):
    # dataset for Generalized Absolute Pose

    def __init__(self, num_samples, nPts, nCamera, noise_std=0.0):
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.nPts = nPts
        self.nCamera = nCamera

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        R, q = generate_Rq()
        t = torch.rand(3)
        
        CameraOffset_ = 2*(torch.rand(3, self.nCamera)-0.5)


        # preprocess the offset and camera rotation
        CameraOffsets = CameraOffset_.repeat(1, self.nPts//self.nCamera)
        nLeft = self.nPts-CameraOffsets.shape[1]
        CameraOffsets = torch.cat(
            [CameraOffsets, CameraOffset_[:, :nLeft]], dim=1)

        worldPts, imagePts_unit = generate_noncentral3D2D_data_torch_noisy(self.nPts, R, t,
                                                                       CameraOffsets, self.noise_std)

        return worldPts, imagePts_unit, R, q, t, CameraOffsets.T
    

def generate_noncentral2D2D_data_torch_noisy(N, R1, t1, CameraOffsets1, 
                                             R2, t2, CameraOffsets2, noise_std):
    # the output is transpose for the sake of batch

    # generate world points
    Pts3D = 2.0 * (torch.rand(3, N) - 0.5)
    directions = Pts3D / torch.norm(Pts3D, dim=0)
    worldPts = (8.0 - 4.0) * Pts3D + 4.0 * directions


    # view1
    body_points1 = R1 @ worldPts + t1.unsqueeze(1)
    body_points1 -= CameraOffsets1
    depths1 = torch.norm(body_points1.T, dim=1, keepdim=True)
    imagePts_unit1 = body_points1.T / depths1

    if noise_std != 0.0:
        imagePts_unit1 = add_noise_batch(noise_std, imagePts_unit1)


    # view2
    body_points2 = R2 @ worldPts + t2.unsqueeze(1)
    body_points2 -= CameraOffsets2
    depths2 = torch.norm(body_points2.T, dim=1, keepdim=True)
    imagePts_unit2 = body_points2.T / depths2

    if noise_std != 0.0:
        imagePts_unit2 = add_noise_batch(noise_std, imagePts_unit2)


    return worldPts.T, imagePts_unit1, imagePts_unit2, torch.cat([depths1, depths2]).squeeze(-1)


class Synthetic2D2DCorrNoncentral(Dataset):

    # dataset for Generalized Relative Pose and Scale

    def __init__(self, num_samples, nPts, nCamera, noise_std=0.0):
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.nPts = nPts
        self.nCamera = nCamera

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # view1
        R1 = generateR_torch_boundR(scale=0.5)
        t1 = (torch.rand(3)-0.5)*4

        CameraOffset_1 = torch.rand(3, self.nCamera)
        CameraOffsets1 = CameraOffset_1.repeat(1, self.nPts//self.nCamera)
        nLeft1 = self.nPts-CameraOffsets1.shape[1]
        CameraOffsets1 = torch.cat([CameraOffsets1, CameraOffset_1[:,:nLeft1]], dim=1)
        
        # view2
        # R2,q2 = generate_Rq()
        R2 = generateR_torch_boundR(scale=0.5)
        t2 = (torch.rand(3)-0.5)*4

        CameraOffset_2 = torch.rand(3, self.nCamera)
        CameraOffsets2 = CameraOffset_2.repeat(1, self.nPts//self.nCamera)
        nLeft2 = self.nPts-CameraOffsets2.shape[1]
        CameraOffsets2 = torch.cat([CameraOffsets2, CameraOffset_2[:,:nLeft2]], dim=1)

        scale = torch.rand(1)*4.9+0.1

        worldPts, imagePts_unit1, imagePts_unit2, depths = generate_noncentral2D2D_data_torch_noisy(self.nPts, 
                                        R1, t1, CameraOffsets1, 
                                        R2, t2, scale[0] * CameraOffsets2, self.noise_std)
        
        R = R1 @ R2.T
        t = - R @ t2 + t1
 
        return worldPts, imagePts_unit1, imagePts_unit2, R, t, depths, scale, CameraOffsets1.T, CameraOffsets2.T
