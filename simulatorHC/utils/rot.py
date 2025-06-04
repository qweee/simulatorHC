import numpy as np
import torch

def quat2rotm(q):
    q1, q2, q3, q4 = q
    R = np.array([[q1*q1+q2*q2-q3*q3-q4*q4, 2*q2*q3-2*q1*q4, 2*q2*q4+2*q1*q3],
                [2*q2*q3+2*q1*q4, q1*q1-q2*q2+q3*q3-q4*q4, 2*q3*q4-2*q1*q2],
                [2*q2*q4-2*q1*q3, 2*q3*q4+2*q1*q2, q1*q1-q2*q2-q3*q3+q4*q4]])
    return R


def rotation_matrix_to_angle_axis(R):
    # Ensure R is a proper rotation matrix here

    theta = torch.acos((R.trace() - 1) / 2)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    sin_theta = torch.sin(theta) + epsilon

    v_x = (R[2, 1] - R[1, 2]) / (2 * sin_theta)
    v_y = (R[0, 2] - R[2, 0]) / (2 * sin_theta)
    v_z = (R[1, 0] - R[0, 1]) / (2 * sin_theta)

    v = torch.tensor([v_x, v_y, v_z])
    v = v / v.norm()  # Normalize to make it a unit vector

    return v, theta

def quat2rotm_tensor(q):
    # Ensure q requires_grad
    # q.requires_grad_(True)
    
    q1, q2, q3, q4 = q.unbind()  # Unbind q for readability

    # Use operations that keep the gradient tracking
    R = torch.stack([
        torch.stack([q1*q1 + q2*q2 - q3*q3 - q4*q4, 2*q2*q3 - 2*q1*q4, 2*q2*q4 + 2*q1*q3]),
        torch.stack([2*q2*q3 + 2*q1*q4, q1*q1 - q2*q2 + q3*q3 - q4*q4, 2*q3*q4 - 2*q1*q2]),
        torch.stack([2*q2*q4 - 2*q1*q3, 2*q3*q4 + 2*q1*q2, q1*q1 - q2*q2 - q3*q3 + q4*q4])
    ], dim=0)

    R = R.to(q.device)  # Ensure R is on the same device as q
    return R.squeeze(-1)


def rotm2quat(R):
    # Ensure the matrix is a numpy array
    # R = np.asarray(R)

    # Allocate space for the quaternion
    q = np.empty(4)

    # Compute the trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s

    # Normalize the quaternion
    q /= np.linalg.norm(q)

    return q


def getR_batch(q):

    # normalize quaternion (should have been normalized)
    q_normalized = q / torch.norm(q, dim=1, keepdim=True)

    # get Rotation from quaternion
    R = torch.zeros(*q_normalized.shape[:-1], 3, 3, device=q.device)
    R[:, 0, 0] = q_normalized[:, 0]**2 + q_normalized[:, 1]**2 - q_normalized[:, 2]**2 - q_normalized[:, 3]**2
    R[:, 0, 1] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 2] - q_normalized[:, 0] * q_normalized[:, 3])
    R[:, 0, 2] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 3] + q_normalized[:, 0] * q_normalized[:, 2])
    R[:, 1, 0] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 2] + q_normalized[:, 0] * q_normalized[:, 3])
    R[:, 1, 1] = q_normalized[:, 0]**2 - q_normalized[:, 1]**2 + q_normalized[:, 2]**2 - q_normalized[:, 3]**2
    R[:, 1, 2] = 2.0 * (q_normalized[:, 2] * q_normalized[:, 3] - q_normalized[:, 0] * q_normalized[:, 1])
    R[:, 2, 0] = 2.0 * (q_normalized[:, 1] * q_normalized[:, 3] - q_normalized[:, 0] * q_normalized[:, 2])
    R[:, 2, 1] = 2.0 * (q_normalized[:, 2] * q_normalized[:, 3] + q_normalized[:, 0] * q_normalized[:, 1])
    R[:, 2, 2] = q_normalized[:, 0]**2 - q_normalized[:, 1]**2 - q_normalized[:, 2]**2 + q_normalized[:, 3]**2

    return R

def batch_rotm2quat(R):
    batch_size = R.size(0)
    q = torch.empty(batch_size, 4, dtype=R.dtype, device=R.device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    s = trace.clone()

    # Case when trace of matrix is greater than zero
    idx1 = trace > 0
    s[idx1] = torch.sqrt(trace[idx1] + 1.0) * 2
    q[idx1, 0] = 0.25 * s[idx1]
    q[idx1, 1] = (R[idx1, 2, 1] - R[idx1, 1, 2]) / s[idx1]
    q[idx1, 2] = (R[idx1, 0, 2] - R[idx1, 2, 0]) / s[idx1]
    q[idx1, 3] = (R[idx1, 1, 0] - R[idx1, 0, 1]) / s[idx1]

    # Case when R[0, 0] is the largest diagonal element
    idx2 = ~idx1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s[idx2] = torch.sqrt(1.0 + R[idx2, 0, 0] - R[idx2, 1, 1] - R[idx2, 2, 2]) * 2
    q[idx2, 0] = (R[idx2, 2, 1] - R[idx2, 1, 2]) / s[idx2]
    q[idx2, 1] = 0.25 * s[idx2]
    q[idx2, 2] = (R[idx2, 0, 1] + R[idx2, 1, 0]) / s[idx2]
    q[idx2, 3] = (R[idx2, 0, 2] + R[idx2, 2, 0]) / s[idx2]

    # Case when R[1, 1] is the largest diagonal element
    idx3 = ~idx1 & (R[:, 1, 1] > R[:, 2, 2])
    s[idx3] = torch.sqrt(1.0 + R[idx3, 1, 1] - R[idx3, 0, 0] - R[idx3, 2, 2]) * 2
    q[idx3, 0] = (R[idx3, 0, 2] - R[idx3, 2, 0]) / s[idx3]
    q[idx3, 1] = (R[idx3, 0, 1] + R[idx3, 1, 0]) / s[idx3]
    q[idx3, 2] = 0.25 * s[idx3]
    q[idx3, 3] = (R[idx3, 1, 2] + R[idx3, 2, 1]) / s[idx3]

    # Remaining case
    idx4 = ~idx1 & ~idx2 & ~idx3
    s[idx4] = torch.sqrt(1.0 + R[idx4, 2, 2] - R[idx4, 0, 0] - R[idx4, 1, 1]) * 2
    q[idx4, 0] = (R[idx4, 1, 0] - R[idx4, 0, 1]) / s[idx4]
    q[idx4, 1] = (R[idx4, 0, 2] + R[idx4, 2, 0]) / s[idx4]
    q[idx4, 2] = (R[idx4, 1, 2] + R[idx4, 2, 1]) / s[idx4]
    q[idx4, 3] = 0.25 * s[idx4]

    # Normalize the quaternions
    q = q / torch.norm(q, dim=1, keepdim=True)

    return q

# for 6D rotation representation
# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    # v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]).to(v.device))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix