import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rt_warp(rotation, translation, invert=False, grid_size=64, epsilon=1e-6):

    rotation_matrix = compute_rotation_matrix(rotation)

    affine_matrix = torch.eye(4, device=rotation.device).repeat(rotation.shape[0], 1, 1)

    affine_matrix[:, :3, :3] = rotation_matrix

    affine_matrix[:, :3, 3] = translation

    affine_matrix[:, :3, :3] += torch.eye(3, device=rotation.device) * epsilon

    if invert:
        try:
            affine_matrix = torch.inverse(affine_matrix)
        except torch._C._LinAlgError:
            print("Singular matrix encountered. Skipping inversion.")
            return None

    grid = F.affine_grid(affine_matrix[:, :3], (rotation.shape[0], 1, grid_size, grid_size, grid_size), align_corners=False)
    grid = grid.permute(0, 4, 1, 2, 3)
    return grid

def compute_rotation_matrix(rotation):

    rotation_rad = rotation * (torch.pi / 180.0)  

    cos_alpha = torch.cos(rotation_rad[:, 0])
    sin_alpha = torch.sin(rotation_rad[:, 0])
    cos_beta = torch.cos(rotation_rad[:, 1])
    sin_beta = torch.sin(rotation_rad[:, 1])
    cos_gamma = torch.cos(rotation_rad[:, 2])
    sin_gamma = torch.sin(rotation_rad[:, 2])

    zero = torch.zeros_like(cos_alpha)
    one = torch.ones_like(cos_alpha)

    R_alpha = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_alpha, -sin_alpha], dim=1),
        torch.stack([zero, sin_alpha, cos_alpha], dim=1)
    ], dim=1)

    R_beta = torch.stack([
        torch.stack([cos_beta, zero, sin_beta], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_beta, zero, cos_beta], dim=1)
    ], dim=1)

    R_gamma = torch.stack([
        torch.stack([cos_gamma, -sin_gamma, zero], dim=1),
        torch.stack([sin_gamma, cos_gamma, zero], dim=1),
        torch.stack([zero, zero, one], dim=1)
    ], dim=1)

    rotation_matrix = torch.matmul(R_alpha, torch.matmul(R_beta, R_gamma))

    return rotation_matrix
