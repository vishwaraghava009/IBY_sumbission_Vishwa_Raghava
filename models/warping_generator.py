import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from models.flowfield import FlowField
from models.rot_mat import compute_rt_warp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPRESS_DIM = 512


class WarpGeneratorS2C(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorS2C, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rs, ts, zs, es):
        assert Rs.shape == (zs.shape[0], 3), f"Expected Rs shape (batch_size, 3), got {Rs.shape}"
        assert ts.shape == (zs.shape[0], 3), f"Expected ts shape (batch_size, 3), got {ts.shape}"
        assert zs.shape == es.shape, f"Expected zs and es to have the same shape, got {zs.shape} and {es.shape}"

        zs_sum = zs + es

        # adaptive_gamma = torch.matmul(zs_sum, self.adaptive_matrix_gamma.T)
        # adaptive_beta = torch.matmul(zs_sum, self.adaptive_matrix_beta.T)
     
        zs_sum = torch.matmul(zs_sum, self.adaptive_matrix_gamma) 
        zs_sum = zs_sum.unsqueeze(-1).unsqueeze(-1) 

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_s2c = self.flowfield(zs_sum,adaptive_gamma,adaptive_beta) 
        logging.debug(f"w_em_s2c:  :{w_em_s2c.shape}") 
        w_rt_s2c = compute_rt_warp(Rs, ts, invert=True, grid_size=64)
        logging.debug(f"w_rt_s2c: :{w_rt_s2c.shape}") 
        
        w_em_s2c_resized = F.interpolate(w_em_s2c, size=w_rt_s2c.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_s2c_resized: {w_em_s2c_resized.shape}")
        w_s2c = w_rt_s2c + w_em_s2c_resized

        return w_s2c


class WarpGeneratorC2D(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorC2D, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rd, td, zd, es):
        assert Rd.shape == (zd.shape[0], 3), f"Expected Rd shape (batch_size, 3), got {Rd.shape}"
        assert td.shape == (zd.shape[0], 3), f"Expected td shape (batch_size, 3), got {td.shape}"
        assert zd.shape == es.shape, f"Expected zd and es to have the same shape, got {zd.shape} and {es.shape}"

        zd_sum = zd + es
        
        # adaptive_gamma = torch.matmul(zd_sum, self.adaptive_matrix_gamma)
        # adaptive_beta = torch.matmul(zd_sum, self.adaptive_matrix_beta)
       
        zd_sum = torch.matmul(zd_sum, self.adaptive_matrix_gamma) 
        zd_sum = zd_sum.unsqueeze(-1).unsqueeze(-1) 

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_c2d = self.flowfield(zd_sum,adaptive_gamma,adaptive_beta)

        w_rt_c2d = compute_rt_warp(Rd, td, invert=False, grid_size=64)

        w_em_c2d_resized = F.interpolate(w_em_c2d, size=w_rt_c2d.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_c2d_resized:{w_em_c2d_resized.shape}" )

        w_c2d = w_rt_c2d + w_em_c2d_resized

        return w_c2d