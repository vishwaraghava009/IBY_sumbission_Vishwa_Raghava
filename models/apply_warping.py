import torch
import torch.nn as nn
import torch.nn.functional as F
import logging



def apply_warping_field(v, warp_field):
    B, C, D, H, W = v.size()
    logging.debug(f"apply_warping_field v:{v.shape}", )
    logging.debug(f"warp_field:{warp_field.shape}" )

    device = v.device


    warp_field = F.interpolate(warp_field, size=(D, H, W), mode='trilinear', align_corners=True)
    logging.debug(f"Resized warp_field:{warp_field.shape}" )

    
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # Shape: [D, H, W, 3]
    logging.debug(f"Canonical grid:{grid.shape}" )

    
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Batch grid:{grid.shape}" )

    warped_grid = grid + warp_field.permute(0, 2, 3, 4, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Warped grid:{warped_grid.shape}" )

    normalization_factors = torch.tensor([W-1, H-1, D-1], device=device)
    logging.debug(f"Normalization factors:{normalization_factors}" )
    warped_grid = 2.0 * warped_grid / normalization_factors - 1.0
    logging.debug(f"Normalized warped grid:{warped_grid.shape}" )

    v_canonical = F.grid_sample(v, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
    logging.debug(f"v_canonical:{v_canonical.shape}" )

    return v_canonical



class AntiAliasInterpolation2d(nn.Module):
    
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka


        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
       
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out

class ImagePyramide(torch.nn.Module):
   
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict
