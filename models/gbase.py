import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from torch.utils.checkpoint import checkpoint

from models.apply_warping import apply_warping_field, ImagePyramide
from models.app_encoder import Eapp
from models.motion_encoder import Emtn
from models.warping_generator import WarpGeneratorS2C, WarpGeneratorC2D
from models.g3d import G3d
from models.g2d import G2d


class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.appearanceEncoder = Eapp()
        self.motionEncoder = Emtn()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512) # source to canonical
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512) # canonical to driving 
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

        self.image_pyramid = ImagePyramide(scales=[0.5, 0.25], num_channels=3)

#    @profile
    def forward(self, xs, xd):
        vs, es = self.appearanceEncoder(xs)
   
        Rs, ts, zs = self.motionEncoder(xs)
        Rd, td, zd = self.motionEncoder(xd)

        logging.debug(f"es shape:{es.shape}")
        logging.debug(f"zs shape:{zs.shape}")


        w_s2c = self.warp_generator_s2c(Rs, ts, zs, es)


        logging.debug(f"vs shape:{vs.shape}") 
        vc = apply_warping_field(vs, w_s2c)
        assert vc.shape[1:] == (96, 16, 64, 64), f"Expected vc shape (_, 96, 16, 64, 64), got {vc.shape}"

        vc2d = self.G3d(vc)

        w_c2d = self.warp_generator_c2d(Rd, td, zd, es)
        logging.debug(f"w_c2d shape:{w_c2d.shape}") 

        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        assert vc2d_warped.shape[1:] == (96, 16, 64, 64), f"Expected vc2d_warped shape (_, 96, 16, 64, 64), got {vc2d_warped.shape}"

        vc2d_projected = torch.sum(vc2d_warped, dim=2)

        xhat_base = self.G2d(vc2d_projected)

        #self.visualize_warp_fields(xs, xd, w_s2c, w_c2d, Rs, ts, Rd, td)
       
        pyramids = self.image_pyramid(xhat_base)

        return xhat_base, pyramids

    def visualize_warp_fields(self, xs, xd, w_s2c, w_c2d, Rs, ts, Rd, td):

        pitch_s, yaw_s, roll_s = Rs[:, 0], Rs[:, 1], Rs[:, 2]
        pitch_d, yaw_d, roll_d = Rd[:, 0], Rd[:, 1], Rd[:, 2]

        logging.debug(f"Source Image Pitch: {pitch_s}, Yaw: {yaw_s}, Roll: {roll_s}")
        logging.debug(f"Driving Image Pitch: {pitch_d}, Yaw: {yaw_d}, Roll: {roll_d}")

        fig = plt.figure(figsize=(15, 10))

        source_image = xs[0].permute(1, 2, 0).cpu().detach().numpy()
        driving_image = xd[0].permute(1, 2, 0).cpu().detach().numpy()



        # source_image = self.draw_axis(source_image, Rs[0,1], Rs[0,0], Rs[0,2])
        # driving_image = self.draw_axis(driving_image, Rd[0,1], Rd[0,0], Rd[0,2])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(source_image)
        axs[0].set_title('Source Image with Axes')
        axs[0].axis('off')

        axs[1].imshow(driving_image)
        axs[1].set_title('Driving Image with Axes')
        axs[1].axis('off')


        ax_w_s2c = fig.add_subplot(2, 3, 4, projection='3d')
        self.plot_warp_field(ax_w_s2c, w_s2c, 'w_s2c Warp Field')

        ax_w_c2d = fig.add_subplot(2, 3, 3, projection='3d')
        self.plot_warp_field(ax_w_c2d, w_c2d, 'w_c2d Warp Field')


        # pitch = Rs[0,1].cpu().detach().numpy() * np.pi / 180
        # yaw = -(Rs[0,0].cpu().detach().numpy() * np.pi / 180)
        # roll = Rs[0,2].cpu().detach().numpy() * np.pi / 180

        # ax_rotations_s = fig.add_subplot(2, 3, 5, projection='3d')
        # self.plot_rotations(ax_rotations_s, pitch,yaw,roll, 'Canonical Head Rotations')


        # pitch = Rd[0,1].cpu().detach().numpy() * np.pi / 180
        # yaw = -(Rd[0,0].cpu().detach().numpy() * np.pi / 180)
        # roll = Rd[0,2].cpu().detach().numpy() * np.pi / 180

        # ax_rotations_d = fig.add_subplot(2, 3, 6, projection='3d')
        # self.plot_rotations(ax_rotations_d, pitch,yaw,roll, 'Driving Head Rotations') 

        plt.tight_layout()
        plt.show()

    def plot_rotations(ax,pitch,yaw, roll,title,bla):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.set_aspect('auto')

        tdx, tdy, tdz = 0, 0, 0

        pitch = pitch * np.pi / 180
        yaw = yaw * np.pi / 180
        roll = roll * np.pi / 180

        x_axis = np.array([np.cos(yaw) * np.cos(roll),
                        np.cos(pitch) * np.sin(roll) + np.sin(pitch) * np.sin(yaw) * np.cos(roll),
                        np.sin(yaw)])
        y_axis = np.array([-np.cos(yaw) * np.sin(roll),
                        np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll),
                        -np.cos(yaw) * np.sin(pitch)])
        z_axis = np.array([np.sin(yaw),
                        -np.cos(yaw) * np.sin(pitch),
                        np.cos(pitch)])

        axis_length = 1

        ax.quiver(tdx, tdy, tdz, x_axis[0], x_axis[1], x_axis[2], color='r', length=axis_length, label='X-axis')
        ax.quiver(tdx, tdy, tdz, y_axis[0], y_axis[1], y_axis[2], color='g', length=axis_length, label='Y-axis')
        ax.quiver(tdx, tdy, tdz, z_axis[0], z_axis[1], z_axis[2], color='b', length=axis_length, label='Z-axis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(title)



    def plot_warp_field(self, ax, warp_field, title, sample_rate=3):

        warp_field_np = warp_field.detach().cpu().numpy()[0]  

        depth, height, width = warp_field_np.shape[1:]

        x = np.arange(0, width, sample_rate)
        y = np.arange(0, height, sample_rate)
        z = np.arange(0, depth, sample_rate)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        U = warp_field_np[0, ::sample_rate, ::sample_rate, ::sample_rate]
        V = warp_field_np[1, ::sample_rate, ::sample_rate, ::sample_rate]
        W = warp_field_np[2, ::sample_rate, ::sample_rate, ::sample_rate]

        mask_pos = (U > 0) | (V > 0) | (W > 0)
        mask_neg = (U < 0) | (V < 0) | (W < 0)

        color_pos = 'red'
        color_neg = 'blue'

        ax.quiver3D(X[mask_pos], Y[mask_pos], Z[mask_pos], U[mask_pos], V[mask_pos], W[mask_pos],
                    color=color_pos, length=0.3, normalize=True)

        ax.quiver3D(X[mask_neg], Y[mask_neg], Z[mask_neg], U[mask_neg], V[mask_neg], W[mask_neg],
                    color=color_neg, length=0.3, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)