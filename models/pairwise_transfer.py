import torch
from torch import nn
from torch.nn import functional as F
from typing import List
import torchvision.models as models
import numpy as np
from models.apply_warping import apply_warping_field
from facenet_pytorch import InceptionResnetV1

class PairwiseTransferLoss(nn.Module):
    def __init__(self):
        super(PairwiseTransferLoss, self).__init__()

    def forward(self, Gbase, I1, I2):
        vs1, es1 = Gbase.appearanceEncoder(I1)
        vs2, es2 = Gbase.appearanceEncoder(I2)

        Rs1, ts1, zs1 = Gbase.motionEncoder(I1)
        Rs2, ts2, zs2 = Gbase.motionEncoder(I2)

        w_s2c_pose = Gbase.warp_generator_s2c(Rs2, ts2, zs1, es1)
        vc_pose = apply_warping_field(vs1, w_s2c_pose)
        vc2d_pose = Gbase.G3d(vc_pose)
        w_c2d_pose = Gbase.warp_generator_c2d(Rs2, ts2, zs1, es1)
        vc2d_warped_pose = apply_warping_field(vc2d_pose, w_c2d_pose)
        vc2d_projected_pose = torch.sum(vc2d_warped_pose, dim=2)
        I_pose = Gbase.G2d(vc2d_projected_pose)

        w_s2c_exp = Gbase.warp_generator_s2c(Rs1, ts1, zs2, es1)
        vc_exp = apply_warping_field(vs1, w_s2c_exp)
        vc2d_exp = Gbase.G3d(vc_exp)
        w_c2d_exp = Gbase.warp_generator_c2d(Rs1, ts1, zs2, es1)
        vc2d_warped_exp = apply_warping_field(vc2d_exp, w_c2d_exp)
        vc2d_projected_exp = torch.sum(vc2d_warped_exp, dim=2)
        I_exp = Gbase.G2d(vc2d_projected_exp)

        loss = F.l1_loss(I_pose, I_exp)

        return loss



class IdentitySimilarityLoss(nn.Module):
    def __init__(self):
        super(IdentitySimilarityLoss, self).__init__()
        self.face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, model, I1, I2):
        I_transfer = model(I1, I2)

        with torch.no_grad():
            id_features1 = self.face_recognition_model(I1)
            id_features_transfer = self.face_recognition_model(I_transfer)

        similarity = F.cosine_similarity(id_features1, id_features_transfer)

        loss = -similarity.mean()

        return loss