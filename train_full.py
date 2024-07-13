import argparse
import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.EmoDataset import EMODataset
import torch.nn.functional as F
import decord
from omegaconf import OmegaConf
from torchvision import models
from models.mpgaze_loss import MPGazeLoss
from models.patchgan_enc import PatchGanEncoder
from models.meta_loss import Vgg19
import cv2
import mediapipe as mp
from memory_profiler import profile
import os
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import autocast, GradScaler
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.utils as vutils
from lpips import LPIPS
from torch.utils.tensorboard import SummaryWriter

from models.gbase import Gbase
from models.disc import MultiscaleDiscriminator
from models.high_res import GHR
from models.student import Student
from models.perceptual_loss import PerceptualLoss 
from models.crop_warp_face import crop_and_warp_face
from models.remove_bg_rgb import remove_background_and_convert_to_rgb
from models.apply_warping import apply_warping_field
from models.forground_mask import get_foreground_mask
from models.pairwise_transfer import PairwiseTransferLoss
# from models.identity_sim_loss import IdentitySimilarityLoss
from models.highres_dataset import HighResDataset

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def calculate_fid(real_images, fake_images):
    real_images = real_images.detach().cpu().numpy()
    fake_images = fake_images.detach().cpu().numpy()
    mu1, sigma1 = real_images.mean(axis=0), np.cov(real_images, rowvar=False)
    mu2, sigma2 = fake_images.mean(axis=0), np.cov(fake_images, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_csim(real_features, fake_features):
    csim = cosine_similarity(real_features.detach().cpu().numpy(), fake_features.detach().cpu().numpy())
    return np.mean(csim)

def calculate_lpips(real_images, fake_images):
    lpips_model = LPIPS(net='alex').cuda()
    lpips_scores = []
    for real, fake in zip(real_images, fake_images):
        real = real.unsqueeze(0).cuda()
        fake = fake.unsqueeze(0).cuda()
        lpips_score = lpips_model(real, fake)
        lpips_scores.append(lpips_score.item())
    return np.mean(lpips_scores)

def discriminator_loss(real_pred, fake_pred, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = torch.mean((real_pred - 1)**2)
        fake_loss = torch.mean(fake_pred**2)
    elif loss_type == 'vanilla':
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    return ((real_loss + fake_loss) * 0.5).requires_grad_()

def multiscale_discriminator_loss(real_preds, fake_preds, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = sum(torch.mean((real_pred - 1)**2) for real_pred in real_preds)
        fake_loss = sum(torch.mean(fake_pred**2) for fake_pred in fake_preds)
    elif loss_type == 'vanilla':
        real_loss = sum(F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) for real_pred in real_preds)
        fake_loss = sum(F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred)) for fake_pred in fake_preds)
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    return ((real_loss + fake_loss) * 0.5).requires_grad_()

def cosine_loss(positive_pairs, negative_pairs, margin=0.5, scale=5):
    def cosine_distance(z_i, z_j):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        cos_sim = torch.sum(z_i * z_j, dim=-1)
        cos_dist = scale * (cos_sim - margin)
        return cos_dist
    pos_cos_dist = [cosine_distance(z_i, z_j) for z_i, z_j in positive_pairs]
    pos_cos_dist = torch.stack(pos_cos_dist)
    neg_cos_dist = [cosine_distance(z_i, z_j) for z_i, z_j in negative_pairs]
    neg_cos_dist = torch.stack(neg_cos_dist)
    loss = -torch.log(torch.exp(pos_cos_dist) / (torch.exp(pos_cos_dist) + torch.sum(torch.exp(neg_cos_dist))))
    return loss.mean().requires_grad_()

def gaze_loss_fn(predicted_gaze, target_gaze, face_image):
    if face_image.dim() == 4 and face_image.shape[0] == 1:
        face_image = face_image.squeeze(0)
    if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
        raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
    
    face_image = face_image.detach().cpu().numpy()
    if face_image.shape[0] == 3:
        face_image = face_image.transpose(1, 2, 0)
    face_image = (face_image * 255).astype(np.uint8)

    results = face_mesh.process(cv2.cvtColor(face_image, cv.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return torch.tensor(0.0, requires_grad=True).to(device)

    eye_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        left_eye_landmarks = [face_landmarks.landmark[idx] for idx in range(len(mp.solutions.face_mesh.FACEMESH_LEFT_EYE))]
        right_eye_landmarks = [face_landmarks.landmark[idx] for idx in range(len(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE))]
        eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

    loss = 0.0
    h, w = face_image.shape[:2]
    for left_eye, right_eye in eye_landmarks:
        left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
        right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

        left_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        right_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        cv2.fillPoly(left_mask[0].detach().cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
        cv2.fillPoly(right_mask[0].detach().cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

        left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
        right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
        loss += left_gaze_loss + right_gaze_loss

    return loss / len(eye_landmarks)

def adversarial_loss(output_frame, driving_frame, discriminator):
    fake_preds = discriminator(output_frame, driving_frame)
    loss = 0
    for fake_pred in fake_preds:
        loss += F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    loss /= len(fake_preds)
    return loss.requires_grad_()

def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss.requires_grad_()

def contrastive_loss(output_frame, source_frame, driving_frame, encoder, margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out, requires_grad=True)
    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for pos_pair in pos_pairs:
        loss = loss + torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) /
                                (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) +
                                 neg_pair_loss(pos_pair, neg_pairs, margin)))
    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for neg_pair in neg_pairs:
        loss = loss + torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss

# def train_base(cfg, Gbase, Dbase, dataloader, dataset, start_epoch=0):
#     patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
#     hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
#     feature_matching_loss = nn.MSELoss()
#     Gbase.train()
#     Dbase.train()
#     optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
#     optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
#     scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
#     scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

#     perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})
#     # pairwise_transfer_loss = PairwiseTransferLoss()
#     #  identity_similarity_loss = IdentitySimilarityLoss()
#     # identity_similarity_loss = PerceptualLoss(device, weights={'vgg19': 0.0, 'vggface': 1.0, 'gaze': 0.0,'lpips':0.0}) # focus on face

#     scaler = GradScaler()
#     writer = SummaryWriter(log_dir='runs/training_logs')

#     for epoch in range(start_epoch, cfg.training.base_epochs):
#         print("Epoch:", epoch)

#         epoch_loss_G = 0
#         epoch_loss_D = 0

#         fid_score = 0
#         csim_score = 0
#         lpips_score = 0

#         for batch in dataloader:
#             source_frames = batch['source_frames']
#             driving_frames = batch['driving_frames']
#             # video_id = batch['video_id'][0]

#             # Access videos from dataloader2 for cycle consistency
#             source_frames2 = batch['source_frames_star']
#             driving_frames2 = batch['driving_frames_star']
#             # video_id2 = batch['video_id_star'][0]

#             num_frames = len(driving_frames)
#             len_source_frames = len(source_frames)
#             len_driving_frames = len(driving_frames)
#             len_source_frames2 = len(source_frames2)
#             len_driving_frames2 = len(driving_frames2)

#             for idx in range(num_frames):
#                 source_frame = source_frames[idx % len_source_frames].to(device)
#                 driving_frame = driving_frames[idx % len_driving_frames].to(device)

#                 source_frame_star = source_frames2[idx % len_source_frames2].to(device)
#                 driving_frame_star = driving_frames2[idx % len_driving_frames2].to(device)

#                 with autocast():
#                     pred_frame, pred_pyramids = Gbase(source_frame, driving_frame)

#                     foreground_mask = get_foreground_mask(source_frame)
#                     foreground_mask = foreground_mask.to(pred_frame.device)
#                     masked_predicted_image = pred_frame * foreground_mask
#                     masked_target_image = driving_frame * foreground_mask

#                     save_images = True
#                     if save_images:
#                         vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
#                         vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
#                         vutils.save_image(pred_frame, f"{output_dir}/pred_frame_{idx}.png")
#                         vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
#                         vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
#                         vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
#                         vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

#                     loss_G_per = 0
#                     for scale, pred_scaled in pred_pyramids.items():
#                         target_scaled = F.interpolate(driving_frame, size=pred_scaled.shape[2:], mode='bilinear', align_corners=False)
#                         loss_G_per += perceptual_loss_fn(pred_scaled, target_scaled)

#                     real_preds = Dbase(driving_frame, source_frame)
#                     fake_preds = Dbase(pred_frame.detach(), source_frame)

#                     valid_list = [Variable(torch.Tensor(np.ones(pred.shape)).to(device), requires_grad=False) for pred in real_preds]
#                     fake_list = [Variable(torch.Tensor(-1 * np.ones(pred.shape)).to(device), requires_grad=False) for pred in fake_preds]

#                     loss_real = sum([hinge_loss(real_pred, valid) for real_pred, valid in zip(real_preds, valid_list)])
#                     loss_fake = sum([hinge_loss(fake_pred, fake) for fake_pred, fake in zip(fake_preds, fake_list)])

#                     optimizer_D.zero_grad()
#                     real_preds = Dbase(driving_frame, source_frame)
#                     fake_preds = Dbase(pred_frame.detach(), source_frame)
#                     loss_D = multiscale_discriminator_loss(real_preds, fake_preds, loss_type='lsgan')

#                     scaler.scale(loss_D).backward()
#                     scaler.step(optimizer_D)
#                     scaler.update()

#                     loss_G_adv = 0.5 * (loss_real + loss_fake)
#                     loss_fm = feature_matching_loss(pred_frame, driving_frame)
#                     writer.add_scalar('Loss/Feature Matching', loss_fm, epoch)

#                     # New disentangling losses - from VASA paper
#                     # I1 and I2 are from the same video, I3 and I4 are from different videos

#                     # Get the next frame index, wrapping around if necessary
#                     next_idx = (idx + 20) % len_source_frames

#                     I1 = source_frame
#                     I2 = source_frames[next_idx].to(device)
#                     I3 = source_frame_star
#                     I4 = source_frames2[next_idx % len_source_frames2].to(device)
#                     # loss_pairwise = pairwise_transfer_loss(Gbase,I1, I2)
#                     # loss_identity = identity_similarity_loss(I3, I4)


#                     # writer.add_scalar('pairwise_transfer_loss', loss_pairwise, epoch)
#                     # writer.add_scalar('identity_similarity_loss', loss_identity, epoch)

#                     cross_reenacted_image, _ = Gbase(source_frame_star, driving_frame)
#                     if save_images:
#                         vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

#                     _, _, z_pred = Gbase.motionEncoder(pred_frame) 
#                     _, _, zd = Gbase.motionEncoder(driving_frame) 

#                     _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
#                     _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

#                     P = [(z_pred, zd), (z_star__pred, zd)]
#                     N = [(z_pred, zd_star), (z_star__pred, zd_star)]
#                     loss_G_cos = cosine_loss(P, N)

#                     # # New disentanglement losses
#                     # loss_pairwise = pairwise_transfer_loss(Gbase, source_frame, driving_frame)
#                     # loss_identity = identity_similarity_loss(Gbase, source_frame, driving_frame_star)
#                     writer.add_scalar('Cycle consistency loss', loss_G_cos, epoch)

#                     optimizer_G.zero_grad()

#                     total_loss = cfg.training.w_per * loss_G_per + \
#                                  cfg.training.w_adv * loss_G_adv + \
#                                  cfg.training.w_fm * loss_fm + \
#                                  cfg.training.w_cos * loss_G_cos #+ \
#                                 #  cfg.training.w_pairwise * loss_pairwise + \
#                                 #  cfg.training.w_identity * loss_identity

#                     scaler.scale(total_loss).backward()
#                     scaler.step(optimizer_G)
#                     scaler.update()

#                     epoch_loss_G += total_loss.item()
#                     epoch_loss_D += loss_D.item()

#                     writer.add_scalar('Loss/Total Generator', epoch_loss_G, epoch)
#                     writer.add_scalar('Loss/Total Discriminator', epoch_loss_D, epoch)

#         fid_score /= num_frames
#         csim_score /= num_frames
#         lpips_score /= num_frames
#         print(f"Epoch {epoch} completed. Generator loss: {epoch_loss_G}, Discriminator loss: {epoch_loss_D}")
#         print(f"FID score: {fid_score}, CSIM score: {csim_score}, LPIPS score: {lpips_score}")

#         if (epoch + 1) % cfg.training.log_interval == 0:
#             print(f"Epoch [{epoch + 1}/{cfg.training.base_epochs}], "
#                   f"Loss_G: {epoch_loss_G:.4f}, Loss_D: {epoch_loss_D:.4f}")
#         if (epoch + 1) % cfg.training.save_interval == 0:
#             torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch + 1}.pth")
#             torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch + 1}.pth")
#         scheduler_G.step()
#         scheduler_D.step()
#     writer.close()
def train_base(cfg, Gbase, Dbase, dataloader, dataset, start_epoch=0):
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})
    scaler = GradScaler()
    writer = SummaryWriter(log_dir='runs/training_logs')

    for epoch in range(start_epoch, cfg.training.base_epochs):
        print("Epoch:", epoch)

        epoch_loss_G = 0
        epoch_loss_D = 0

        fid_score = 0
        csim_score = 0
        lpips_score = 0

        for batch in dataloader:
            source_frames = batch['source_frames']
            driving_frames = batch['driving_frames']
            source_frames2 = batch['source_frames_star']
            driving_frames2 = batch['driving_frames_star']

            num_frames = len(driving_frames)
            len_source_frames = len(source_frames)
            len_driving_frames = len(driving_frames)
            len_source_frames2 = len(source_frames2)
            len_driving_frames2 = len(driving_frames2)

            for idx in range(num_frames):
                source_frame = source_frames[idx % len_source_frames].to(device)
                driving_frame = driving_frames[idx % len_driving_frames].to(device)

                source_frame_star = source_frames2[idx % len_source_frames2].to(device)
                driving_frame_star = driving_frames2[idx % len_driving_frames2].to(device)

                with autocast():
                    pred_frame, pred_pyramids = Gbase(source_frame, driving_frame)

                    foreground_mask = get_foreground_mask(source_frame)
                    foreground_mask = foreground_mask.to(pred_frame.device)
                    masked_predicted_image = pred_frame * foreground_mask
                    masked_target_image = driving_frame * foreground_mask

                    save_images = True
                    if save_images:
                        vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
                        vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
                        vutils.save_image(pred_frame, f"{output_dir}/pred_frame_{idx}.png")
                        vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
                        vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
                        vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
                        vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

                    loss_G_per = 0
                    for scale, pred_scaled in pred_pyramids.items():
                        target_scaled = F.interpolate(driving_frame, size=pred_scaled.shape[2:], mode='bilinear', align_corners=False)
                        loss_G_per += perceptual_loss_fn(pred_scaled, target_scaled)

                    real_preds = Dbase(driving_frame, source_frame)
                    fake_preds = Dbase(pred_frame.detach(), source_frame)

                    valid_list = [Variable(torch.Tensor(np.ones(pred.shape)).to(device), requires_grad=False) for pred in real_preds]
                    fake_list = [Variable(torch.Tensor(-1 * np.ones(pred.shape)).to(device), requires_grad=False) for pred in fake_preds]

                    loss_real = sum([hinge_loss(real_pred, valid) for real_pred, valid in zip(real_preds, valid_list)])
                    loss_fake = sum([hinge_loss(fake_pred, fake) for fake_pred, fake in zip(fake_preds, fake_list)])

                    optimizer_D.zero_grad()
                    real_preds = Dbase(driving_frame, source_frame)
                    fake_preds = Dbase(pred_frame.detach(), source_frame)
                    loss_D = multiscale_discriminator_loss(real_preds, fake_preds, loss_type='lsgan')

                    scaler.scale(loss_D).backward()
                    scaler.step(optimizer_D)
                    scaler.update()

                    loss_G_adv = 0.5 * (loss_real + loss_fake)
                    loss_fm = feature_matching_loss(pred_frame, driving_frame)
                    writer.add_scalar('Loss/Feature Matching', loss_fm, epoch)

                    next_idx = (idx + 20) % len_source_frames

                    I1 = source_frame
                    I2 = source_frames[next_idx].to(device)
                    I3 = source_frame_star
                    I4 = source_frames2[next_idx % len_source_frames2].to(device)

                    cross_reenacted_image, _ = Gbase(source_frame_star, driving_frame)
                    if save_images:
                        vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

                    _, _, z_pred = Gbase.motionEncoder(pred_frame) 
                    _, _, zd = Gbase.motionEncoder(driving_frame) 

                    _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                    _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

                    P = [(z_pred, zd), (z_star__pred, zd)]
                    N = [(z_pred, zd_star), (z_star__pred, zd_star)]
                    loss_G_cos = cosine_loss(P, N)

                    writer.add_scalar('Cycle consistency loss', loss_G_cos, epoch)

                    optimizer_G.zero_grad()

                    total_loss = cfg.training.w_per * loss_G_per + \
                                 cfg.training.w_adv * loss_G_adv + \
                                 cfg.training.w_fm * loss_fm + \
                                 cfg.training.w_cos * loss_G_cos

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer_G)
                    scaler.update()

                    epoch_loss_G += total_loss.item()
                    epoch_loss_D += loss_D.item()

                    writer.add_scalar('Loss/Total Generator', epoch_loss_G, epoch)
                    writer.add_scalar('Loss/Total Discriminator', epoch_loss_D, epoch)

        fid_score /= num_frames
        csim_score /= num_frames
        lpips_score /= num_frames
        print(f"Epoch {epoch} completed. Generator loss: {epoch_loss_G}, Discriminator loss: {epoch_loss_D}")
        print(f"FID score: {fid_score}, CSIM score: {csim_score}, LPIPS score: {lpips_score}")

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch + 1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {epoch_loss_G:.4f}, Loss_D: {epoch_loss_D:.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch + 1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch + 1}.pth")
        scheduler_G.step()
        scheduler_D.step()
    writer.close()

def train_hr(cfg, GHR, dataloader_hr):
    print("Starting training high res")
    GHR.train()
    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    optimizer_G = torch.optim.AdamW(GHR.Genh.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.hr_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.hr_epochs):
        print(f'epoch number {epoch}')
        for batch in dataloader_hr:
            source_frames = torch.stack(batch['source_frames']).to(device)
            driving_frames = torch.stack(batch['driving_frames']).to(device)

            num_frames = len(source_frames)
            for idx in range(num_frames):
                source_frame = source_frames[idx]
                driving_frame = driving_frames[idx]

                with torch.no_grad():
                    xhat_base, _ = GHR.Gbase(source_frame, driving_frame)  # Extract the tensor

                optimizer_G.zero_grad()
                xhat_hr = GHR.Genh(xhat_base)

                xhat_hr_vgg_features = vgg19(xhat_hr)
                driving_vgg_features = vgg19(driving_frame)
                loss_perceptual = 0
                for xhat_hr_feat, driving_feat in zip(xhat_hr_vgg_features, driving_vgg_features):
                    loss_perceptual += perceptual_loss_fn(xhat_hr_feat, driving_feat.detach())

                loss_supervised = perceptual_loss_fn(xhat_hr, driving_frame)
                loss_unsupervised = perceptual_loss_fn(xhat_hr, xhat_base)
                loss_gaze = gaze_loss_fn(xhat_hr, driving_frame, source_frame)  # Pass the face_image argument
                loss_G = (cfg.training.lambda_supervised * loss_supervised
                          + cfg.training.lambda_unsupervised * loss_unsupervised
                          + cfg.training.lambda_perceptual * loss_perceptual
                          + cfg.training.lambda_gaze * loss_gaze)
                
                loss_G.backward()
                optimizer_G.step()

        scheduler_G.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.hr_epochs}], Loss_G: {loss_G.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(GHR.Genh.state_dict(), f"Genh_epoch{epoch+1}.pth")
    print("done training high res")

def find_latest_checkpoint(checkpoint_dir, base_name):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(base_name)]
    if not checkpoint_files:
        return None
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_epoch')[1].split('.pth')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])


# def load_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D):
#     if os.path.isfile(checkpoint_path):
#         print(f"Loading checkpoint '{checkpoint_path}'")
#         checkpoint = torch.load(checkpoint_path)
#         model_G.load_state_dict(checkpoint['model_G_state_dict'])
#         model_D.load_state_dict(checkpoint['model_D_state_dict'])
#         optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
#         optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
#     else:
#         print(f"No checkpoint found at '{checkpoint_path}'")
#         start_epoch = 0
#     return start_epoch
def load_checkpoint(model_G, model_D, epoch):
    G_checkpoint_path = f"Gbase_epoch{epoch}.pth"
    D_checkpoint_path = f"Dbase_epoch{epoch}.pth"

    if os.path.isfile(G_checkpoint_path) and os.path.isfile(D_checkpoint_path):
        print(f"Loading checkpoints '{G_checkpoint_path}' and '{D_checkpoint_path}'")
        model_G.load_state_dict(torch.load(G_checkpoint_path))
        model_D.load_state_dict(torch.load(D_checkpoint_path))
        print(f"Loaded checkpoints (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoints found for epoch {epoch}")
        return 0

def train_student(cfg, Student, GHR, dataloader_avatars):
    print("starting training student")
    Student.train()
    optimizer_S = torch.optim.AdamW(Student.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_S = CosineAnnealingLR(optimizer_S, T_max=cfg.training.student_epochs, eta_min=1e-6)
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            avatar_indices = batch['avatar_indices'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            with torch.no_grad():
                xhat_hr = GHR(driving_frames)
            optimizer_S.zero_grad()
            xhat_student = Student(driving_frames, avatar_indices)
            loss_S = F.mse_loss(xhat_student, xhat_hr)
            loss_S.backward()
            optimizer_S.step()
        scheduler_S.step()
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.student_epochs}], Loss_S: {loss_S.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Student.state_dict(), f"Student_epoch{epoch+1}.pth")


# def main(cfg: OmegaConf) -> None:
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter()
#     ])

#     base_dataset = EMODataset(
#         use_gpu=use_cuda,
#         width=cfg.data.train_width,
#         height=cfg.data.train_height,
#         n_sample_frames=cfg.training.n_sample_frames,
#         sample_rate=cfg.training.sample_rate,
#         img_scale=(1.0, 1.0),
#         video_dir=cfg.training.video_dir,
#         json_file=cfg.training.json_file,
#         transform=transform
#     )

#     base_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

#     Gbase_model = Gbase().to(device)
#     Dbase_model = MultiscaleDiscriminator().to(device)
#     optimizer_G = torch.optim.AdamW(Gbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
#     optimizer_D = torch.optim.AdamW(Dbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)

#     checkpoint_path = find_latest_checkpoint(cfg.training.checkpoint_path, 'checkpoint_epoch')
#     start_epoch = 0
#     if checkpoint_path:
#         start_epoch = load_checkpoint(checkpoint_path, Gbase_model, Dbase_model, optimizer_G, optimizer_D)

#     torch.cuda.empty_cache()

#     train_base(cfg, Gbase_model, Dbase_model, base_dataloader, base_dataset, start_epoch)

#     torch.save(Gbase_model.state_dict(), 'Gbase_trained.pth')
def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter()
    ])

    base_dataset = EMODataset(
        use_gpu=use_cuda,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    base_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    Gbase_model = Gbase().to(device)
    Dbase_model = MultiscaleDiscriminator().to(device)
    optimizer_G = torch.optim.AdamW(Gbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)

    start_epoch = load_checkpoint(Gbase_model, Dbase_model, cfg.training.load_epoch)
    torch.cuda.empty_cache()

    train_base(cfg, Gbase_model, Dbase_model, base_dataloader, base_dataset, start_epoch)

    torch.save(Gbase_model.state_dict(), 'Gbase_trained.pth')
    torch.save(Dbase_model.state_dict(), 'Dbase_trained.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)


    # # Lazy initialization of high-resolution model
    # del Gbase_model, Dbase_model, optimizer_G, optimizer_D  # Free up memory
    # torch.cuda.empty_cache()

    # # Initialize HighResDataset for high-resolution model training
    # highres_dataset = HighResDataset(
    #     image_dir=cfg.data.highres_image_dir,
    #     json_file=cfg.data.highres_json_file,
    #     transform=transform
    # )

    # highres_dataloader = DataLoader(highres_dataset, batch_size=2, shuffle=True, num_workers=4)

    # GHR_model = GHR().to(device)
    # GHR_model.Gbase.load_state_dict(torch.load('Gbase_trained.pth'))  # Load pretrained weights
    # train_hr(cfg, GHR_model, highres_dataloader)

    # # Save weights for the high-resolution model
    # torch.save(GHR_model.state_dict(), 'GHR_trained.pth')

    # # Lazy initialization of student model
    # del GHR_model  # Free up memory
    # torch.cuda.empty_cache()

    # Student_model = Student(num_avatars=4).to(device)
    # train_student(cfg, Student_model, GHR_model, highres_dataloader)
    
    # torch.save(Student_model.state_dict(), 'Student_trained.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/train_config.yaml")
    main(config)