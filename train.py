import argparse
import sys
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
from omegaconf import OmegaConf
import mediapipe as mp
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils
import time
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from lpips import LPIPS


from models.perceptual_loss import PerceptualLoss 
from models.crop_warp_face import crop_and_warp_face
from models.remove_bg_rgb import remove_background_and_convert_to_rgb
from models.apply_warping import apply_warping_field
from models.forground_mask import get_foreground_mask
from models.gbase import Gbase
from models.disc import MultiscaleDiscriminator
from models.pairwise_transfer import PairwiseTransferLoss




from torch.utils.tensorboard import SummaryWriter

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
    lpips_model = LPIPS(net='alex').cuda()  # 'alex', 'vgg', 'squeeze'
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

def train_base(cfg, Gbase, Dbase, dataloader, dataset,start_epoch=0):
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
                video_id = batch['video_id'][0]

                # Access videos from dataloader2 for cycle consistency
                source_frames2 = batch['source_frames_star']
                driving_frames2 = batch['driving_frames_star']
                video_id2 = batch['video_id_star'][0]


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

                         
                        pred_frame,pred_pyramids = Gbase(source_frame, driving_frame)

                        

                        save_images = True
                        if save_images:
                            # vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
                            # vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
                            vutils.save_image(pred_frame, f"{output_dir}/pred_frame_{idx}.png")
                            # vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
                            # vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
                            # vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
                            # vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

                        # Calculate perceptual losses - use pyramid 
                        # loss_G_per = perceptual_loss_fn(pred_frame, source_frame)
                      
                        loss_G_per = 0
                        for scale, pred_scaled in pred_pyramids.items():
                            target_scaled = F.interpolate(driving_frame, size=pred_scaled.shape[2:], mode='bilinear', align_corners=False)
                            loss_G_per += perceptual_loss_fn(pred_scaled, target_scaled)

                  
                        # real loss
                        real_preds = Dbase(driving_frame, source_frame)
                        fake_preds = Dbase(pred_frame.detach(), source_frame)

                        valid_list = [Variable(torch.Tensor(np.ones(pred.shape)).to(device), requires_grad=False) for pred in real_preds]
                        fake_list = [Variable(torch.Tensor(-1 * np.ones(pred.shape)).to(device), requires_grad=False) for pred in fake_preds]

                        loss_real = sum([hinge_loss(real_pred, valid) for real_pred, valid in zip(real_preds, valid_list)])
                        loss_fake = sum([hinge_loss(fake_pred, fake) for fake_pred, fake in zip(fake_preds, fake_list)])

   
                        # # fake loss
                        # fake_pred = Dbase(pred_frame.detach(), source_frame)
                        # loss_fake = hinge_loss(fake_pred, fake)

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
                        
                        
                        cross_reenacted_image,_ = Gbase(source_frame_star, driving_frame)
                        if save_images:
                            vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

                        
                        _, _, z_pred = Gbase.motionEncoder(pred_frame) 
                        _, _, zd = Gbase.motionEncoder(driving_frame) 
                        
                        _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                        _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

              
                       

                        P = [(z_pred, zd)     ,(z_star__pred, zd)]
                        N = [(z_pred, zd_star),(z_star__pred, zd_star)]
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





        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        
        writer.add_scalar('Loss/Generator', avg_loss_G, epoch)
        writer.add_scalar('Loss/Discriminator', avg_loss_D, epoch)


        writer.add_scalar('FID Score', fid_score, epoch)
        writer.add_scalar('CSIM Score', csim_score, epoch)
        writer.add_scalar('LPIPS Score', lpips_score, epoch)


        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G_cos.item():.4f}, Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_G_state_dict': Gbase.state_dict(),
                'model_D_state_dict': Dbase.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"checkpoint_epoch{epoch+1}.pth")


        # Progressively load more videos after some epochs
        # if (epoch + 1) % 5 == 0:
        #     dataset.add_more_videos()
            
        # Calculate FID score for the current epoch
        # with torch.no_grad():
        #     real_images = torch.cat(real_preds)
        #     fake_images = torch.cat(fake_preds)
        #     fid_score = calculate_fid(real_images, fake_images)
        #     csim_score = calculate_csim(real_images, fake_images)
        #     lpips_score = calculate_lpips(real_images, fake_images)
  
        #     writer.add_scalar('FID Score', fid_score, epoch)
        #     writer.add_scalar('CSIM Score', csim_score, epoch)
        #     writer.add_scalar('LPIPS Score', lpips_score, epoch)



def load_checkpoint(model_G_path, model_D_path, model_G, model_D):
    start_epoch = 0
    if os.path.isfile(model_G_path):
        print(f"Loading generator model from '{model_G_path}'")
        checkpoint_G = torch.load(model_G_path)
        model_G.load_state_dict(checkpoint_G)
    else:
        print(f"No generator model found at '{model_G_path}'")
    
    if os.path.isfile(model_D_path):
        print(f"Loading discriminator model from '{model_D_path}'")
        checkpoint_D = torch.load(model_D_path)
        model_D.load_state_dict(checkpoint_D)
    else:
        print(f"No discriminator model found at '{model_D_path}'")
    
    return start_epoch

from models.disc import SingleScaleDiscriminator
from models.disc import Discriminator

def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        remove_background=True,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform,
        apply_crop_warping=True
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    Gbase_model = Gbase().to(device)
    Dbase_model = MultiscaleDiscriminator().to(device)
    # Dbase_model = Discriminator().to(device)


    optimizer_G = torch.optim.AdamW(Gbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase_model.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)

    start_epoch = load_checkpoint('Gbase.pth', 'Dbase.pth', Gbase_model, Dbase_model)

    train_base(cfg, Gbase_model, Dbase_model, dataloader, start_epoch)

    torch.save(Gbase_model.state_dict(), 'Gbase.pth')
    torch.save(Dbase_model.state_dict(), 'Dbase.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/train_config.yaml")
    main(config)
