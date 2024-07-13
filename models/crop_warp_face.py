import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
import numpy as np

from rembg import remove


def crop_and_warp_face(image_tensor, video_name, frame_idx, output_dir="output_images", pad_to_original=False, apply_warping=True,warp_strength=0.05):

    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")

    if os.path.exists(output_path):

        existing_image = Image.open(output_path).convert("RGBA")
        return to_tensor(existing_image)
    
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)
    
    image = to_pil_image(image_tensor)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    bg_removed_bytes = remove(img_byte_arr)
    bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
    
    bg_removed_image_rgb = bg_removed_image.convert("RGB")

    face_locations = face_recognition.face_locations(np.array(bg_removed_image_rgb))

    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]

        
        face_image = bg_removed_image.crop((left, top, right, bottom))

        face_array = np.array(face_image)

        rows, cols = face_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * 0.1)

        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)

        warped_face_array = warp(face_array, tps, output_shape=(rows, cols))

        warped_face_image = Image.fromarray((warped_face_array * 255).astype(np.uint8))

        if pad_to_original:
            padded_image = Image.new('RGBA', bg_removed_image.size)

            padded_image.paste(warped_face_image, (left, top))

            return to_tensor(padded_image)
        else:
            return to_tensor(warped_face_image)
    else:
        return None