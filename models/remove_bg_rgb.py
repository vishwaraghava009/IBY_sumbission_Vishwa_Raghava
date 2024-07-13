import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from rembg import remove


def remove_background_and_convert_to_rgb(image_tensor):
    image = to_pil_image(image_tensor)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    bg_removed_bytes = remove(img_byte_arr)
    bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
    
    bg_removed_image_rgb = bg_removed_image.convert("RGB")
    
    return bg_removed_image_rgb