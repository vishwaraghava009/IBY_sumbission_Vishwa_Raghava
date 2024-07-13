import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms




def get_foreground_mask(image):

    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image, torch.Tensor):
        if image.dim() == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        input_tensor = transform(image).unsqueeze(0)
    else:
        input_tensor = transforms.ToTensor()(image)
        input_tensor = transform(input_tensor).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    _, mask = torch.max(output['out'], dim=1)

    foreground_mask = (mask == 15).float()  

    return foreground_mask.to(device)
