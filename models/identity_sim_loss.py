import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from facenet_pytorch import InceptionResnetV1

class IdentitySimilarityLoss(nn.Module):
    def __init__(self, device):
        super(IdentitySimilarityLoss, self).__init__()
        self.device = device
        self.face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def forward(self, Gbase, I3, I4):
        full_transfer_output = Gbase(I3, I4)

        full_transfer = full_transfer_output[0] if isinstance(full_transfer_output, tuple) else full_transfer_output

        def prepare_input(x):
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected a tensor, got {type(x)}")
            if x.dim() == 3:
                x = x.unsqueeze(0)  
            if x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2)  
            return x.float().to(self.device)  

        I3 = prepare_input(I3)
        full_transfer = prepare_input(full_transfer)

        with torch.no_grad():
            try:
                id_features_source = self.face_recognition_model(I3)
                id_features_transfer = self.face_recognition_model(full_transfer)
            except RuntimeError as e:
                print(f"Error in face recognition model: {e}")
                print(f"I3 shape: {I3.shape}, full_transfer shape: {full_transfer.shape}")
                raise

        similarity = F.cosine_similarity(id_features_source, id_features_transfer)

        loss = -similarity.mean()

        return loss
