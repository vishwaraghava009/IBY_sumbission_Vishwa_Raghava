import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp



class MPGazeLoss(nn.Module):
    def __init__(self, device):
        super(MPGazeLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_gaze, target_gaze, face_image):
        if face_image.dim() == 4 and face_image.shape[0] == 1:
            face_image = face_image.squeeze(0)
        if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
            raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
        
        face_image = face_image.detach().cpu().numpy()
        if face_image.shape[0] == 3:  
            face_image = face_image.transpose(1, 2, 0)
        face_image = (face_image * 255).astype(np.uint8)

        results = self.face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return torch.tensor(0.0).to(self.device)

        eye_landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
            eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

        loss = 0.0
        h, w = face_image.shape[:2]
        for left_eye, right_eye in eye_landmarks:
            left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
            right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

            left_mask = torch.zeros((1, h, w)).to(self.device)
            right_mask = torch.zeros((1, h, w)).to(self.device)
            cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
            cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

            left_gaze_loss = self.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
            right_gaze_loss = self.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
            loss += left_gaze_loss + right_gaze_loss

        return loss / len(eye_landmarks)