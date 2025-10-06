import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .data_preparation import DeepfakeDataset

class DeepfakeDetectorInference:
    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = torch.load(model_path, map_location=self.device) if isinstance(model_path, str) else model_path
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = DeepfakeDataset.get_transforms(augment=False)
    
    def extract_forensic_features(self, image: np.ndarray) -> torch.Tensor:
        features = []
        
        # Color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        features.extend(hist_h / (hist_h.sum() + 1e-6))
        features.extend(hist_s / (hist_s.sum() + 1e-6))
        features.extend(hist_v / (hist_v.sum() + 1e-6))
        
        # Noise features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(noise / 1000.0)
        
        return torch.FloatTensor(features)
    
    def predict(self, image_path: str) -> dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            transformed = self.transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Extract forensic features
            forensic_features = self.extract_forensic_features