"""Grad-CAM visualization."""

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ..config import image_transform, IMAGE_SIZE
from ..models import ModelManager


class GradCAMVisualizer:
    """Generates Grad-CAM visualizations for model interpretability."""
    
    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize with a model manager.
        
        Args:
            model_manager: ModelManager instance for model access.
        """
        self.model_manager = model_manager
        self._target_layers = [model_manager.get_layer("layer4")[-1]]
        self._cam = GradCAM(model=model_manager.model, target_layers=self._target_layers)
    
    def generate(self, image: Image.Image) -> Image.Image:
        """Generate Grad-CAM overlay for an image.
        
        Args:
            image: Input PIL Image.
            
        Returns:
            PIL Image with Grad-CAM heatmap overlay.
        """
        input_tensor = image_transform(image).unsqueeze(0)
        rgb_img = np.array(image.resize(IMAGE_SIZE)) / 255.0
        
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(
            rgb_img.astype(np.float32), 
            grayscale_cam, 
            use_rgb=True
        )
        
        return Image.fromarray(visualization)
