"""ResNet model management."""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from ..config import image_transform, IMAGENET_LABELS


class ModelManager:
    """Manages ResNet18 model for inference and layer access."""
    
    def __init__(self) -> None:
        """Initialize with pretrained ResNet18."""
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        self._layer_mapping: dict[str, nn.Module] = {
            "conv1": self.model.conv1,
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        }
    
    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a specific layer by name.
        
        Args:
            layer_name: Name of the layer (conv1, layer1-4)
            
        Returns:
            The requested layer module.
            
        Raises:
            KeyError: If layer_name is not valid.
        """
        if layer_name not in self._layer_mapping:
            raise KeyError(f"Unknown layer: {layer_name}. Available: {self.get_layer_names()}")
        return self._layer_mapping[layer_name]
    
    def get_layer_names(self) -> list[str]:
        """Get list of available layer names."""
        return list(self._layer_mapping.keys())
    
    def predict(self, image: Image.Image, top_k: int = 5) -> list[tuple[str, float]]:
        """Run inference and return top-k predictions.
        
        Args:
            image: PIL Image to classify.
            top_k: Number of top predictions to return.
            
        Returns:
            List of (class_name, probability) tuples.
        """
        input_tensor = image_transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_probs, top_indices = torch.topk(probs, top_k)
        
        return [
            (IMAGENET_LABELS[idx], prob)
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass on tensor input."""
        return self.model(input_tensor)
