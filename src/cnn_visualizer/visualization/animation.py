"""Layer-by-layer animation visualization."""

import io
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from ..config import image_transform, IMAGE_SIZE
from ..models import ModelManager


class LayerAnimator:
    """Creates animated visualizations of feature maps across layers."""
    
    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize with a model manager.
        
        Args:
            model_manager: ModelManager instance for layer access.
        """
        self.model_manager = model_manager
    
    def update_model(self, model_manager: ModelManager) -> None:
        """Update to use a new model manager.
        
        Args:
            model_manager: New ModelManager instance.
        """
        self.model_manager = model_manager
    
    def _extract_feature_maps(self, image: Image.Image, layer_name: str) -> np.ndarray:
        """Extract all feature maps from a layer.
        
        Args:
            image: Input PIL Image.
            layer_name: Name of the layer.
            
        Returns:
            NumPy array of shape (num_filters, height, width).
        """
        input_tensor = image_transform(image).unsqueeze(0)
        activations: list[torch.Tensor] = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        target_layer = self.model_manager.get_layer(layer_name)
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                self.model_manager.forward(input_tensor)
        finally:
            handle.remove()
        
        if activations:
            return activations[0].squeeze(0).numpy()
        
        return np.zeros((1, 7, 7))
    
    def _create_frame(
        self, 
        feature_maps: np.ndarray,
        layer_name: str, 
        layer_idx: int, 
        total_layers: int,
        num_maps: int
    ) -> Image.Image:
        """Create a single animation frame with grid of feature maps.
        
        Args:
            feature_maps: Array of shape (num_filters, H, W).
            layer_name: Name of the current layer.
            layer_idx: Index of the current layer (0-based).
            total_layers: Total number of layers.
            num_maps: Number of maps to show in grid.
            
        Returns:
            PIL Image of the frame.
        """
        n_maps = min(num_maps, feature_maps.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_maps)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        for idx, ax in enumerate(axes.flat):
            if idx < n_maps:
                ax.imshow(feature_maps[idx], cmap='viridis')
                ax.set_title(f"#{idx}", fontsize=7)
            ax.axis('off')
        
        h, w = feature_maps.shape[1], feature_maps.shape[2]
        fig.suptitle(
            f"Layer: {layer_name}  |  Size: {h}×{w}  |  Step {layer_idx + 1}/{total_layers}",
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf).convert('RGB')
    
    def generate_gif(
        self, 
        image: Image.Image, 
        num_filters: int = 16,
        duration_ms: int = 800,
        loop: bool = True
    ) -> str:
        """Generate an animated GIF showing feature maps across all layers.
        
        Args:
            image: Input PIL Image.
            num_filters: Number of feature maps to show per layer.
            duration_ms: Duration of each frame in milliseconds.
            loop: Whether the GIF should loop infinitely.
            
        Returns:
            Path to the generated GIF file.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        layer_names = self.model_manager.get_layer_names()
        frames: list[Image.Image] = []
        
        for idx, layer_name in enumerate(layer_names):
            feature_maps = self._extract_feature_maps(image, layer_name)
            frame = self._create_frame(feature_maps, layer_name, idx, len(layer_names), num_filters)
            frames.append(frame)
        
        gif_path = tempfile.mktemp(suffix='.gif')
        
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1
        )
        
        return gif_path
