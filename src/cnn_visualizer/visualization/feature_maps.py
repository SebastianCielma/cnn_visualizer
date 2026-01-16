"""Feature map extraction and visualization."""

import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ..config import image_transform
from ..models import ModelManager


class FeatureMapExtractor:
    """Extracts and visualizes feature maps from CNN layers."""

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

    def extract(self, image: Image.Image, layer_name: str) -> np.ndarray:
        """Extract feature maps from a specific layer.

        Args:
            image: Input PIL Image.
            layer_name: Name of layer to extract from.

        Returns:
            NumPy array of shape (num_filters, height, width).
        """
        input_tensor = image_transform(image).unsqueeze(0)
        activations: list[torch.Tensor] = []

        def hook_fn(module: object, input: object, output: torch.Tensor) -> None:
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
        return np.array([])

    def visualize(self, feature_maps: np.ndarray, num_maps: int = 16) -> Image.Image:
        """Create a grid visualization of feature maps.

        Args:
            feature_maps: Array of shape (num_filters, H, W).
            num_maps: Number of maps to display in grid.

        Returns:
            PIL Image containing the visualization grid.
        """
        n_maps = min(num_maps, feature_maps.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_maps)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), squeeze=False)

        for idx, ax in enumerate(axes.flat):
            if idx < n_maps:
                ax.imshow(feature_maps[idx], cmap="viridis")
                ax.set_title(f"#{idx}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return Image.open(buf)
