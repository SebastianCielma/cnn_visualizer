"""ResNet model management."""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from ..config import IMAGENET_LABELS, MODEL_CONFIGS, ModelConfig, image_transform


class ModelManager:
    """Manages CNN models for inference and layer access."""

    def __init__(self, model_name: str = "ResNet18") -> None:
        """Initialize with specified model.

        Args:
            model_name: Name of the model to load.

        Raises:
            KeyError: If model_name is not supported.
        """
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model: {model_name}. Available: {self.get_available_models()}")

        self.model_name = model_name
        self.config: ModelConfig = MODEL_CONFIGS[model_name]
        self.model = self._load_model()
        self.model.eval()
        self._layer_mapping = self._build_layer_mapping()

    def _load_model(self) -> nn.Module:
        """Load the pretrained model based on configuration."""
        model_loaders = {
            "ResNet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
            "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            "MobileNetV2": lambda: models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            ),
            "EfficientNet-B0": lambda: models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            ),
        }
        return model_loaders[self.model_name]()

    def _build_layer_mapping(self) -> dict[str, nn.Module]:
        """Build mapping from layer names to layer modules."""
        mapping = {}
        for layer_name in self.config.layers:
            mapping[layer_name] = self._get_layer_by_name(layer_name)
        return mapping

    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Get layer module by dot-notation name (e.g., 'features.0')."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def get_layer(self, layer_name: str) -> nn.Module:
        """Get a specific layer by name.

        Args:
            layer_name: Name of the layer.

        Returns:
            The requested layer module.

        Raises:
            KeyError: If layer_name is not valid.
        """
        if layer_name not in self._layer_mapping:
            raise KeyError(f"Unknown layer: {layer_name}. Available: {self.get_layer_names()}")
        return self._layer_mapping[layer_name]

    def get_gradcam_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM visualization."""
        layer = self._get_layer_by_name(self.config.gradcam_layer)
        if hasattr(layer, "__getitem__"):
            return layer[-1]
        return layer

    def get_layer_names(self) -> list[str]:
        """Get list of available layer names for this model."""
        return list(self.config.layers)

    def get_max_filters(self) -> int:
        """Get maximum number of filters for this model."""
        return self.config.max_filters

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of all available model names."""
        return list(MODEL_CONFIGS.keys())

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
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True)
        ]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass on tensor input."""
        return self.model(input_tensor)
