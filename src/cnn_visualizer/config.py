"""CNN Visualizer - Configuration and constants."""

from dataclasses import dataclass
from typing import Final

from torchvision import transforms


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a CNN model."""
    
    layers: tuple[str, ...]
    max_filters: int
    gradcam_layer: str
    weights_enum: str


MODEL_CONFIGS: Final[dict[str, ModelConfig]] = {
    "ResNet18": ModelConfig(
        layers=("conv1", "layer1", "layer2", "layer3", "layer4"),
        max_filters=512,
        gradcam_layer="layer4",
        weights_enum="ResNet18_Weights.IMAGENET1K_V1",
    ),
    "ResNet50": ModelConfig(
        layers=("conv1", "layer1", "layer2", "layer3", "layer4"),
        max_filters=2048,
        gradcam_layer="layer4",
        weights_enum="ResNet50_Weights.IMAGENET1K_V1",
    ),
    "MobileNetV2": ModelConfig(
        layers=("features.0", "features.3", "features.6", "features.13", "features.17"),
        max_filters=1280,
        gradcam_layer="features.17",
        weights_enum="MobileNet_V2_Weights.IMAGENET1K_V1",
    ),
}

AVAILABLE_MODELS: Final = list(MODEL_CONFIGS.keys())
DEFAULT_MODEL: Final = "ResNet18"

IMAGE_SIZE: Final = (224, 224)
NORMALIZATION_MEAN: Final = [0.485, 0.456, 0.406]
NORMALIZATION_STD: Final = [0.229, 0.224, 0.225]

IMAGENET_LABELS_URL: Final = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

DEFAULT_NUM_FILTERS: Final = 16
MIN_FILTERS: Final = 4
FILTER_STEP: Final = 4

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])


def load_imagenet_labels() -> list[str]:
    """Load ImageNet class labels from remote URL.
    
    Returns:
        List of 1000 ImageNet class names, or fallback generic names on error.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
            return [line.decode('utf-8').strip() for line in f.readlines()]
    except Exception:
        return [f"Class {i}" for i in range(1000)]


IMAGENET_LABELS: Final = load_imagenet_labels()
