"""CNN Visualizer - Configuration and constants."""

from typing import Final
from torchvision import transforms

IMAGE_SIZE: Final = (224, 224)
NORMALIZATION_MEAN: Final = [0.485, 0.456, 0.406]
NORMALIZATION_STD: Final = [0.229, 0.224, 0.225]

IMAGENET_LABELS_URL: Final = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

AVAILABLE_LAYERS: Final = ["conv1", "layer1", "layer2", "layer3", "layer4"]
DEFAULT_LAYER: Final = "layer2"

DEFAULT_NUM_FILTERS: Final = 16
MIN_FILTERS: Final = 4
MAX_FILTERS: Final = 64
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
