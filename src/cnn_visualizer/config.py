"""CNN Visualizer - Configuration and constants."""

from torchvision import transforms

# Image preprocessing settings
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Available layers for visualization
AVAILABLE_LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4"]
DEFAULT_LAYER = "layer2"

# Visualization defaults
DEFAULT_NUM_FILTERS = 16
MIN_FILTERS = 4
MAX_FILTERS = 64
FILTER_STEP = 4

# Image transforms pipeline
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


# Pre-loaded labels (loaded once at import time)
IMAGENET_LABELS = load_imagenet_labels()
