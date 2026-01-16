"""Shared fixtures for CNN Visualizer tests."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.cnn_visualizer.config import MODEL_CONFIGS, ModelConfig


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB PIL Image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        mode="RGB",
    )


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a sample grayscale PIL Image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224), dtype=np.uint8),
        mode="L",
    )


@pytest.fixture
def sample_feature_maps() -> np.ndarray:
    """Create sample feature maps tensor."""
    return np.random.rand(64, 56, 56).astype(np.float32)


@pytest.fixture
def sample_input_tensor() -> torch.Tensor:
    """Create a sample normalized input tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def mock_conv_layer() -> nn.Conv2d:
    """Create a mock convolutional layer."""
    return nn.Conv2d(3, 64, kernel_size=3, padding=1)


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Create a mock ModelConfig for testing."""
    return ModelConfig(
        layers=("layer1", "layer2", "layer3"),
        max_filters=256,
        gradcam_layer="layer3",
        weights_enum="MockWeights",
    )


@pytest.fixture
def resnet18_config() -> ModelConfig:
    """Get the actual ResNet18 configuration."""
    return MODEL_CONFIGS["ResNet18"]


class MockModule(nn.Module):
    """Mock neural network module for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(3, 64, 3, padding=1)
        self.layer2 = nn.Conv2d(64, 128, 3, padding=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


@pytest.fixture
def mock_module() -> MockModule:
    """Create a mock neural network module."""
    return MockModule()


@pytest.fixture
def mock_model_manager(mock_module: MockModule) -> MagicMock:
    """Create a mock ModelManager with controlled behavior."""
    manager = MagicMock()
    manager.model = mock_module
    manager.model_name = "MockModel"
    manager.get_layer_names.return_value = ["layer1", "layer2", "layer3"]
    manager.get_layer.side_effect = lambda name: getattr(mock_module, name)
    manager.get_gradcam_layer.return_value = mock_module.layer3[-1]
    manager.get_max_filters.return_value = 256
    manager.forward.side_effect = lambda x: mock_module(x)
    return manager


@pytest.fixture
def patched_model_manager() -> Generator[MagicMock]:
    """Patch ModelManager to avoid loading real weights."""
    with patch("src.cnn_visualizer.models.resnet.ModelManager") as mock:
        yield mock
