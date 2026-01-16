"""Unit tests for ModelManager class."""

from typing import Final

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.cnn_visualizer.config import MODEL_CONFIGS
from src.cnn_visualizer.models import ModelManager


class TestModelManagerInitialization:
    """Tests for ModelManager initialization."""

    VALID_MODELS: Final = tuple(MODEL_CONFIGS.keys())

    def test_initialization_with_default_model(self) -> None:
        """Verify ModelManager initializes with default ResNet18."""
        manager = ModelManager()
        assert manager.model_name == "ResNet18"
        assert manager.model is not None

    @pytest.mark.parametrize("model_name", VALID_MODELS)
    def test_initialization_with_each_valid_model(self, model_name: str) -> None:
        """Verify ModelManager initializes with each supported model."""
        manager = ModelManager(model_name)
        assert manager.model_name == model_name
        assert manager.config == MODEL_CONFIGS[model_name]

    def test_initialization_with_invalid_model_raises_key_error(self) -> None:
        """Verify ModelManager raises KeyError for unknown models."""
        with pytest.raises(KeyError, match="Unknown model"):
            ModelManager("NonExistentModel")

    def test_model_is_in_eval_mode(self) -> None:
        """Verify loaded model is set to evaluation mode."""
        manager = ModelManager("ResNet18")
        assert not manager.model.training


class TestModelManagerLayerAccess:
    """Tests for layer access methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager instance for testing."""
        return ModelManager("ResNet18")

    def test_get_layer_names_returns_configured_layers(self, manager: ModelManager) -> None:
        """Verify get_layer_names returns layers from config."""
        layer_names = manager.get_layer_names()
        assert layer_names == list(MODEL_CONFIGS["ResNet18"].layers)

    def test_get_layer_returns_nn_module(self, manager: ModelManager) -> None:
        """Verify get_layer returns valid neural network module."""
        layer = manager.get_layer("conv1")
        assert isinstance(layer, nn.Module)

    def test_get_layer_with_invalid_name_raises_key_error(self, manager: ModelManager) -> None:
        """Verify get_layer raises KeyError for invalid layer names."""
        with pytest.raises(KeyError, match="Unknown layer"):
            manager.get_layer("nonexistent_layer")

    def test_get_gradcam_layer_returns_module(self, manager: ModelManager) -> None:
        """Verify get_gradcam_layer returns target layer for Grad-CAM."""
        layer = manager.get_gradcam_layer()
        assert isinstance(layer, nn.Module)

    def test_get_max_filters_returns_positive_int(self, manager: ModelManager) -> None:
        """Verify get_max_filters returns expected value."""
        max_filters = manager.get_max_filters()
        assert max_filters == MODEL_CONFIGS["ResNet18"].max_filters


class TestModelManagerDotNotationLayers:
    """Tests for dot-notation layer access (e.g., 'features.0')."""

    def test_mobilenet_dot_notation_layers(self) -> None:
        """Verify dot-notation layer access works for MobileNetV2."""
        manager = ModelManager("MobileNetV2")
        layer = manager.get_layer("features.0")
        assert isinstance(layer, nn.Module)

    def test_efficientnet_nested_layers(self) -> None:
        """Verify nested layer access works for EfficientNet."""
        manager = ModelManager("EfficientNet-B0")
        layer = manager.get_layer("features.2")
        assert isinstance(layer, nn.Module)


class TestModelManagerInference:
    """Tests for model inference methods."""

    @pytest.fixture
    def manager(self) -> ModelManager:
        """Create a ModelManager instance for testing."""
        return ModelManager("ResNet18")

    def test_forward_returns_tensor(self, manager: ModelManager) -> None:
        """Verify forward pass returns output tensor."""
        input_tensor = torch.randn(1, 3, 224, 224)
        output = manager.forward(input_tensor)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1000)

    def test_predict_returns_top_k_predictions(
        self, manager: ModelManager, sample_rgb_image: Image.Image
    ) -> None:
        """Verify predict returns correct number of predictions."""
        predictions = manager.predict(sample_rgb_image, top_k=5)
        assert len(predictions) == 5
        assert all(isinstance(pred, tuple) for pred in predictions)

    def test_predict_returns_class_name_and_probability(
        self, manager: ModelManager, sample_rgb_image: Image.Image
    ) -> None:
        """Verify prediction format is (class_name, probability)."""
        predictions = manager.predict(sample_rgb_image, top_k=1)
        class_name, probability = predictions[0]
        assert isinstance(class_name, str)
        assert isinstance(probability, float)
        assert 0 <= probability <= 1

    def test_predict_probabilities_are_sorted_descending(
        self, manager: ModelManager, sample_rgb_image: Image.Image
    ) -> None:
        """Verify predictions are sorted by probability descending."""
        predictions = manager.predict(sample_rgb_image, top_k=5)
        probabilities = [prob for _, prob in predictions]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_predict_with_custom_top_k(
        self, manager: ModelManager, sample_rgb_image: Image.Image
    ) -> None:
        """Verify custom top_k parameter is respected."""
        predictions = manager.predict(sample_rgb_image, top_k=3)
        assert len(predictions) == 3


class TestModelManagerStaticMethods:
    """Tests for static methods."""

    def test_get_available_models_returns_all_configured(self) -> None:
        """Verify get_available_models returns all MODEL_CONFIGS keys."""
        available = ModelManager.get_available_models()
        assert set(available) == set(MODEL_CONFIGS.keys())

    def test_get_available_models_returns_list(self) -> None:
        """Verify get_available_models returns a list."""
        available = ModelManager.get_available_models()
        assert isinstance(available, list)
