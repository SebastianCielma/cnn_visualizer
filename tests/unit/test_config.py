"""Unit tests for CNN Visualizer configuration module."""

from typing import Final

import pytest
import torch

from src.cnn_visualizer.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_NUM_FILTERS,
    FILTER_STEP,
    IMAGE_SIZE,
    MIN_FILTERS,
    MODEL_CONFIGS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    ModelConfig,
    image_transform,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_is_frozen(self) -> None:
        """Verify ModelConfig instances are immutable."""
        config = ModelConfig(
            layers=("layer1",),
            max_filters=64,
            gradcam_layer="layer1",
            weights_enum="TestWeights",
        )
        with pytest.raises(AttributeError):
            config.max_filters = 128

    def test_model_config_fields(self) -> None:
        """Verify ModelConfig contains expected fields with correct types."""
        config = ModelConfig(
            layers=("conv1", "layer1"),
            max_filters=512,
            gradcam_layer="layer1",
            weights_enum="ResNet18_Weights.IMAGENET1K_V1",
        )
        assert config.layers == ("conv1", "layer1")
        assert config.max_filters == 512
        assert config.gradcam_layer == "layer1"
        assert config.weights_enum == "ResNet18_Weights.IMAGENET1K_V1"


class TestModelConfigs:
    """Tests for MODEL_CONFIGS dictionary."""

    EXPECTED_MODELS: Final = ("ResNet18", "ResNet50", "MobileNetV2", "EfficientNet-B0")

    def test_all_expected_models_present(self) -> None:
        """Verify all expected models are configured."""
        for model_name in self.EXPECTED_MODELS:
            assert model_name in MODEL_CONFIGS

    def test_available_models_matches_config_keys(self) -> None:
        """Verify AVAILABLE_MODELS contains all MODEL_CONFIGS keys."""
        assert set(AVAILABLE_MODELS) == set(MODEL_CONFIGS.keys())

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_each_config_has_valid_structure(self, model_name: str) -> None:
        """Verify each model config has required attributes."""
        config = MODEL_CONFIGS[model_name]
        assert isinstance(config.layers, tuple)
        assert len(config.layers) > 0
        assert isinstance(config.max_filters, int)
        assert config.max_filters > 0
        assert isinstance(config.gradcam_layer, str)
        assert config.gradcam_layer in config.layers

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_gradcam_layer_is_in_layers(self, model_name: str) -> None:
        """Verify gradcam_layer exists in the model's layer list."""
        config = MODEL_CONFIGS[model_name]
        assert config.gradcam_layer in config.layers


class TestConstants:
    """Tests for module-level constants."""

    def test_default_model_is_valid(self) -> None:
        """Verify DEFAULT_MODEL exists in configurations."""
        assert DEFAULT_MODEL in MODEL_CONFIGS

    def test_image_size_is_valid_tuple(self) -> None:
        """Verify IMAGE_SIZE is a valid dimension tuple."""
        assert isinstance(IMAGE_SIZE, tuple)
        assert len(IMAGE_SIZE) == 2
        assert all(isinstance(dim, int) and dim > 0 for dim in IMAGE_SIZE)

    def test_normalization_values_are_valid(self) -> None:
        """Verify ImageNet normalization values are correct."""
        assert len(NORMALIZATION_MEAN) == 3
        assert len(NORMALIZATION_STD) == 3
        assert all(0 <= v <= 1 for v in NORMALIZATION_MEAN)
        assert all(0 < v <= 1 for v in NORMALIZATION_STD)

    def test_filter_constraints_are_valid(self) -> None:
        """Verify filter count constraints make sense."""
        assert MIN_FILTERS > 0
        assert DEFAULT_NUM_FILTERS >= MIN_FILTERS
        assert FILTER_STEP > 0
        assert DEFAULT_NUM_FILTERS % FILTER_STEP == 0


class TestImageTransform:
    """Tests for image_transform composition."""

    def test_transform_produces_correct_shape(self, sample_rgb_image) -> None:
        """Verify transform outputs tensor with expected shape."""
        result = image_transform(sample_rgb_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def test_transform_normalizes_values(self, sample_rgb_image) -> None:
        """Verify transform applies normalization."""
        result = image_transform(sample_rgb_image)
        assert result.min() < 0 or result.max() > 1

    def test_transform_handles_different_input_sizes(self) -> None:
        """Verify transform resizes arbitrary input dimensions."""
        import numpy as np
        from PIL import Image

        small_image = Image.fromarray(
            np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8),
            mode="RGB",
        )
        result = image_transform(small_image)
        assert result.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def test_transform_is_deterministic(self, sample_rgb_image) -> None:
        """Verify same input produces identical outputs."""
        result1 = image_transform(sample_rgb_image)
        result2 = image_transform(sample_rgb_image)
        assert torch.equal(result1, result2)
