"""Unit tests for FeatureMapExtractor class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.cnn_visualizer.visualization.feature_maps import FeatureMapExtractor


class TestFeatureMapExtractorInitialization:
    """Tests for FeatureMapExtractor initialization."""

    def test_initialization_stores_model_manager(self, mock_model_manager: MagicMock) -> None:
        """Verify extractor stores the provided model manager."""
        extractor = FeatureMapExtractor(mock_model_manager)
        assert extractor.model_manager is mock_model_manager

    def test_update_model_replaces_manager(self, mock_model_manager: MagicMock) -> None:
        """Verify update_model replaces the model manager."""
        extractor = FeatureMapExtractor(mock_model_manager)
        new_manager = MagicMock()
        extractor.update_model(new_manager)
        assert extractor.model_manager is new_manager


class TestFeatureMapExtraction:
    """Tests for feature map extraction functionality."""

    @pytest.fixture
    def extractor(self, mock_model_manager: MagicMock) -> FeatureMapExtractor:
        """Create a FeatureMapExtractor with mock manager."""
        return FeatureMapExtractor(mock_model_manager)

    def test_extract_returns_numpy_array(
        self, extractor: FeatureMapExtractor, sample_rgb_image: Image.Image
    ) -> None:
        """Verify extract returns numpy array."""
        result = extractor.extract(sample_rgb_image, "layer1")
        assert isinstance(result, np.ndarray)

    def test_extract_returns_3d_array(
        self, extractor: FeatureMapExtractor, sample_rgb_image: Image.Image
    ) -> None:
        """Verify extract returns array with shape (filters, height, width)."""
        result = extractor.extract(sample_rgb_image, "layer1")
        assert result.ndim == 3

    def test_extract_calls_forward_hook(
        self, mock_model_manager: MagicMock, sample_rgb_image: Image.Image
    ) -> None:
        """Verify extract registers and removes forward hook."""
        extractor = FeatureMapExtractor(mock_model_manager)
        extractor.extract(sample_rgb_image, "layer1")
        mock_model_manager.get_layer.assert_called_with("layer1")
        mock_model_manager.forward.assert_called_once()

    def test_extract_returns_empty_array_when_no_activation(
        self, sample_rgb_image: Image.Image
    ) -> None:
        """Verify extract handles case with no activations."""
        manager = MagicMock()
        manager.get_layer.return_value = MagicMock()
        manager.get_layer.return_value.register_forward_hook.return_value = MagicMock()
        manager.forward.side_effect = lambda x: None

        extractor = FeatureMapExtractor(manager)

        with patch.object(
            manager.get_layer.return_value,
            "register_forward_hook",
            return_value=MagicMock(),
        ):
            result = extractor.extract(sample_rgb_image, "layer1")
            assert isinstance(result, np.ndarray)


class TestFeatureMapVisualization:
    """Tests for feature map visualization functionality."""

    @pytest.fixture
    def extractor(self, mock_model_manager: MagicMock) -> FeatureMapExtractor:
        """Create a FeatureMapExtractor with mock manager."""
        return FeatureMapExtractor(mock_model_manager)

    def test_visualize_returns_pil_image(
        self, extractor: FeatureMapExtractor, sample_feature_maps: np.ndarray
    ) -> None:
        """Verify visualize returns PIL Image."""
        result = extractor.visualize(sample_feature_maps, num_maps=16)
        assert isinstance(result, Image.Image)

    def test_visualize_respects_num_maps_parameter(self, extractor: FeatureMapExtractor) -> None:
        """Verify visualize limits displayed maps to num_maps."""
        feature_maps = np.random.rand(100, 28, 28).astype(np.float32)
        result = extractor.visualize(feature_maps, num_maps=4)
        assert isinstance(result, Image.Image)

    def test_visualize_with_fewer_maps_than_requested(self, extractor: FeatureMapExtractor) -> None:
        """Verify visualize handles case with fewer maps than requested."""
        feature_maps = np.random.rand(4, 28, 28).astype(np.float32)
        result = extractor.visualize(feature_maps, num_maps=16)
        assert isinstance(result, Image.Image)

    def test_visualize_produces_non_empty_image(
        self, extractor: FeatureMapExtractor, sample_feature_maps: np.ndarray
    ) -> None:
        """Verify visualize produces an image with content."""
        result = extractor.visualize(sample_feature_maps, num_maps=16)
        assert result.size[0] > 0
        assert result.size[1] > 0

    @pytest.mark.parametrize("num_maps", [1, 4, 9, 16, 25])
    def test_visualize_with_various_grid_sizes(
        self, extractor: FeatureMapExtractor, num_maps: int
    ) -> None:
        """Verify visualize works with different grid sizes."""
        feature_maps = np.random.rand(64, 28, 28).astype(np.float32)
        result = extractor.visualize(feature_maps, num_maps=num_maps)
        assert isinstance(result, Image.Image)


class TestFeatureMapExtractorIntegration:
    """Integration tests with real ModelManager."""

    @pytest.fixture
    def real_extractor(self) -> FeatureMapExtractor:
        """Create extractor with real ModelManager for integration tests."""
        from src.cnn_visualizer.models import ModelManager

        manager = ModelManager("ResNet18")
        return FeatureMapExtractor(manager)

    def test_extract_from_real_model(
        self, real_extractor: FeatureMapExtractor, sample_rgb_image: Image.Image
    ) -> None:
        """Verify extraction works with real ResNet18 model."""
        result = real_extractor.extract(sample_rgb_image, "conv1")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] == 64

    def test_full_pipeline_extract_and_visualize(
        self, real_extractor: FeatureMapExtractor, sample_rgb_image: Image.Image
    ) -> None:
        """Verify full extraction and visualization pipeline."""
        feature_maps = real_extractor.extract(sample_rgb_image, "layer1")
        visualization = real_extractor.visualize(feature_maps, num_maps=16)
        assert isinstance(visualization, Image.Image)
        assert visualization.size[0] > 100
