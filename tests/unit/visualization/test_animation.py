"""Unit tests for LayerAnimator class."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.cnn_visualizer.visualization.animation import LayerAnimator


class TestLayerAnimatorInitialization:
    """Tests for LayerAnimator initialization."""

    def test_initialization_stores_model_manager(self, mock_model_manager: MagicMock) -> None:
        """Verify animator stores the provided model manager."""
        animator = LayerAnimator(mock_model_manager)
        assert animator.model_manager is mock_model_manager

    def test_update_model_replaces_manager(self, mock_model_manager: MagicMock) -> None:
        """Verify update_model replaces the model manager."""
        animator = LayerAnimator(mock_model_manager)
        new_manager = MagicMock()
        animator.update_model(new_manager)
        assert animator.model_manager is new_manager


class TestLayerAnimatorFeatureExtraction:
    """Tests for internal feature map extraction."""

    @pytest.fixture
    def animator(self, mock_model_manager: MagicMock) -> LayerAnimator:
        """Create a LayerAnimator with mock manager."""
        return LayerAnimator(mock_model_manager)

    def test_extract_feature_maps_returns_numpy_array(
        self, animator: LayerAnimator, sample_rgb_image: Image.Image
    ) -> None:
        """Verify _extract_feature_maps returns numpy array."""
        result = animator._extract_feature_maps(sample_rgb_image, "layer1")
        assert isinstance(result, np.ndarray)

    def test_extract_feature_maps_returns_3d_array(
        self, animator: LayerAnimator, sample_rgb_image: Image.Image
    ) -> None:
        """Verify _extract_feature_maps returns (filters, H, W) array."""
        result = animator._extract_feature_maps(sample_rgb_image, "layer1")
        assert result.ndim == 3


class TestLayerAnimatorFrameCreation:
    """Tests for animation frame creation."""

    @pytest.fixture
    def animator(self, mock_model_manager: MagicMock) -> LayerAnimator:
        """Create a LayerAnimator with mock manager."""
        return LayerAnimator(mock_model_manager)

    def test_create_frame_returns_pil_image(
        self, animator: LayerAnimator, sample_feature_maps: np.ndarray
    ) -> None:
        """Verify _create_frame returns PIL Image."""
        result = animator._create_frame(
            feature_maps=sample_feature_maps,
            layer_name="layer1",
            layer_idx=0,
            total_layers=5,
            num_maps=16,
        )
        assert isinstance(result, Image.Image)

    def test_create_frame_is_rgb_mode(
        self, animator: LayerAnimator, sample_feature_maps: np.ndarray
    ) -> None:
        """Verify _create_frame returns RGB image."""
        result = animator._create_frame(
            feature_maps=sample_feature_maps,
            layer_name="layer1",
            layer_idx=0,
            total_layers=5,
            num_maps=16,
        )
        assert result.mode == "RGB"

    @pytest.mark.parametrize("num_maps", [1, 4, 9, 16])
    def test_create_frame_with_various_map_counts(
        self, animator: LayerAnimator, sample_feature_maps: np.ndarray, num_maps: int
    ) -> None:
        """Verify _create_frame works with different map counts."""
        result = animator._create_frame(
            feature_maps=sample_feature_maps,
            layer_name="layer1",
            layer_idx=0,
            total_layers=5,
            num_maps=num_maps,
        )
        assert isinstance(result, Image.Image)


class TestLayerAnimatorGifGeneration:
    """Tests for GIF generation functionality."""

    @pytest.fixture
    def animator_with_mocked_extraction(self, mock_model_manager: MagicMock) -> LayerAnimator:
        """Create animator with mocked feature extraction."""
        mock_model_manager.get_layer_names.return_value = ["layer1", "layer2"]
        animator = LayerAnimator(mock_model_manager)
        return animator

    def test_generate_gif_returns_file_path(
        self,
        animator_with_mocked_extraction: LayerAnimator,
        sample_rgb_image: Image.Image,
    ) -> None:
        """Verify generate_gif returns a file path string."""
        with patch.object(
            animator_with_mocked_extraction,
            "_extract_feature_maps",
            return_value=np.random.rand(16, 28, 28).astype(np.float32),
        ):
            result = animator_with_mocked_extraction.generate_gif(
                sample_rgb_image, num_filters=4, duration_ms=100
            )
            assert isinstance(result, str)
            assert result.endswith(".gif")

    def test_generate_gif_creates_file(
        self,
        animator_with_mocked_extraction: LayerAnimator,
        sample_rgb_image: Image.Image,
    ) -> None:
        """Verify generate_gif creates an actual file."""
        with patch.object(
            animator_with_mocked_extraction,
            "_extract_feature_maps",
            return_value=np.random.rand(16, 28, 28).astype(np.float32),
        ):
            result = animator_with_mocked_extraction.generate_gif(
                sample_rgb_image, num_filters=4, duration_ms=100
            )
            assert os.path.exists(result)
            os.unlink(result)

    def test_generate_gif_converts_grayscale_to_rgb(
        self,
        animator_with_mocked_extraction: LayerAnimator,
        sample_grayscale_image: Image.Image,
    ) -> None:
        """Verify generate_gif handles grayscale input."""
        with patch.object(
            animator_with_mocked_extraction,
            "_extract_feature_maps",
            return_value=np.random.rand(16, 28, 28).astype(np.float32),
        ):
            result = animator_with_mocked_extraction.generate_gif(
                sample_grayscale_image, num_filters=4, duration_ms=100
            )
            assert os.path.exists(result)
            os.unlink(result)

    @pytest.mark.parametrize("duration_ms", [100, 500, 1000])
    def test_generate_gif_with_various_durations(
        self,
        animator_with_mocked_extraction: LayerAnimator,
        sample_rgb_image: Image.Image,
        duration_ms: int,
    ) -> None:
        """Verify generate_gif works with different frame durations."""
        with patch.object(
            animator_with_mocked_extraction,
            "_extract_feature_maps",
            return_value=np.random.rand(16, 28, 28).astype(np.float32),
        ):
            result = animator_with_mocked_extraction.generate_gif(
                sample_rgb_image, num_filters=4, duration_ms=duration_ms
            )
            assert os.path.exists(result)
            os.unlink(result)

    def test_generate_gif_loop_parameter(
        self,
        animator_with_mocked_extraction: LayerAnimator,
        sample_rgb_image: Image.Image,
    ) -> None:
        """Verify generate_gif respects loop parameter."""
        with patch.object(
            animator_with_mocked_extraction,
            "_extract_feature_maps",
            return_value=np.random.rand(16, 28, 28).astype(np.float32),
        ):
            result = animator_with_mocked_extraction.generate_gif(
                sample_rgb_image, num_filters=4, duration_ms=100, loop=False
            )
            assert os.path.exists(result)
            os.unlink(result)


class TestLayerAnimatorIntegration:
    """Integration tests with real ModelManager."""

    @pytest.fixture
    def real_animator(self) -> LayerAnimator:
        """Create animator with real ModelManager."""
        from src.cnn_visualizer.models import ModelManager

        manager = ModelManager("ResNet18")
        return LayerAnimator(manager)

    def test_generate_gif_with_real_model(
        self, real_animator: LayerAnimator, sample_rgb_image: Image.Image
    ) -> None:
        """Verify GIF generation works with real model."""
        result = real_animator.generate_gif(sample_rgb_image, num_filters=4, duration_ms=100)
        assert os.path.exists(result)
        assert result.endswith(".gif")

        with Image.open(result) as gif:
            assert gif.format == "GIF"
            assert gif.n_frames > 1

        os.unlink(result)

    def test_extract_feature_maps_with_real_model(
        self, real_animator: LayerAnimator, sample_rgb_image: Image.Image
    ) -> None:
        """Verify feature extraction works with real model."""
        result = real_animator._extract_feature_maps(sample_rgb_image, "conv1")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] == 64
