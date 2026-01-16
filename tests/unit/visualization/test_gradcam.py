"""Unit tests for GradCAMVisualizer class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.cnn_visualizer.config import IMAGE_SIZE
from src.cnn_visualizer.visualization.gradcam import GradCAMVisualizer


class TestGradCAMVisualizerInitialization:
    """Tests for GradCAMVisualizer initialization."""

    def test_initialization_stores_model_manager(self, mock_model_manager: MagicMock) -> None:
        """Verify visualizer stores the provided model manager."""
        with patch("src.cnn_visualizer.visualization.gradcam.GradCAM"):
            visualizer = GradCAMVisualizer(mock_model_manager)
            assert visualizer.model_manager is mock_model_manager

    def test_initialization_creates_gradcam_instance(self, mock_model_manager: MagicMock) -> None:
        """Verify GradCAM instance is created during initialization."""
        with patch("src.cnn_visualizer.visualization.gradcam.GradCAM") as mock_gradcam:
            GradCAMVisualizer(mock_model_manager)
            mock_gradcam.assert_called_once()

    def test_update_model_replaces_manager_and_rebuilds_cam(
        self, mock_model_manager: MagicMock
    ) -> None:
        """Verify update_model replaces manager and rebuilds CAM."""
        with patch("src.cnn_visualizer.visualization.gradcam.GradCAM") as mock_gradcam:
            visualizer = GradCAMVisualizer(mock_model_manager)
            new_manager = MagicMock()
            new_manager.model = MagicMock()
            new_manager.get_gradcam_layer.return_value = MagicMock()

            visualizer.update_model(new_manager)

            assert visualizer.model_manager is new_manager
            assert mock_gradcam.call_count == 2


class TestGradCAMGeneration:
    """Tests for Grad-CAM generation functionality."""

    @pytest.fixture
    def visualizer(self, mock_model_manager: MagicMock) -> GradCAMVisualizer:
        """Create a GradCAMVisualizer with mock dependencies."""
        with patch("src.cnn_visualizer.visualization.gradcam.GradCAM") as mock_gradcam:
            mock_cam_instance = MagicMock()
            mock_cam_instance.return_value = np.random.rand(1, 224, 224).astype(np.float32)
            mock_gradcam.return_value = mock_cam_instance
            return GradCAMVisualizer(mock_model_manager)

    def test_generate_returns_pil_image(
        self, visualizer: GradCAMVisualizer, sample_rgb_image: Image.Image
    ) -> None:
        """Verify generate returns PIL Image."""
        with patch("src.cnn_visualizer.visualization.gradcam.show_cam_on_image") as mock_show:
            mock_show.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
            result = visualizer.generate(sample_rgb_image)
            assert isinstance(result, Image.Image)

    def test_generate_calls_gradcam_with_input_tensor(
        self, mock_model_manager: MagicMock, sample_rgb_image: Image.Image
    ) -> None:
        """Verify generate calls GradCAM with proper input."""
        with patch("src.cnn_visualizer.visualization.gradcam.GradCAM") as mock_gradcam:
            mock_cam_instance = MagicMock()
            mock_cam_instance.return_value = np.random.rand(1, 224, 224).astype(np.float32)
            mock_gradcam.return_value = mock_cam_instance

            with patch("src.cnn_visualizer.visualization.gradcam.show_cam_on_image") as mock_show:
                mock_show.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
                visualizer = GradCAMVisualizer(mock_model_manager)
                visualizer.generate(sample_rgb_image)
                mock_cam_instance.assert_called_once()


class TestGradCAMIntegration:
    """Integration tests with real ModelManager."""

    @pytest.fixture
    def real_visualizer(self) -> GradCAMVisualizer:
        """Create visualizer with real ModelManager."""
        from src.cnn_visualizer.models import ModelManager

        manager = ModelManager("ResNet18")
        return GradCAMVisualizer(manager)

    def test_generate_with_real_model(
        self, real_visualizer: GradCAMVisualizer, sample_rgb_image: Image.Image
    ) -> None:
        """Verify Grad-CAM generation works with real model."""
        result = real_visualizer.generate(sample_rgb_image)
        assert isinstance(result, Image.Image)

    def test_generate_output_has_correct_dimensions(
        self, real_visualizer: GradCAMVisualizer, sample_rgb_image: Image.Image
    ) -> None:
        """Verify output image has expected dimensions."""
        result = real_visualizer.generate(sample_rgb_image)
        assert result.size == IMAGE_SIZE

    def test_generate_output_is_rgb(
        self, real_visualizer: GradCAMVisualizer, sample_rgb_image: Image.Image
    ) -> None:
        """Verify output image is RGB mode."""
        result = real_visualizer.generate(sample_rgb_image)
        assert result.mode == "RGB"

    def test_generate_with_grayscale_input(
        self, real_visualizer: GradCAMVisualizer, sample_grayscale_image: Image.Image
    ) -> None:
        """Verify Grad-CAM works with grayscale input converted to RGB."""
        rgb_image = sample_grayscale_image.convert("RGB")
        result = real_visualizer.generate(rgb_image)
        assert isinstance(result, Image.Image)

    @pytest.mark.parametrize("model_name", ["ResNet18", "ResNet50", "MobileNetV2"])
    def test_generate_with_different_models(
        self, model_name: str, sample_rgb_image: Image.Image
    ) -> None:
        """Verify Grad-CAM works with different model architectures."""
        from src.cnn_visualizer.models import ModelManager

        manager = ModelManager(model_name)
        visualizer = GradCAMVisualizer(manager)
        result = visualizer.generate(sample_rgb_image)
        assert isinstance(result, Image.Image)
