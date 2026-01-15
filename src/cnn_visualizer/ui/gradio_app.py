"""Gradio interface for CNN Visualizer."""

import gradio as gr
from PIL import Image

from ..config import (
    AVAILABLE_LAYERS,
    DEFAULT_LAYER,
    DEFAULT_NUM_FILTERS,
    MIN_FILTERS,
    MAX_FILTERS,
    FILTER_STEP,
    IMAGE_SIZE,
)
from ..models import ModelManager
from ..visualization import FeatureMapExtractor, GradCAMVisualizer


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface.
    
    Returns:
        Configured Gradio Blocks interface.
    """
    model_manager = ModelManager()
    feature_extractor = FeatureMapExtractor(model_manager)
    gradcam_visualizer = GradCAMVisualizer(model_manager)
    
    def process_image(
        image: Image.Image | None, 
        layer: str, 
        num_filters: int
    ) -> tuple[Image.Image | None, Image.Image | None, str]:
        """Process image and generate all visualizations.
        
        Args:
            image: Input PIL Image or None.
            layer: Layer name for feature extraction.
            num_filters: Number of feature maps to display.
            
        Returns:
            Tuple of (feature_map_image, gradcam_image, predictions_text).
        """
        if image is None:
            return None, None, "Upload an image"
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        feature_maps = feature_extractor.extract(image, layer)
        feature_map_img = feature_extractor.visualize(feature_maps, int(num_filters))
        
        gradcam_img = gradcam_visualizer.generate(image)
        
        predictions = model_manager.predict(image)
        predictions_text = "\n".join([
            f"{name}: {prob:.2%}" for name, prob in predictions
        ])
        
        return feature_map_img, gradcam_img, predictions_text
    
    with gr.Blocks(title="CNN Visualizer") as demo:
        gr.Markdown("# CNN Feature Map Visualizer")
        gr.Markdown(f"*Processed image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px*")
        
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image")
            layer_select = gr.Dropdown(
                choices=AVAILABLE_LAYERS,
                value=DEFAULT_LAYER,
                label="Layer"
            )
            num_filters = gr.Slider(
                minimum=MIN_FILTERS, 
                maximum=MAX_FILTERS, 
                value=DEFAULT_NUM_FILTERS, 
                step=FILTER_STEP, 
                label="Number of filters"
            )
            analyze_btn = gr.Button("Analyze")
        
        with gr.Row():
            feature_maps_output = gr.Image(label="Feature Maps")
            gradcam_output = gr.Image(label="Grad-CAM")
        
        predictions_output = gr.Textbox(label="Top 5 Predictions", lines=5)
        
        analyze_btn.click(
            fn=process_image,
            inputs=[input_image, layer_select, num_filters],
            outputs=[feature_maps_output, gradcam_output, predictions_output]
        )
    
    return demo
