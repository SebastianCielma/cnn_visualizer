"""Gradio interface for CNN Visualizer."""

import gradio as gr
from PIL import Image

from ..config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_NUM_FILTERS,
    FILTER_STEP,
    IMAGE_SIZE,
    MIN_FILTERS,
    MODEL_CONFIGS,
)
from ..models import ModelManager
from ..visualization import FeatureMapExtractor, GradCAMVisualizer, LayerAnimator


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface.

    Returns:
        Configured Gradio Blocks interface.
    """
    model_manager = ModelManager(DEFAULT_MODEL)
    feature_extractor = FeatureMapExtractor(model_manager)
    gradcam_visualizer = GradCAMVisualizer(model_manager)
    layer_animator = LayerAnimator(model_manager)

    def on_model_change(model_name: str) -> tuple[gr.Dropdown, gr.Slider]:
        """Handle model selection change."""
        nonlocal model_manager, feature_extractor, gradcam_visualizer, layer_animator

        model_manager = ModelManager(model_name)
        feature_extractor.update_model(model_manager)
        gradcam_visualizer.update_model(model_manager)
        layer_animator.update_model(model_manager)

        config = MODEL_CONFIGS[model_name]
        new_layers = list(config.layers)
        new_max_filters = config.max_filters

        return (
            gr.Dropdown(choices=new_layers, value=new_layers[0]),
            gr.Slider(
                minimum=MIN_FILTERS,
                maximum=new_max_filters,
                value=min(DEFAULT_NUM_FILTERS, new_max_filters),
                step=FILTER_STEP,
            ),
        )

    def process_image(
        image: Image.Image | None, layer: str, num_filters: int
    ) -> tuple[Image.Image | None, Image.Image | None, str]:
        """Process image and generate all visualizations."""
        if image is None:
            return None, None, "Upload an image"

        if image.mode != "RGB":
            image = image.convert("RGB")

        feature_maps = feature_extractor.extract(image, layer)
        feature_map_img = feature_extractor.visualize(feature_maps, int(num_filters))

        gradcam_img = gradcam_visualizer.generate(image)

        predictions = model_manager.predict(image)
        predictions_text = "\n".join([f"{name}: {prob:.2%}" for name, prob in predictions])

        return feature_map_img, gradcam_img, predictions_text

    def generate_animation(image: Image.Image | None, num_filters: int, speed: int) -> str | None:
        """Generate layer-by-layer animation."""
        if image is None:
            return None

        duration_ms = {0: 1500, 1: 800, 2: 400}.get(speed, 800)
        gif_path = layer_animator.generate_gif(
            image, num_filters=int(num_filters), duration_ms=duration_ms
        )
        return gif_path

    default_config = MODEL_CONFIGS[DEFAULT_MODEL]

    with gr.Blocks(title="CNN Visualizer") as demo:
        gr.Markdown("# CNN Feature Map Visualizer")
        gr.Markdown(f"*Processed image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px*")

        with gr.Row():
            model_select = gr.Dropdown(choices=AVAILABLE_MODELS, value=DEFAULT_MODEL, label="Model")
            input_image = gr.Image(type="pil", label="Input Image")

        with gr.Tabs():
            with gr.TabItem("Feature Maps & Grad-CAM"):
                with gr.Row():
                    layer_select = gr.Dropdown(
                        choices=list(default_config.layers),
                        value=default_config.layers[0],
                        label="Layer",
                    )
                    num_filters = gr.Slider(
                        minimum=MIN_FILTERS,
                        maximum=default_config.max_filters,
                        value=DEFAULT_NUM_FILTERS,
                        step=FILTER_STEP,
                        label="Number of filters",
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")

                with gr.Row():
                    feature_maps_output = gr.Image(label="Feature Maps")
                    gradcam_output = gr.Image(label="Grad-CAM")

                predictions_output = gr.Textbox(label="Top 5 Predictions", lines=5)

                analyze_btn.click(
                    fn=process_image,
                    inputs=[input_image, layer_select, num_filters],
                    outputs=[feature_maps_output, gradcam_output, predictions_output],
                )

            with gr.TabItem("Layer Animation"):
                gr.Markdown(
                    "*Animated GIF showing feature maps through all layers (uses filter count from above)*"
                )

                with gr.Row():
                    speed_select = gr.Radio(
                        choices=["Slow", "Normal", "Fast"],
                        value="Normal",
                        label="Animation Speed",
                        type="index",
                    )
                    anim_btn = gr.Button("Generate Animation", variant="primary")

                animation_output = gr.Image(label="Layer Animation (GIF)")

                anim_btn.click(
                    fn=generate_animation,
                    inputs=[input_image, num_filters, speed_select],
                    outputs=[animation_output],
                )

        model_select.change(
            fn=on_model_change, inputs=[model_select], outputs=[layer_select, num_filters]
        )

    return demo
