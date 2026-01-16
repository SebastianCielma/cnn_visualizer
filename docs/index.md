# CNN Visualizer

Interactive tool for visualizing convolutional neural network activations.

## Features

- **Feature Map Extraction** - Visualize activations from any convolutional layer
- **Grad-CAM** - Generate class activation maps for model interpretability
- **Layer Animation** - Animated GIFs showing feature maps across all layers
- **Multiple Models** - Support for ResNet18, ResNet50, MobileNetV2, EfficientNet-B0

## Quick Start

```bash
# Install with uv
uv sync

# Run the application
uv run python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

## Documentation

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [API Reference](api/config.md)
