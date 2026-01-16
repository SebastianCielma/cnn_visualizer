# Quick Start

## Running the Application

After [installation](installation.md), start the Gradio interface:

```bash
uv run python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

## Using the Interface

### 1. Select a Model

Choose from available pre-trained models:

- **ResNet18** - Fast, lightweight (default)
- **ResNet50** - More accurate, slower
- **MobileNetV2** - Optimized for mobile
- **EfficientNet-B0** - Balanced performance

### 2. Upload an Image

Upload any image (JPG, PNG). It will be resized to 224×224 pixels.

### 3. Feature Maps & Grad-CAM

1. Select a layer from the dropdown
2. Adjust the number of filters to display
3. Click **Analyze**

You'll see:

- **Feature Maps** - Grid of activations from the selected layer
- **Grad-CAM** - Heatmap showing important regions for classification
- **Predictions** - Top 5 class predictions with probabilities

### 4. Layer Animation

1. Go to the **Layer Animation** tab
2. Select animation speed
3. Click **Generate Animation**

This creates an animated GIF showing how feature maps evolve through all layers.
