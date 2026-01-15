"""CNN Visualizer - Feature map and Grad-CAM visualization tool."""

from src.cnn_visualizer.ui import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
