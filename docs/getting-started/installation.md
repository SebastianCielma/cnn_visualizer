# Installation

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/SebastianCielma/cnn_visualizer.git
cd cnn_visualizer

# Install dependencies
uv sync

# Run the application
uv run python app.py
```

## Installation with pip

```bash
# Clone the repository
git clone https://github.com/SebastianCielma/cnn_visualizer.git
cd cnn_visualizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the application
python app.py
```

## Development Installation

For development, install with dev dependencies:

```bash
uv sync --all-groups
```

This includes:

- `pytest` - Testing framework
- `mypy` - Static type checking
- `ruff` - Linting and formatting
- `pre-commit` - Git hooks
