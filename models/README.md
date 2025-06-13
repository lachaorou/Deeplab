# Models Directory

This directory contains all network architectures and training-related modules.

## Usage
- Import model definitions from here in your training, evaluation, and prediction scripts.
- Example: `deeplabv3_plus.py` defines the DeepLabV3+ model, which is used in `train.py` and `get_miou.py`.

## Script Relationships
- If you modify model structure or file names here, update all import statements in `train.py`, `predict.py`, and other scripts that use these models.
