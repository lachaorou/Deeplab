# Utils Directory

This directory contains general-purpose utility modules, such as data loaders, color mapping, metrics, and training helpers.

## Usage
- These modules are imported by main scripts in the Scripts and Models directories.
- Example: `dataloader0.py` is used by `train.py` for loading datasets.
- Example: `colorize.py` is used for label and prediction visualization in both training and inspection scripts.

## Script Relationships
- If you modify utility functions here, all scripts that import them (e.g., `train.py`, `get_miou.py`, `visualize_label_pred.py`) may be affected.
- Always check for import path consistency after moving or renaming utility files.
