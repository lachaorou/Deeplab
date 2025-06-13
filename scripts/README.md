# Scripts Directory

This directory contains all main entry-point scripts for training, evaluation, and prediction.

## Usage
- Run scripts here to train models, evaluate mIoU, predict on new data, or perform batch operations.
- Example: `train.py` is the main training script; `get_miou.py` is for evaluation; `predict.py` is for inference.

## Script Relationships
- These scripts depend on modules from Models, Utils, Dataset, and may call Automation or Inspect scripts for data preparation and checking.
- If you change dataset structure, model definitions, or utility functions, update the relevant paths and imports in these scripts.
