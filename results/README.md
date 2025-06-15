# Results Directory

This directory contains all archived training logs, model weights, evaluation outputs, and visualizations.

## Usage

- Training and evaluation scripts should output logs, weights, and mIoU results here.
- Example: `logs_deeplab/` stores training logs and weights; `mious_deeplab/` stores evaluation results and visualizations.

## Script Relationships

- If you change the output directory structure, update the save/load paths in `train.py`, `get_miou.py`, and any scripts that read results.
