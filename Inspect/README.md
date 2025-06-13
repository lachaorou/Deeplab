# Inspect Directory

This directory contains scripts for data/label/prediction inspection, repair, and visualization.

## Usage
- Use scripts here to check label value consistency, repair label images, visualize predictions, and compare results.
- Example: `check_and_fix_label_values.py` can fix label pixel values, which is critical for correct training and evaluation in `train.py` and `get_miou.py`.
- Example: `visualize_label_pred.py` and `visualize_compare.py` help you visually inspect model predictions and label quality.

## Script Relationships
- If you modify label images or dataset structure with these scripts, you must re-run or update your training and evaluation scripts to use the corrected data.
- Visualization scripts depend on the output of `train.py` and `get_miou.py` (e.g., prediction results and logs).
