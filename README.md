# Bias-Balanced Margin (BBM) Learning

This repository contains the implementation of Bias-Balanced Margin Learning with advanced ensemble and model averaging techniques. The project combines enhanced ensemble methods with Exponential Moving Average (EMA) and Stochastic Weight Averaging (SWA) for improved model performance and stability.

## Features

- **Traditional Training**: Standard baseline and margin loss training
- **EMA Training**: Exponential Moving Average for model stability
- **SWA Training**: Stochastic Weight Averaging for improved generalization
- **Enhanced Ensemble Methods**: 
  - Top-K model selection based on validation metrics
  - Epoch-wise model saving for ensemble creation
  - Validation score tracking for ensemble optimization
- **Continual Learning**: Ability to resume training from saved checkpoints
- **Comprehensive Evaluation**: Individual model evaluation and ensemble evaluation

## Files

- `margin_loss.py`: Standard baseline and margin loss training
- `margin_loss_ema.py`: Training with Exponential Moving Average
- `margin_loss_swa.py`: Training with Stochastic Weight Averaging
- `margin_loss_ema_grad.py`: EMA training with grad_proj ensemble features
- `margin_loss_swa_grad.py`: SWA training with grad_proj ensemble features
- `margin_loss_continue.py`: Continual training from saved models
- `balance_validation_sets.py`: Tools for dataset balancing
- `predict.py`: Prediction utilities
- `requirements.txt`: Required Python packages

## Usage

### Standard Training
```bash
python margin_loss.py --type baseline --dataset celeba --train
python margin_loss.py --type margin --dataset celeba --train
```

### EMA Training
```bash
python margin_loss_ema.py --type baseline --dataset celeba --train
python margin_loss_ema.py --type margin --dataset celeba --train
```

### SWA Training
```bash
python margin_loss_swa.py --type baseline --dataset celeba --train --swa
python margin_loss_swa.py --type margin --dataset celeba --train --swa
```

### Training with Ensemble Features
```bash
python margin_loss_ema_grad.py --type baseline --dataset celeba --train
python margin_loss_swa_grad.py --type margin --dataset celeba --train --swa
```

### Testing with Ensemble
```bash
python margin_loss_ema_grad.py --type baseline --dataset celeba --test-only
python margin_loss_swa_grad.py --type margin --dataset celeba --test-only
```

### Continual Training
```bash
python margin_loss_continue.py --type baseline --dataset celeba --train
```

### Evaluation Options
- `--train`: Perform training
- `--val-only`: Evaluate on validation set once
- `--test-only`: Evaluate on test set once
- `--clustering`: Run clustering analysis only
- `--bias`: Use bias-amplified training
- `--resume`: Resume training from checkpoint
- `--swa`: Use Stochastic Weight Averaging

### Datasets
- `celeba`: CelebA dataset
- `waterbirds`: Waterbirds dataset

## Ensemble Methodology

The enhanced ensemble approach combines the advantages of:
1. **EMA/SWA**: Improving model stability and generalization through weight averaging
2. **Top-K Selection**: Selecting the best models based on validation performance
3. **Epoch-based Saving**: Creating diverse models across training epochs

This combination allows for both improved performance and robustness in model predictions.

## Configuration

Most configuration parameters are defined in the `utils.config` module. Key parameters include:
- Learning rates (`base_lr`)
- Batch sizes (`base_batch_size`)
- Number of epochs (`base_epochs`)
- Model architecture parameters
- Paths for saving/loading models

## Results

The ensemble methods with EMA/SWA typically achieve:
- Improved stability across different random seeds
- Better generalization on bias-sensitive datasets
- Higher overall accuracy on both balanced and imbalanced data

## Dependencies

See `requirements.txt` for the complete list of dependencies.

## Citation

If you use this code in your research, please cite the relevant papers.