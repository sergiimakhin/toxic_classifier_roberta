# Toxic Comments Classification with RoBERTa

A multi-label text classification project for detecting toxic comments using RoBERTa transformer model. The project classifies comments into 6 different toxicity categories.

## Project Structure

```
TOXIC_COMMENTS_ROBERTA/
├── .gitignore
├── config.py              # Configuration parameters and hyperparameters
├── dataset.py             # Custom PyTorch dataset for toxic comments
├── loss.py                # Custom loss functions (Focal Loss, Weighted BCE)
├── model.py               # Model definition and initialization
├── train.py               # Training and evaluation functions
├── utils.py               # Utility functions for plotting and model saving
├── main.py                # Main training script
├── Data/                  # Dataset directory (ignored by git)
├── model/                 # Saved model directory
│   ├── toxic_roberta_best.pt
│   └── toxic_roberta.pt
└── metrics/               # Training metrics and visualizations
    ├── FocalLoss/
    └── WeightedBCELoss/
```

## Features

- **Multi-label Classification**: Classifies comments into 6 toxicity categories
- **RoBERTa Base Model**: Uses pre-trained RoBERTa for robust text understanding
- **Custom Loss Functions**: Implements Focal Loss and Weighted BCE Loss for handling class imbalance
- **Mixed Precision Training**: Uses automatic mixed precision for faster training
- **Gradient Clipping**: Prevents gradient explosion during training
- **Learning Rate Scheduling**: Includes warmup and weight decay for stable training
- **Comprehensive Evaluation**: Generates training curves and confusion matrices

## Key Components

### Model Configuration
- **Model**: RoBERTa-base with 6 output labels
- **Max Length**: 256 tokens
- **Batch Size**: 8 (optimized for stability)
- **Learning Rate**: 1e-5 with warmup
- **Epochs**: 3 (prevents overfitting)

### Loss Functions
- **Focal Loss**: Handles class imbalance by focusing on hard examples
- **Weighted BCE Loss**: Alternative loss function with class weighting

### Training Features
- Gradient clipping (max norm: 1.0)
- Mixed precision training with GradScaler
- Learning rate warmup (10% of training steps)
- Weight decay regularization (0.01)

## Usage

1. **Setup**: Ensure your toxic comments dataset is placed in the `Data/` directory
2. **Configuration**: Modify hyperparameters in `config.py` if needed
3. **Training**: Run the main training script:
   ```bash
   python main.py
   ```
4. **Evaluation**: Training metrics and visualizations are automatically saved to `metrics/`

## Model Output

The model outputs probabilities for 6 toxicity categories. A threshold of 0.5 is used to convert probabilities to binary predictions for each label.

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

## Model Artifacts

Trained models are saved in the `model/` directory:
- `toxic_roberta_best.pt`: Best performing model checkpoint
- `toxic_roberta.pt`: Final model checkpoint