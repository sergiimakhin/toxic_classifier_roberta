import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("model", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)


def plot_training_curves(train_losses, val_losses, val_f1s, save_path="metrics/training_curves.png"):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label="Val Macro F1")
    plt.legend()
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(val_labels, val_preds, save_path="metrics/confusion_matrix.png"):
    """Plot confusion matrix for multi-label classification"""
    # For multi-label, we'll create a confusion matrix for each label
    num_labels = val_labels.shape[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_labels):
        cm = confusion_matrix(val_labels[:, i], val_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"Label {i} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_model(model, model_path):
    """Save model state dict"""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def print_label_distribution(y_train, y_val):
    """Print label distribution for training and validation sets"""
    print("Label distribution in y_train:", np.sum(y_train, axis=0))
    print("Label distribution in y_val:", np.sum(y_val, axis=0))