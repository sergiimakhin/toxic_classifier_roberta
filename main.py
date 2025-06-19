import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_scheduler
from torch.amp import GradScaler

from config import *
from dataset import ToxicCommentsDataset
from loss import FocalLoss, WeightedBCELoss
from model import get_model
from train import train_epoch, eval_model
from utils import (
    create_directories, 
    plot_training_curves, 
    plot_confusion_matrix, 
    save_model, 
    print_label_distribution
)


def calculate_pos_weights(y_train):
    """Calculate positive weights for class imbalance"""
    pos_counts = np.sum(y_train, axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weights = neg_counts / pos_counts
    return torch.tensor(pos_weights, dtype=torch.float32)


def main():
    # Create necessary directories
    create_directories()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/train.csv")
    X = df["comment_text"].fillna("none").values
    y = df[df.columns[2:]].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]  # Stratify on first label
    )
    print_label_distribution(y_train, y_val)

    # Calculate class weights for imbalanced data
    pos_weights = calculate_pos_weights(y_train).to(DEVICE)
    print("Positive weights:", pos_weights.cpu().numpy())

    # Prepare datasets and loaders
    print("Preparing datasets...")
    train_ds = ToxicCommentsDataset(X_train, y_train)
    val_ds = ToxicCommentsDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize model, loss, optimizer, scheduler, and scaler
    print("Initializing model and training components...")
    model = get_model().to(DEVICE)
    
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    # loss_fn = WeightedBCELoss(pos_weight=pos_weights)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    scaler = GradScaler(device=DEVICE)

    # Training tracking
    train_losses, val_losses, val_f1s = [], [], []
    best_f1 = 0.0

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, DEVICE
        )
        
        # Evaluate
        val_loss, val_probs, val_labels = eval_model(
            model, val_loader, loss_fn, DEVICE, THRESHOLD
        )
        
        # Calculate metrics
        val_preds = (val_probs > THRESHOLD).astype(int)
        macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(val_labels, val_preds, average='micro', zero_division=0)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(macro_f1)
        
        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            save_model(model, MODEL_PATH.replace('.pt', '_best.pt'))
            print(f"New best model saved! F1: {best_f1:.4f}")

    # Save final model
    print("\nSaving final model...")
    save_model(model, MODEL_PATH)

    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(train_losses, val_losses, val_f1s)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    val_preds_final = (val_probs > THRESHOLD).astype(int)
    plot_confusion_matrix(val_labels, val_preds_final)

    # Print final classification report
    print("\nFinal Classification Report:")
    target_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print(classification_report(val_labels, val_preds_final, target_names=target_names, zero_division=0))

    print(f"\nTraining completed! Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    main()