import torch
import torch.nn.utils as torch_utils
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from config import MAX_GRAD_NORM


def train_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss.item()}")
                continue

        scaler.scale(loss).backward()
        
        # Gradient clipping - import MAX_GRAD_NORM from config
        from config import MAX_GRAD_NORM
        scaler.unscale_(optimizer)
        torch_utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Call scheduler AFTER optimizer.step() - this is the correct order
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def eval_model(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(outputs, labels)
                
            total_loss += loss.item()
            
            # Store logits and probabilities
            logits = outputs.cpu().numpy()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_logits.append(logits)
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    val_logits = np.concatenate(all_logits)
    val_probs = np.concatenate(all_preds)
    val_labels = np.concatenate(all_labels)
    val_preds = (val_probs > threshold).astype(int)

    # Print diagnostics
    print(f"🧪 Logits range: [{val_logits.min():.3f}, {val_logits.max():.3f}]")
    print(f"🧪 Probs range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")
    print("🧪 Sample val_probs:", val_probs[:3])
    print("🧪 Sample val_preds:", val_preds[:3])
    print("🧪 Sample val_labels:", val_labels[:3])

    return total_loss / len(loader), val_probs, val_labels