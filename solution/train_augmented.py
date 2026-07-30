import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os
import time
import argparse
from train import CNN, normalize, IMAGE_SIZE, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--timeout_seconds', type=int, default=1800)
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)


class RobustCachedDataset(Dataset):
    def __init__(self, X, y, meta, augment=False, robustify_prob=0.5):
        self.X = X
        self.y = y
        self.meta = meta
        self.augment = augment
        self.robustify_prob = robustify_prob
        self.flip = transforms.RandomHorizontalFlip()

    def __len__(self):
        return len(self.y)

    def _degrade(self, img):
        choice = np.random.choice(['jpeg', 'blur', 'downscale', 'noise'])

        if choice == 'jpeg':
            pil = transforms.functional.to_pil_image(img)
            quality = int(np.random.randint(30, 90))
            buf = io.BytesIO()
            pil.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            pil2 = Image.open(buf).convert('RGB')
            img = transforms.functional.pil_to_tensor(pil2).float() / 255.0

        elif choice == 'blur':
            sigma = float(np.random.uniform(0.5, 2.0))
            img = transforms.functional.gaussian_blur(img, kernel_size=5, sigma=sigma)

        elif choice == 'downscale':
            scale = float(np.random.uniform(0.4, 0.8))
            _, h, w = img.shape
            small = torch.nn.functional.interpolate(
                img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False
            )
            img = torch.nn.functional.interpolate(
                small, size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)

        elif choice == 'noise':
            noise = torch.randn_like(img) * float(np.random.uniform(0.01, 0.05))
            img = (img + noise).clamp(0, 1)

        return img

    def __getitem__(self, idx):
        img = self.X[idx].float() / 255.0
        if self.augment:
            img = self.flip(img)
            if np.random.rand() < self.robustify_prob:
                img = self._degrade(img)
        img = normalize(img)
        return img, self.meta[idx], self.y[idx]


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_proba, all_labels = [], []
    with torch.no_grad():
        for images, meta, labels in dataloader:
            output = model(images, meta)
            loss = criterion(output, labels.long())
            total_loss += loss.item()
            proba = torch.softmax(output, dim=1)[:, 1]
            all_proba.extend(proba.tolist())
            all_labels.extend(labels.tolist())

    proba = np.array(all_proba)
    labels_arr = np.array(all_labels)
    real_mask = labels_arr == 0
    best_recall_at_fpr = 0.0
    for t in np.arange(0.05, 1.0, 0.01):
        preds_t = (proba >= t).astype(int)
        fpr = (preds_t[real_mask] == 1).mean()
        if fpr <= 0.20:
            recall = (preds_t[labels_arr == 1] == 1).mean()
            best_recall_at_fpr = max(best_recall_at_fpr, recall)

    return total_loss / len(dataloader), best_recall_at_fpr


def train():
    start = time.time()

    train_data = torch.load('artifacts/train_tensors1.pt')
    X_train, y_train, meta_train = train_data['X'], train_data['y'], train_data['meta']

    val_data = torch.load('artifacts/validation_tensors1.pt')
    X_val, y_val, meta_val = val_data['X'], val_data['y'], val_data['meta']

    val_aug_data = torch.load('artifacts/validation_augmented_tensors1.pt')
    X_val_aug, y_val_aug, meta_val_aug = val_aug_data['X'], val_aug_data['y'], val_aug_data['meta']

    # Balancieren
    counts = torch.bincount(y_train)
    n_real = counts[0].item()
    real_idx = (y_train == 0).nonzero(as_tuple=True)[0]
    ai_idx = (y_train == 1).nonzero(as_tuple=True)[0]
    ai_idx_sampled = ai_idx[torch.randperm(len(ai_idx))[:n_real]]
    all_idx = torch.cat([real_idx, ai_idx_sampled])
    all_idx = all_idx[torch.randperm(len(all_idx))]
    X_train_bal = X_train[all_idx]
    y_train_bal = y_train[all_idx]
    meta_train_bal = meta_train[all_idx]
    print(f"Balanced: {len(X_train_bal)} images ({n_real} Real, {n_real} AI)")

    train_dataset = RobustCachedDataset(X_train_bal, y_train_bal, meta_train_bal, augment=True, robustify_prob=0.3)
    val_dataset = RobustCachedDataset(X_val, y_val, meta_val, augment=False)
    val_aug_dataset = RobustCachedDataset(X_val_aug, y_val_aug, meta_val_aug, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=0)
    val_aug_loader = DataLoader(val_aug_dataset, batch_size=48, shuffle=False, num_workers=0)

    model = CNN()
    model.load_state_dict(torch.load('artifacts/task02/model.pt', weights_only=True))
    print("Task-2-Checkpoint loaded, start fine-tuning with robustness augmentation")

    criterion = FocalLoss(gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_metric = -1.0
    patience = 7
    no_improve = 0

    for epoch in range(50):
        model.train()
        total_loss = 0
        for images, meta, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(images, meta)
            loss = criterion(output, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, recall_clean = evaluate(model, val_loader, criterion)
        _, recall_aug = evaluate(model, val_aug_loader, criterion)
        scheduler.step(recall_aug)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f} - "
            f"Recall_clean: {recall_clean:.4f} - Recall_augmented: {recall_aug:.4f} - "
            f"LR: {current_lr:.6f} - Time: {elapsed:.0f}s"
        )

        os.makedirs('artifacts/task03', exist_ok=True)

        if recall_aug > best_metric:
            best_metric = recall_aug
            no_improve = 0
            torch.save(model.state_dict(), 'artifacts/task03/model.pt')
            print(f"  !!! new best model (Recall_augmented: {recall_aug:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after Epoch {epoch+1}")
                break

        if elapsed > args.timeout_seconds * 0.95:
            torch.save(model.state_dict(), 'artifacts/task03/model_last.pt')
            print("Timeout reached")
            break

    print("Training finished")


if __name__ == '__main__':
    train()