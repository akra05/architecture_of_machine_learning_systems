import pandas as pd
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import time
import argparse

torch.set_num_threads(os.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument('--timeout_seconds', type=int, default=1800)
args = parser.parse_args()

#to determinize the training
torch.manual_seed(42)
np.random.seed(42)

#hyperparameters
IMAGE_SIZE = 128
PRUNE_EPOCH = 10    
PRUNE_THRESHOLD = 0.9


#image_net data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

#indexing needed to delete Images efficient with Prunning
class CachedDatasetWithIdx(Dataset):
    
    def __init__(self, X, y, meta, augment=False):
        self.X = X
        self.y = y
        self.meta = meta
        self.augment = augment
        self.flip = transforms.RandomHorizontalFlip()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx].float() / 255.0
        img = normalize(img)
        if self.augment:
            img = self.flip(img)
        return img, self.meta[idx], self.y[idx], idx

#loss which lowers the impact of gradients from images which are already predicted correctly with high confidence
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
 
    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(
            input, target, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)  
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
k = 32
class CNN(nn.Module):
    # model architecture 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, padding=1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(k, 2*k, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2*k, 4*k, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*k),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8*k + 6, 128),  # +2 for width/height for meta data +4 for other meta data
           
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x, meta):
        f = self.features(x)
        pooled = torch.cat([self.gap(f), self.gmp(f)], dim=1)
        pooled = pooled.flatten(1)
        combined = torch.cat([pooled, meta], dim=1)
        return self.classifier(combined)

#evaluate model after each epoch
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_proba = []
    all_labels = []
    with torch.no_grad():
        for images, meta, labels, idx in dataloader:
            output = model(images, meta)
            loss = criterion(output, labels.long())
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
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

    return total_loss / len(dataloader), correct / total, best_recall_at_fpr

def train():
    start = time.time()

    print("load cached tensors")
    train_data = torch.load('artifacts/train_tensors1.pt')
    X_train, y_train, meta_train = train_data['X'], train_data['y'], train_data['meta']
    print(f"Training loaded: {X_train.shape[0]} images")

    val_data = torch.load('artifacts/validation_tensors1.pt')
    X_val, y_val, meta_val = val_data['X'], val_data['y'], val_data['meta']
    print(f"Validation loaded: {X_val.shape[0]} images")

    # balance dataset
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

    # create datasets for training
    train_dataset = CachedDatasetWithIdx(X_train_bal, y_train_bal, meta_train_bal, augment=True)
    val_dataset = CachedDatasetWithIdx(X_val, y_val, meta_val, augment=False)

    # init data loader
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=0)

    #define learning parameters
    model = CNN()
    criterion = FocalLoss(gamma=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_metric = -1.0
    patience = 5
    no_improve = 0

    # train in epochs
    for epoch in range(50):
        model.train()
        total_loss = 0

        # prune values only needed in PRUNE_EPOCH
        collect_pt = (epoch == PRUNE_EPOCH)
        if collect_pt:
            pt_storage = torch.zeros(len(y_train_bal))

        for images, meta, labels, idx in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(images, meta)
            loss = criterion(output, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if collect_pt:
                with torch.no_grad():
                    proba = torch.softmax(output, dim=1)
                    pt = proba.gather(1, labels.unsqueeze(1)).squeeze(-1)
                    pt_storage[idx] = pt  # an Original-Position ablegen

        train_loss = total_loss / len(train_loader)
        val_loss, val_acc, recall_at_fpr20 = evaluate(model, val_loader, criterion)
        scheduler.step(recall_at_fpr20)

        # delete the top 10% most confident predicted images are deleted
        if collect_pt:
            dynamic_threshold = pt_storage.quantile(PRUNE_THRESHOLD)
            keep_mask = pt_storage < dynamic_threshold
            print(f"Pruning: {keep_mask.sum().item()} of {len(keep_mask)} images kept "
                f"(Threshold: {dynamic_threshold:.3f})")

            X_train_bal = X_train_bal[keep_mask]
            y_train_bal = y_train_bal[keep_mask]
            meta_train_bal = meta_train_bal[keep_mask]

            train_dataset = CachedDatasetWithIdx(X_train_bal, y_train_bal, meta_train_bal, augment=True)
            train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
            f"Recall_AI@FPR<=20%: {recall_at_fpr20:.4f} - "
            f"LR: {current_lr:.6f} - Time: {elapsed:.0f}s"
        )

        # after each epoch save model if performance is best yet
        os.makedirs('artifacts/task02', exist_ok=True)

        if recall_at_fpr20 > best_metric:
            best_metric = recall_at_fpr20
            no_improve = 0
            torch.save(model.state_dict(), 'artifacts/task02/model.pt')
            print(f"  !!! new best Model (Recall_AI@FPR<=20%: {recall_at_fpr20:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after Epoch {epoch+1}")
                break

        # save last model to continue computation with more time
        # end if 95% of time is over
        if elapsed > args.timeout_seconds * 0.95:
            print("Timeout reached")
            torch.save(model.state_dict(), 'artifacts/task02/model_last.pt')
            break

    print("Training finished")

if __name__ == '__main__':
    train()