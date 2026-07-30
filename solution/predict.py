import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os
from train import CNN, normalize

IMAGE_SIZE = 128


class PredictDataset(Dataset):
    def __init__(self, df, has_labels=False):
        self.images = df['image'].tolist()
        self.has_labels = has_labels
        if has_labels:
            self.labels = df['label'].tolist()
        else:
            self.row_ids = df['row_id'].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bytes = self.images[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        w, h = img.size
        meta = torch.tensor([w / 640.0, h / 640.0], dtype=torch.float32)

        img = transforms.Resize(IMAGE_SIZE)(img)
        img = transforms.CenterCrop(IMAGE_SIZE)(img)
        img = transforms.functional.pil_to_tensor(img).float() / 255.0
        img = normalize(img)

        if self.has_labels:
            return img, meta, self.labels[idx]
        return img, meta, self.row_ids[idx]


def find_threshold(model):
    df_cal = pd.read_parquet('data/calibration/')
    df_cal['label'] = df_cal['source_class'].apply(lambda x: 0 if x == 0 else 1)

    dataset = PredictDataset(df_cal, has_labels=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_proba = []
    all_labels = []

    with torch.no_grad():
        for images, meta, labels in dataloader:
            output = model(images, meta)
            proba = torch.softmax(output, dim=1)[:, 1]
            all_proba.extend(proba.tolist())
            all_labels.extend(labels.tolist())

    all_proba = np.array(all_proba)
    all_labels = np.array(all_labels)
    real_mask = all_labels == 0

    best_threshold = 0.5
    best_recall = 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (all_proba >= t).astype(int)
        fpr = (preds[real_mask] == 1).mean()
        recall = (preds[all_labels == 1] == 1).mean()
        if fpr <= 0.20 and recall > best_recall:
            best_recall = recall
            best_threshold = t

    print(f"calibrated threshhold: {best_threshold:.2f} (Recall_AI: {best_recall:.3f})")
    return best_threshold


def predict():
    model = CNN()
    model.load_state_dict(torch.load('artifacts/task02/model.pt', weights_only=True))
    model.eval()

    
    threshold = find_threshold(model)    
    df_pred = pd.read_parquet('data/predict/')
    dataset = PredictDataset(df_pred, has_labels=False)
    dataloader = DataLoader(dataset, batch_size=48, shuffle=False, num_workers=0)

    all_row_ids = []
    all_preds = []
    with torch.no_grad():
        for images, meta, row_ids in dataloader:
            output = model(images, meta)
            proba = torch.softmax(output, dim=1)[:, 1]
            preds = (proba >= threshold).long()
            all_row_ids.extend(row_ids.tolist())
            all_preds.extend(preds.tolist())


    os.makedirs('artifacts/task02', exist_ok=True)
    df_out = pd.DataFrame({'row_id': all_row_ids, 'predicted_label': all_preds})    
    df_out = df_out.sort_values('row_id')
    df_out.to_csv('artifacts/task02/predictions.csv', index=False)      
    print(f"predictions saved: {len(df_out)} rows")   
    print(df_out['predicted_label'].value_counts())  

if __name__ == '__main__':
    predict()