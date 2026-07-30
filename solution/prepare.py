import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import io
import os
import time
import numpy as np
from PIL import Image, ImageFilter

IMAGE_SIZE = 128



#transform the .parquet data into Tensors with meta data
def transform_to_tensor(src):
    df = pd.read_parquet(src)
    n = len(df)

    X = torch.empty((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    y = torch.empty(n, dtype=torch.long)
    # meta[:, 0] = width, meta[:, 1] = height
    meta = torch.empty((n, 6), dtype=torch.float32)

    #iterate over all images
    for i, (img_bytes, label, w, h, res_mean, res_std, edges, sat) in enumerate(zip(df['image'], df['label'], df['width'], df['height'], df['residual_mean'], df['residual_std'], df['edge_density'], df['saturation'])):
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        X[i] = transforms.functional.pil_to_tensor(img)
        y[i] = label
        # 640 was the maximum value in the train data 
        # normalization [0,1]
        meta[i, 0] = w / 640.0
        meta[i, 1] = h / 640.0

        meta[i, 2] = res_mean / 50.0
        meta[i, 3] = res_std / 50.0
        meta[i, 4] = edges / 255.0
        meta[i, 5] = sat / 255.0


        if i % 2000 == 0:
            print(f"{i}/{n} processed")

    #save the data in train_tensors1, calibration_tensors1, validation_tensors1
    name = os.path.basename(src).replace('.parquet', '').replace('_cleaned', '')
    torch.save({'X': X, 'y': y, 'meta': meta}, f'artifacts/{name}_tensors1.pt')
    print(f"Gespeichert: {X.shape[0]} Bilder -> artifacts/{name}_tensors1.pt")


if __name__ == '__main__':
    start = time.time()
    os.makedirs('artifacts', exist_ok=True)

    transform_to_tensor('artifacts/train_cleaned.parquet')
    transform_to_tensor('artifacts/validation_cleaned.parquet')
    transform_to_tensor('artifacts/calibration_cleaned.parquet')
    transform_to_tensor('artifacts/validation_augmented_cleaned.parquet')

    print(f"Time: {time.time() - start:.2f} seconds")