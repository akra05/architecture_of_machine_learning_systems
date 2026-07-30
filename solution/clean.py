import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import io
import os
import time
from PIL import Image, ImageFilter
import numpy as np

start = time.time()
#Hyperparameters
IMAGE_SIZE = 128
to_tensor = transforms.ToTensor()

#extract center of image
def center_crop(img, size=IMAGE_SIZE):
    w, h = img.size

    top = (h - size) // 2
    left = (w -size) // 2
    bottom = top + size
    right = left + size

    return img.crop((left, top, right, bottom))

#return img as bytes
def img_to_bytes(img):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()  

def extract_meta_features(img: Image.Image) -> np.ndarray:
    """Schnelle, handgefertigte Features zusätzlich zu Breite/Höhe.
    Läuft in wenigen Millisekunden pro Bild."""
    img_np = np.array(img.convert('RGB'), dtype=np.float32)

    # Rausch-Residual: Original minus weichgezeichnete Version
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_np = np.array(blurred.convert('RGB'), dtype=np.float32)
    residual = img_np - blurred_np

    # Kennzahlen pro Kanal gemittelt
    residual_mean = np.abs(residual).mean()
    residual_std = residual.std()

    # Kantendichte (Sobel-artig über PIL)
    edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_density = np.array(edges, dtype=np.float32).mean()

    # Sättigungsstatistik (KI-Bilder oft unnatürlich gesättigt/entsättigt)
    hsv = img.convert('HSV')
    saturation = np.array(hsv, dtype=np.float32)[:, :, 1].mean()

    return np.array([residual_mean, residual_std, edge_density, saturation], dtype=np.float32)

#main 
def clean_data(src):
    df = pd.read_parquet(src)

    # merge labels: 
    # label 0 : real images
    # label 1 : ai generated images 
    df['label'] = df['source_class'].apply(lambda x: 0 if x == 0 else 1)

    # delete duplicates
    df = df.drop_duplicates(subset=['image'])
    print(f"After duplicates were removed:  {len(df)} images")

    # save needed meta data 
    sizes = [Image.open(io.BytesIO(b)).size for b in df['image']]
    df['width'] = [s[0] for s in sizes]
    df['height'] = [s[1] for s in sizes]
    df['aspect_ratio'] = df['width'] / df['height']

    print(df.groupby('label')[['width', 'height', 'aspect_ratio']].describe())

    # now the filters are applied
    # first: image is under MIN_DIM width
    # second: images with width 0.4 or lower and 2.5 or higher factor width/height
    MIN_DIM = 64        
    MIN_ASPECT = 0.4    
    MAX_ASPECT = 2.5

    before = len(df)
    df = df[(df['width'] >= MIN_DIM) & (df['height'] >= MIN_DIM)]
    df = df[(df['aspect_ratio'] >= MIN_ASPECT) & (df['aspect_ratio'] <= MAX_ASPECT)]
    print(f"After filters: {len(df)} of {before} images "
          f"({before - len(df)} removed)")
    print(df['label'].value_counts())

    #iterate over dataframe and resize all images and append meta features
    cleaned_images = []
    extra_features = []

    for img in df['image']:
        img = Image.open(io.BytesIO(img))

        feats = extract_meta_features(img)
        extra_features.append(feats)

        ratio = 128 / min(img.size)
        new_w = int(img.size[0] * ratio)
        new_h = int(img.size[1] * ratio)
        img = img.resize((new_w, new_h))
        cropped_img = center_crop(img)
        byte_image = img_to_bytes(cropped_img)
        cleaned_images.append(byte_image)

    #add additional meta data
    df_cleaned = df.copy()
    df_cleaned['image'] = cleaned_images
    df_cleaned['residual_mean'] = [f[0] for f in extra_features]
    df_cleaned['residual_std'] = [f[1] for f in extra_features]
    df_cleaned['edge_density'] = [f[2] for f in extra_features]
    df_cleaned['saturation'] = [f[3] for f in extra_features]

    #save .parquet on disc
    os.makedirs('artifacts', exist_ok=True)
    name = os.path.basename(src)
    df_cleaned.to_parquet(f'artifacts/{name}_cleaned.parquet')

#repeat the process for train, validation and calibration
if __name__ == "__main__":
    src = 'data/train'
    clean_data(src)
    src = 'data/validation'
    clean_data(src)
    src = 'data/calibration'
    clean_data(src)
    clean_data('data/validation_augmented')
    
end = time.time()
print(f"Time: {end - start:.2f} seconds")