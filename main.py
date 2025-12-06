import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import glob
import torch
import numpy as np


from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, CacheDataset, DataLoader, partition_dataset
from monai.bundle import download, load,ConfigParser
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord,EnsureChannelFirstd,RandCropByPosNegLabeld,
    RandRotate90d, RandFlipd, RandZoomd, RandGaussianNoised, CropForegroundd
)

set_determinism(seed=0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bundle_dir = "prostate_mri_anatomy"
download(name=bundle_dir, bundle_dir=".", progress=True)

#store it into the data variables
data = [
    {"image": row["T2W_NIFTI"], "label": row["Gland_NIFTI"]}
    for _, row in df.iterrows()
]

#store paths of label an/d input images
data = [
    {'image': f"/content/file_data/{d['image']}", 'label': f"/content/file_data/{d['label']}"}
    for d in data

]
print(data)

bundle_dir = "prostate_mri_anatomy"
model_path = os.path.join(bundle_dir, "models", "model.pt")
print(model_path)

print(os.listdir(bundle_dir))
# models directory
models_dir = os.path.join(bundle_dir, "models")
if os.path.exists(models_dir):
    print(os.listdir(models_dir))

#get the model weights and parse the config file
config_parser = ConfigParser()
config_file = os.path.join(bundle_dir, "configs", "train.json")
print(config_file)
if os.path.exists(config_file):
    model = torch.load(model_path, map_location=device)

    config_parser.read_config(config_file)
    network_def=config_parser.get_parsed_content("network_def", instantiate=True)
    network_def.load_state_dict(model)
    model=network_def
    print(config_parser.get("network_def", default=None))
    model.eval()
    model=model.to(device)

    print("Model loaded successfully!")