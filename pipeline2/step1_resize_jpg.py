import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from io import BytesIO
from itertools import product
from multiprocessing import Lock, Process
from pathlib import Path
from random import sample
from typing import *

# import cv2
import joblib
import numpy as np
# import pandas as pd
import torch
import tqdm
import xarray as xr
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from xarray.core.duck_array_ops import asarray
from xarray.core.parallel import dataset_to_dataarray
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image


cfg = OmegaConf.load('pipeline2/config.yaml')
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class Resizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize([256,256]),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            return x 
transforms = torch.nn.Sequential(
    # T.RandomCrop(224),
    # T.RandomHorizontalFlip(p=0.3),
    T.Resize([256,256])

)

files = [
    x for x 
    in Path(cfg.img_dir).glob('**/*.jpg')
    ]

resizer = Resizer()
batch_size = 512
num_files = len(files)
steps, residue = np.divmod(num_files, batch_size)
if residue >0: steps += 1

cc, yy, xx = np.meshgrid(
    np.arange(3, dtype=np.uint8),
    np.arange(256, dtype=np.uint8),
    np.arange(256, dtype=np.uint8),
    indexing='ij'
)

for nth_step in tqdm.tqdm(range(steps)):        
    iloc = slice(nth_step*batch_size, (nth_step+1)*batch_size)
    batch_files = files[iloc]
    filenames = [x.name for x in batch_files]
    batch_img = torch.stack([
        read_image(file.as_posix())
        for file
        in batch_files])
    
    resized_imgs = resizer(batch_img)
    # c = np.memmap('img.npy', dtype='uint8', mode='w+', shape=(num_files,3, 256, 256), order='C')
    # c[iloc] = resized_imgs.numpy()
    
    filenames = np.asarray(filenames, dtype='U50')
    da = xr.DataArray(
        data=resized_imgs,
        # data = {'rgb':},
        # name='rgb',
        dims=["filename", "cc", "yy", "xx"],
        coords=dict(
            channel=(["cc", "yy", "xx"], cc),
            height=(["cc", "yy", "xx"], yy),
            width=(["cc", "yy", "xx"], xx),
            filename=filenames,
        )
    )
    
    if nth_step == 0:
        da.to_dataset(name='rgb').to_zarr(cfg.img_zarr_path, mode='w')            
    else:
        da.to_dataset(name='rgb').to_zarr(cfg.img_zarr_path, append_dim='filename', mode='a')

    

