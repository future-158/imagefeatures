import argparse
import shutil
import os
import filecmp
import pickle
from itertools import chain, dropwhile, filterfalse
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from sys import getsizeof
from typing import *
from typing import Any, List
import json

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import MISSING, DictConfig, OmegaConf
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

cfg = OmegaConf.load('pipeline2/config.yaml')

tabular_features = [
    'lag00_u',
    'lag00_v',
    'lag00_temp',
    'lag00_sst',
    'lag00_qff',
    'lag00_rh',
    'lag00_vis',
    'lag00_ASTD',
    'lag00_Td',
    'lag00_temp-Td',
    'lag00_sst-Td',
    'hour',
    'month']

aux_vars  = [
    'split',
    'label']

usecols = {
    'images.name': 'filename',
    'annotations.air_temp': 'lag00_temp',
    'annotations.humidity': 'lag00_rh',
    'annotations.pressure': 'lag00_qff',
    'annotations.visibility': 'lag00_vis',
    'annotations.wind_dir': '',
    'annotations.wind_speed': '',
    'annotations.sst': 'lag00_sst',
    'datetime': '',
    'label': '',
    'stem': '',
    'split': '',
    'info.CCTV_code': 'obs_code',
    }

usecols = {k:v if v != '' else k for k,v in usecols.items()}
all_cols = pd.read_csv(cfg.json_path, nrows=1).columns
df = pd.read_csv(cfg.json_path, usecols=usecols, parse_dates=['datetime'])
df = df[df.split == 'test']
df = df.rename(columns=usecols)

enc = OneHotEncoder(handle_unknown='ignore')
ohe_arr = pd.DataFrame(
    data=enc.fit_transform(df[['obs_code']]).toarray(),
    columns=enc.categories_[0]).astype(np.int64).values

df['lag00_T'] = df.lag00_temp + 273.15
df['lag00_ASTD'] = df.lag00_temp - df.lag00_sst
df['lag00_Td']= df.lag00_temp-((100-df.lag00_rh)/5)*np.sqrt(df['lag00_T']/300)-0.00135*(df.lag00_rh-84)**2+0.35
df['lag00_temp-Td'] = df.lag00_temp - df.lag00_Td
df['lag00_sst-Td'] =  df.lag00_sst - df['lag00_Td']
df['lag00_u'] = df['annotations.wind_speed'] * np.sin(np.deg2rad(df['annotations.wind_dir']))
df['lag00_v'] = df['annotations.wind_speed'] * np.cos(np.deg2rad(df['annotations.wind_dir']))
wind_cols = ['lag00_u', 'lag00_v', 'annotations.wind_speed', 'annotations.wind_dir']

df['hour'] = df.datetime.dt.hour
df['month'] = df.datetime.dt.month

with sqlite3.connect(cfg.img_db_path) as con:
    img_df = pd.read_sql('select * from img_feature;', con)
    # img_df.validation.value_counts()

img_features = [ 'msv_sel', 'msv_sel50', 'msharp_sel',
       'msharp_sel50', 'msv_atm', 'msv_diff', 'h_msharp_sel', 'h_msharp_sel2',
       'h_msharp_sel3', 'h_msharp_sel22', 'h_msharp_sel33', 'h_msharp_sel222',
       'h_msharp_sel333', 'psnr0', 'psnr1', 'psnr2', 'px_intensity1',
       'px_intensity2']


df = pd.merge(df, img_df[[*img_features, 'filename']], on=['filename'])

df_xr = (
    df
    .drop_duplicates(subset=['filename'])
    .set_index('filename')    
    [[*tabular_features,*img_features, *aux_vars]]
    .dropna()
    .to_xarray()
)



ds = xr.open_zarr(cfg.img_zarr_path)
cumsum = 0
# for split in ['train', 'val', 'test']:
for split in ['test']:
    nc_dest = Path(cfg.output_dir) / f'{split}.nc'
    if Path(nc_dest).exists():
        continue
    
    iloc = np.flatnonzero(df_xr.split == split)
    loc = df_xr.isel(filename=iloc).filename
    
    merged = xr.merge([
        ds.sel(filename=loc),
        df_xr.sel(filename=loc).drop_vars(['split'])
        ])

    print(split)
    this_sum = merged.filename.size
    cumsum += this_sum
    print(this_sum)    
    print(cumsum)    
    merged.to_netcdf(nc_dest)

    iloc = np.flatnonzero(df.split == split)
    dest = Path(cfg.output_dir)  / f'{split}.npy'
    np.save(dest, ohe_arr[iloc])

