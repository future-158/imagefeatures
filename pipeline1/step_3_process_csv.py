import argparse
import json
import math
import os
import pickle
import re
import shutil
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import reduce
from itertools import product
from pathlib import Path
from typing import *
from typing import Any, List

import hydra
import joblib
import numpy as np
import pandas as pd
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
# from sklearn.preprocessing import OneHotEncoder

def get_xy(data_csv_path):
    SAMPLES_PER_HOUR = 60
    LAG_IN_MINUTES = 120
    SPAN_IN_MINUTES = 5

    df = pd.read_csv(data_csv_path, index_col=['obs_time'], parse_dates=['obs_time'], infer_datetime_format=True, encoding='cp949')
    id_vars =  ['obs_code', 'obs_name']
    value_vars =['water_temperature', 'temperature', 'pressure',
        'wind_direction', 'wind_speed', 'wind(U)', 'wind(V)', 'humidity',
        'visibility']

    renamer = {
        'wind_direction': 'wd',
        'wind_speed': 'ws',
        'temperature': 'temp',
        'wind(U)': 'u',
        'wind(V)': 'v',
        'water_temperature': 'sst',
        'pressure': 'qff',
        'humidity': 'rh',
        'visibility': 'vis',
        'obs_time':'datetime'}

    df = df.rename(columns=renamer)
    df['cls_vis'] = df.vis.le(1000).astype(int)
    df['fog_count'] =  df['cls_vis'].rolling('60T').sum()
    data_cols = ['u','v','temp', 'sst', 'qff', 'rh','vis']

    df['T'] = df.temp + 273.15
    df['ASTD'] = df.temp - df.sst
    df['Td']= df.temp-((100-df.rh)/5)*np.sqrt(df['T']/300)-0.00135*(df.rh-84)**2+0.35
    df['temp-Td'] = df.temp - df.Td
    df['sst-Td'] =  df.sst - df['Td']

    lagcols = [
        *data_cols,
        'ASTD','Td','temp-Td','sst-Td'
        ]

    assert (df.index[1:] - df.index[:-1]).nunique() == 1
    assert (df.index[1:] - df.index[:-1]).value_counts().idxmax() == pd.Timedelta(minutes=1)

    df_wide = pd.concat(
        [df.shift(lag)[lagcols].add_prefix(f'lag{lag:02d}_') for lag in range(0, LAG_IN_MINUTES, SPAN_IN_MINUTES)], axis=1) # exclusive임.

    std_cols = df.rolling('60T')[['ws', 'ASTD', 'rh']].std().add_prefix('std_')
    summary_cols = df.rolling('60T')[['fog_count']].sum().add_prefix('reduce_')
    df_wide = df_wide.join(std_cols.join(summary_cols))
    df_wide['obs_time'] = df_wide.index
    df_wide['obs_code'] = data_csv_path.stem.split('_')[-1]
    return df_wide

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.tabular_path = (Path(cfg.root ) / cfg.tabular_path).as_posix()
    folder = Path(cfg.root ) / cfg.tabular_dir
    files = [x for x in folder.glob('*.csv')]
    assert len(files) == 13

    dfs = [get_xy(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)

    df.to_parquet(cfg.tabular_path, compression='gzip', index=False) 
    # pd.to_csv(df, cfg.tabular_path, index=False)

if __name__ =='__main__':
    main()

    
    