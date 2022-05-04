import argparse
import json
import math
import os
import pickle
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import reduce
from itertools import dropwhile, product
from pathlib import Path
from typing import *
from typing import Any, List

import cloudpickle
import hydra
import joblib
import numpy as np
import pandas as pd
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
from sklearn.preprocessing import OneHotEncoder


@hydra.main(config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
        
    id_vars = ['obs_code', 'obs_time']
    label = (
        pd.read_csv(cfg.json_path, parse_dates=['datetime'])
        .rename(columns=dict(datetime='obs_time'))
    )
    tabular = (
        pd.read_parquet(cfg.tabular_path)
        .sort_values(id_vars)
    )
    for obs_code, part in tabular.groupby(['obs_code']):
        assert part.obs_time.is_monotonic
        assert part.obs_time.diff().value_counts().nunique() == 1

    data_vars = ['msv_sel', 'msv_sel50', 'msharp_sel',
       'msharp_sel50', 'msv_atm', 'msv_diff', 'h_msharp_sel', 'h_msharp_sel2',
       'h_msharp_sel3', 'h_msharp_sel22', 'h_msharp_sel33', 'h_msharp_sel222',
       'h_msharp_sel333', 'psnr0', 'psnr1', 'psnr2', 'px_intensity1',
       'px_intensity2']

    data_vars.extend(tabular.columns[tabular.dtypes == np.number])
    assert sum([x.startswith('lag') for x in data_vars]) > 100
    
    with sqlite3.connect(cfg.img_path) as con:
        query = 'select * from img_feature;'
        img_features = (
            pd.read_sql(query, con=con)
            .query('validation == "valid"')
        )

    cat = pd.merge(label, img_features, left_on=['images.name'], right_on=['filename'])

    
    idx_per_hours = []
    pred_hours = [0,1,2,3]
    pred_hours = range(-3, 4) # no sure direction. 
    for pred_hour in pred_hours:
        idx_per_hour = cat[id_vars].copy()
        idx_per_hour.obs_time += pd.Timedelta(hours=pred_hour)
        idx_per_hours.append(idx_per_hour)

    valid_idx =pd.concat(idx_per_hours).drop_duplicates().sort_values(['obs_code','obs_time'])
    assert pd.Index(cat[['obs_code','obs_time']]).isin(pd.Index(valid_idx)).mean() == 1

    tabular = pd.merge(valid_idx, tabular, how='left', on=id_vars)       
    cat = pd.merge(cat, tabular, how='right', on=id_vars)

    # label, split, filename, Images
    # obs_code, obs_time   
    enc = OneHotEncoder(handle_unknown='ignore')
    
    ohe_appendix = pd.DataFrame(
        data=enc.fit_transform(cat[['obs_code']]).toarray(),
        columns=enc.categories_[0]).astype(np.int64)

    np.testing.assert_allclose(
        np.flatnonzero(cat.obs_code == 'BRD'),
        np.flatnonzero(ohe_appendix.BRD == 1)
    )

    data_vars = [*data_vars, *enc.categories_[0]]
    necessary_vars = ['filename', 'label', 'split']
    usecols = [*data_vars, *id_vars, *necessary_vars]
 
    data = (
        cat
        .reset_index(drop=True)
        .join(ohe_appendix)
        [usecols]
    )
    data.to_parquet(cfg.ds_path, index=False)

if __name__ =='__main__':
    main()


