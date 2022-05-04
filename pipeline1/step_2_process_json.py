import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import product
from pathlib import Path
from typing import *

import hydra
import numpy as np
import pandas as pd
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print('processing', __file__)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    # cfg = OmegaConf.load('run_config.yaml')
    # cfg.station_code = 'SF_0003'

    cfg.json_dir = ( Path(cfg.root) /  cfg.json_dir).as_posix()
    cfg.json_path = (Path(cfg.root) /  cfg.json_path).as_posix()

    files = list(Path(cfg.json_dir).glob('**/*.json'))
    txt_list = []
    for file in tqdm.tqdm(files):
        with open(file,'r', encoding='utf-8') as f:           
            txt = f.read()
            txt = txt.strip()
            assert txt.startswith('{')
            assert txt.endswith('}')
            txt_list.append(txt)

    print('total file size is: ', len(files))

    string_json  ='[' + ','.join(txt_list) + ']'
    df = pd.json_normalize(json.loads(string_json))
    df['datetime'] = df['info.CCTV_date']
    df['label'] = df['annotations.class_name']        
    df['h_px'] = df["images.horizon_pixel"]
    df['px_1k'] = df["images.1km_pixel"]
    df['px05k'] = df["images.500m_pixel"]
    df['stem'] = df['images.name'].astype(str).str.split('.',expand=True).iloc[:,0]

    split_getter = {
        '1.Training':'train',
        '2.Validation':'val',
        '3.Test':'test'
    }

    code_book = {
        "BRD":"백령도",
        "DBD":"대부도",
        "DJD":"덕적도",
        "GHD":"강화도",
        "JD":"지도",
        "KOEM":"연안부두",
        "SD":"송도",
        "YHD":"영흥도",
        "YJD":"영종도",
        "YPD":"연평도",   
        "PTDJ":"평택당진항",
        "INCHEON":"인천항",
        "DAESAN":"대산항"
    }

    code_getter = {v:k for k,v in  code_book.items()}

    df['split'] = [x.parent.name for x in files]
    # df.split = df.split.map(split_getter)
    df['obs_code'] = [x.parent.parent.name for x in files]
    df.obs_code = df['obs_code'].map(code_getter)

    # df.c = df.c.map(lambda x: x)
    # class_map = {'No-fog': 3, 'Low-vis': 2, 'Fog': 1, 'Dense-fog': 0}
    class_map = {'No-fog': 0, 'Low-vis': 1, 'Fog': 2, 'Dense-fog': 3}
    df.label = df['label'].map(class_map)
    df['datetime'] = pd.to_datetime(df['datetime'])
    label = df.set_index('datetime').squeeze()
    Path(cfg.json_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.json_path)
if __name__ == '__main__':
    main()


