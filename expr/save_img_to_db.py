import argparse
import math
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Lock, Process
from pathlib import Path
from queue import Queue
from typing import *

import numpy as np
import tqdm
from joblib import Parallel, delayed
from omegaconf import MISSING, DictConfig, OmegaConf
from PIL import Image

cfg = OmegaConf.load('conf/config.yml')

src_dir = Path(cfg.catalogue.image_dir)
out_path = Path(cfg.catalogue.image_db_path)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

with sqlite3.connect(cfg.out_path) as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS IMG
            (   filename TEXT NOT NULL UNIQUE,
                idatetime INTEGER NOT NULL UNIQUE,
                station_code TEXT,
                label INTEGER,
                data BLOB NOT NULL,
                PRIMARY KEY(filename)
                );'''
                )

files = {
    file
    for file
    in Path(src_dir).glob("**/*.jpg")
    }

@dataclass
class CCTV:
    filename: str
    idatetime: int
    station_code: str
    label: int
    data: bytes

with sqlite3.connect(src_dir,  timeout=30.0) as con:
    cur = con.cursor()
    pe_filenames = cur.execute('SELECT filename from IMG;').fetchall()

files = {
    file for file in files
    if file.name not in pe_filenames
    }

def process_file(q: Queue, file: Path ) -> int:
    _, _, station_code, dt_label = file.stem.split('_')
    _idatetime, label = dt_label.split('-')
    idatetime = int(_idatetime)
    # img = CCTV(file.name, idatetime, station_code, int(label), Image.open(file).tobytes())
    tup = (file.name, idatetime, station_code, int(label), Image.open(file).tobytes())
    q.put(tup)
    return 0

batch_size = 1024
num_batches, residue = np.divmod(len(files), batch_size)
if residue > 0:
    num_batches +=1

files = list(files)
for i in range(num_batches):
    batch_files = files[i*batch_size:(i+1)*batch_size]

    q = Queue()
    Parallel(n_jobs=40, prefer='threads', require='sharedmem')(delayed(process_file)(q, file) for file in tqdm.tqdm(batch_files))
    tups = []
    while q.qsize() > 0:
        tup = q.get()
        tups.append(tup)

    with sqlite3.connect(cfg.img_db_path,  timeout=30.0) as con:
        cur = con.cursor()
        query = '''INSERT OR IGNORE INTO IMG (filename, idatetime, station_code, label, data)
        VALUES (?, ?, ?, ?, ?);
        '''
        cur.executemany(query,tups)
    del tups



    
    


    










