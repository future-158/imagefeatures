import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Lock, Process
from pathlib import Path
from typing import *

import cv2
import hydra
import joblib
import numpy as np
import pandas as pd
import tqdm
from hydra.utils import get_original_cwd, to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# from blazingsql import BlazingContext

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def create_if_not_exists(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS IMG_FEATURE
            (   filename TEXT NOT NULL UNIQUE,
                validation TEXT NOT NULL,
                msv_sel REAL NOT NULL,
                msv_sel50 REAL NOT NULL,
                msharp_sel REAL NOT NULL,
                msharp_sel50 REAL NOT NULL,
                msv_atm REAL NOT NULL,
                msv_diff REAL NOT NULL,
                h_msharp_sel REAL NOT NULL,
                h_msharp_sel2 REAL NOT NULL,
                h_msharp_sel3 REAL NOT NULL,
                h_msharp_sel22 REAL NOT NULL,
                h_msharp_sel33 REAL NOT NULL,
                h_msharp_sel222 REAL NOT NULL,
                h_msharp_sel333 REAL NOT NULL,
                psnr0 REAL NOT NULL,
                psnr1 REAL NOT NULL,
                psnr2 REAL NOT NULL,
                px_intensity1 REAL NOT NULL,
                px_intensity2 REAL NOT NULL,
                PRIMARY KEY(filename)
                );'''
                )

# def handle_bsql():
#     pd.options.display.max_rows = 100
#     cluster = LocalCUDACluster()
#     client = Client(cluster)
#     bc = BlazingContext(dask_client=client)
#     bc.create_table('img', 'tmp.gzip', file_format='parquet')
#     gdf = bc.sql('SELECT mean(rgb) FROM img group by filename, channel')


@dataclass
class Annotation:
    fullname: str
    filename: str
    h_px: int
    px_1k: int
    px05k: int

@dataclass
class Image_feature:
    filename: str
    validation: str
    msv_sel: float = 0
    msv_sel50: float = 0
    msharp_sel: float = 0
    msharp_sel50: float = 0
    msv_atm: float = 0
    msv_diff: float = 0
    h_msharp_sel: float = 0
    h_msharp_sel2: float = 0
    h_msharp_sel3: float = 0
    h_msharp_sel22: float = 0
    h_msharp_sel33: float = 0
    h_msharp_sel222: float = 0
    h_msharp_sel333: float = 0
    psnr0: float = 0
    psnr1: float = 0
    psnr2: float = 0
    px_intensity1: float = 0
    px_intensity2: float = 0

def process_file(r: Annotation) -> Image_feature:
    h_px1 = r.h_px - 20
    h_px2 = r.h_px + 20
    px_1k = r.px_1k
    px05k = r.px05k
    fullname = r.fullname


    result = {}
    result['filename'] = r.filename
    result['validation'] = 'valid'

    img = cv2.imread(fullname)[:,:,::-1]
    if px05k >= 720:
        result['validation'] = 'tilted'
        return Image_feature(**result)
    # early stop cases
    if img.shape != (720,1280,3):
        result['validation'] = 'defacto_gray'
        return Image_feature(**result)
        
    x_sample_pts = np.array([100, 200, 300, 400, 500, 600])
    y_sample_pts = np.array([   0,  100,  200,  300,  400,  500,  600,  700,  800,  900, 1000,1100, 1200])

    if (img[x_sample_pts][:, y_sample_pts].std(axis=-1) ** 2).sum() < 1e-3:
        result['validation'] = 'dejure_gray'
        return Image_feature(**result)

    def image_colorfulness(img):
        if img.size == 0:
            return


        R, G, B = cv2.split(img)       
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        return stdRoot + (0.3 * meanRoot)



    psnr0_ = image_colorfulness(img[h_px1 - 30 : h_px1, :, :])
    psnr1_ = image_colorfulness(img[px_1k:px05k, :, :])
    psnr2_ = image_colorfulness(img[px05k : px05k + 120, :, :])

    if any([
        psnr0_ is None,
        psnr1_ is None,
        psnr2_ is None,
    ]):
        result['validation'] = 'pnsr_int_zero_size'
        return Image_feature(**result)

    result['psnr0'] = psnr0_
    result['psnr1'] = psnr1_
    result['psnr2'] = psnr2_

    # 왜 hsv는 255로 나누어서 계산하는지 궁금함
    I_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / 255
    sv = I_hsv[:, :, 1] * I_hsv[:, :, 2] * 100 
  
    # atmosphere
    sv_sel = sv[h_px1:h_px2, :]  # below horizon
    result['msv_sel'] = np.mean(sv_sel, None)   
    result['msv_sel50'] = msv_sel50 = sv_sel[sv_sel <= np.quantile(sv_sel, 0.5)].mean()
    
    sv_atm = sv[:335, :]
    result['msv_atm'] = msv_atm = np.mean(sv_atm, None)  # sv for atm
    result['msv_diff'] = msv_diff = msv_atm - msv_sel50  # sv diff

    I_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result['px_intensity1'] = np.mean(I_gray[px_1k:px05k, :], None)
    result['px_intensity2'] = np.mean(I_gray[px05k : px05k + 120, :], None)

    # sharpness
    
    gx, gy = np.gradient(I_gray)
    sharp1 = np.sqrt(gx ** 2 + gy ** 2)
    sharp_sel = sharp1[h_px1:h_px2, :]
    result['msharp_sel'] = np.mean(sharp_sel, None)
    result['msharp_sel50'] = sharp_sel[sharp_sel <= np.quantile(sharp_sel, 0.5)].mean()
         
    # sharpness in horizontal line
    result['h_msharp_sel'] = sharp_sel[sharp_sel >= np.quantile(sharp_sel, 0.8)].mean()

    def calc_partitions(slice_img):
        # sharpness in 500~1km line
        # np.testing.assert_allclose(
        #     slice_img[-1, -128:],
        #     slice_img.reshape(-1, 128)[-1])

        chunk = slice_img.reshape(-1, 10, 128)
        q25 = np.quantile(chunk, 0.25, axis=(0,2))
        q75 = np.quantile(chunk, 0.75, axis=(0,2))
        
        n_chunk = 10
        mean_list = [
            chunk[:,i,:][np.logical_and(
                chunk[:,i,:] >= q25[i],
                chunk[:,i,:] <= q75[i]
            )].mean()
            for i in range(n_chunk)]

        return max(mean_list), min(mean_list)


        # height width
        # height (10  128) 
        # (height 10) 128

    h_sharp_sel2 = sharp1[px_1k:px05k, :]
    result['h_msharp_sel2'] = np.median(h_sharp_sel2, None)
    result['h_msharp_sel222'], result['h_msharp_sel22'] = calc_partitions(h_sharp_sel2)
    
    # sharpness in 500~ line
    h_sharp_sel3 = sharp1[px05k : px05k + 120, :]
    result['h_msharp_sel333'], result['h_msharp_sel33'] = calc_partitions(h_sharp_sel3)
    result['h_msharp_sel3'] = np.median(h_sharp_sel3, None)
    return Image_feature(**result)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))    
    table = pd.read_csv(cfg.json_path, parse_dates=['datetime'])
    table = table.drop_duplicates(subset=['info.name'], keep='last')

    img_dir =  (Path(cfg.img_dir) /   'test') if cfg.stage == 'test' else cfg.img_dir
    valid_files = [
        file
        for file
        # in Path(cfg.img_dir).glob("**/*.jpg")
        in Path(img_dir).glob("**/*.jpg")
        ]

    left = pd.DataFrame(data=dict(
        filename= [x.name for x in valid_files],
        fullname = [x.as_posix() for x in valid_files],
    ))

    table = pd.merge(left, table, how='inner', left_on = 'filename', right_on='images.name')

    # fullname, filename, h_px, px_1k, px05k
    Path(cfg.img_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cfg.img_path) as con:   
        cur = con.cursor()
        tables = cur.execute("SELECT * FROM sqlite_master  WHERE type='table' ").fetchall()
        if not len(tables):
            create_if_not_exists(con)
        query = 'SELECT filename from IMG_FEATURE;'
        pe_table = cur.execute(query).fetchall()
    
    pe_filenames = set(map(lambda x: x[0], pe_table))
    if not cfg.stage == 'test':
        table = table[~table.filename.isin(pe_filenames)]

    usecols  = ['fullname', 'filename', 'h_px', 'px_1k', 'px05k']
    records =  table[usecols]
        
    annots = [
            Annotation(
            fullname = tup.fullname,                
            filename = tup.filename,                
            h_px = tup.h_px,
            px_1k = tup.px_1k,
            px05k = tup.px05k,
            )
            for tup 
            in table[usecols].itertuples()]

    # status_codes = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
    #     delayed(process_file)(annot) for annot in tqdm.tqdm(annots))

    n_samples = len(annots)
    batch_size = 2000

    steps_per_epoch, residue  = np.divmod(n_samples, batch_size)
    if residue > 0: steps_per_epoch+=1

    for ith in tqdm.tqdm( range(steps_per_epoch), desc = 'outer batch', total=steps_per_epoch):
        batch = annots[ith*batch_size:(ith+1)*batch_size]
        batch_img_features = Parallel(n_jobs=-1)(
            delayed(process_file)(annot) for annot in batch)
            # enviroentment 
        with sqlite3.connect(cfg.img_path,  timeout=30.0) as con:
            cur = con.cursor()
            query = '''INSERT OR REPLACE INTO IMG_FEATURE (filename, validation, msv_sel, msv_sel50, msharp_sel, msharp_sel50, msv_atm, msv_diff, h_msharp_sel, h_msharp_sel2, h_msharp_sel3, h_msharp_sel22, h_msharp_sel33, h_msharp_sel222, h_msharp_sel333, psnr0, psnr1, psnr2, px_intensity1, px_intensity2)
            VALUES (:filename, :validation, :msv_sel, :msv_sel50, :msharp_sel, :msharp_sel50, :msv_atm, :msv_diff, :h_msharp_sel, :h_msharp_sel2, :h_msharp_sel3, :h_msharp_sel22, :h_msharp_sel33, :h_msharp_sel222, :h_msharp_sel333, :psnr0, :psnr1, :psnr2, :px_intensity1, :px_intensity2);
            '''
            cur.executemany(query, [bif.__dict__ for bif in batch_img_features])
            # con.commit()
                 
if __name__ == "__main__":
    main()


