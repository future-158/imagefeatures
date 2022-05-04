import argparse
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import *
from typing import Any, List

import autogluon.core as ag
import hydra
import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from autogluon.core.utils import try_import_lightgbm
from autogluon.tabular import TabularDataset, TabularPredictor
from omegaconf import MISSING, DictConfig, OmegaConf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             jaccard_score, multilabel_confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def calc_metrics(obs: pd.Series, pred:np.ndarray, binary=True) -> Dict:
    if binary:
        tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
        return dict(
            ACC=accuracy_score(obs,pred),
            CSI=jaccard_score(obs, pred),
            PAG=precision_score(obs, pred),
            POD=recall_score(obs, pred),
            F1=f1_score(obs,pred),
            TN=tn,
            FP=fp,
            FN=fn,
            TP=tp,
        )

    else:
        metrics = {}
        metrics['ACC'] = accuracy_score(obs, pred )
        metrics['macro_CSI']  = jaccard_score(obs,pred, average='macro')
        metrics['macro_PAG'] = precision_score(obs,pred, average='macro')
        metrics['macro_POD'] = recall_score(obs, pred, average='macro')
        metrics['macro_F1'] = f1_score(obs, pred, average='macro')

        metrics['micro_CSI']  = jaccard_score(obs,pred, average='micro')
        metrics['micro_PAG'] = precision_score(obs,pred, average='micro')
        metrics['micro_POD'] = recall_score(obs, pred, average='micro')
        metrics['micro_F1'] = f1_score(obs, pred, average='micro')
        return metrics


def calc_metrics_nia(obs: np.ndarray, pred:np.ndarray, binary=True) -> Any:
    assert np.unique(obs).size == np.unique(pred).size
    classes = np.unique(obs)
    metrics = defaultdict(dict)
    for i, c in enumerate(classes):            
        obs_ = np.where(obs==c, 1, 0)
        pred_ = np.where(pred==c, 1, 0)
        cmat= confusion_matrix(obs_, pred_).flatten()
        names = ['tn', 'fp', 'fn', 'tp']

        for name, cmat_cell in zip(names, cmat):
            metrics[c][name] = cmat_cell

        metrics[c]['ACC'] = accuracy_score(obs_, pred_)
        metrics[c]['CSI']  = jaccard_score(obs_,pred_)
        metrics[c]['PAG'] = precision_score(obs_,pred_)
        metrics[c]['POD'] = recall_score(obs_, pred_)
        metrics[c]['F1'] = f1_score(obs_, pred_)

    return metrics

    

class Run_type(str, Enum):
    Train = 'train'
    Val = 'val'
    Test = 'test'


def get_train_val_test_with_format(df: pd.DataFrame, pred_hour: int) -> pd.DataFrame:
    id_vars = ['obs_time', 'obs_code']
    df = (
        df
        .set_index(id_vars)
        .drop_duplicates(keep='first')
    )
    label = df.pop('label')   

    label = label.reset_index()
    label.obs_time = label.obs_time - pd.Timedelta(hours=pred_hour)
    label = label.set_index(id_vars)

    target_time = df.index.get_level_values('obs_time') + pd.Timedelta(hours=pred_hour)
    df['hour'] = target_time.hour
    df['month'] = target_time.month
 
    df = (
        df.join(label)
        .dropna()
    )

    split = df.pop('split')
    train_mask = split.astype(str).str.lower() == 'train'
    val_mask = split.astype(str).str.lower() == 'val'
    test_mask = split.astype(str).str.lower() == 'test'

    usecols = """msv_sel msv_sel50 msharp_sel msharp_sel50 msv_atm msv_diff h_msharp_sel h_msharp_sel2 h_msharp_sel3 h_msharp_sel22 h_msharp_sel33
    h_msharp_sel222 h_msharp_sel333 psnr0 psnr1 psnr2 px_intensity1 px_intensity2 lag00_u lag00_v lag00_temp lag00_sst
    lag00_qff lag00_rh lag00_vis lag00_ASTD lag00_Td lag00_temp-Td lag00_sst-Td lag05_u lag05_v lag05_temp lag05_sst
    lag05_qff lag05_rh lag05_vis lag05_ASTD lag05_Td lag05_temp-Td lag05_sst-Td lag10_u lag10_v lag10_temp
    lag10_sst lag10_qff lag10_rh lag10_vis lag10_ASTD lag10_Td lag10_temp-Td lag10_sst-Td lag15_u lag15_v
    lag15_temp lag15_sst lag15_qff lag15_rh lag15_vis lag15_ASTD lag15_Td lag15_temp-Td lag15_sst-Td lag20_u
    lag20_v lag20_temp lag20_sst lag20_qff lag20_rh lag20_vis lag20_ASTD lag20_Td lag20_temp-Td lag20_sst-Td
    lag25_u lag25_v lag25_temp lag25_sst lag25_qff lag25_rh lag25_vis lag25_ASTD lag25_Td lag25_temp-Td
    lag25_sst-Td lag30_u lag30_v lag30_temp lag30_sst lag30_qff lag30_rh lag30_vis lag30_ASTD lag30_Td
    lag30_temp-Td lag30_sst-Td lag35_u lag35_v lag35_temp lag35_sst lag35_qff lag35_rh lag35_vis lag35_ASTD
    lag35_Td lag35_temp-Td lag35_sst-Td lag40_u lag40_v lag40_temp lag40_sst lag40_qff lag40_rh lag40_vis
    lag40_ASTD lag40_Td lag40_temp-Td lag40_sst-Td lag45_u lag45_v lag45_temp lag45_sst lag45_qff lag45_rh
    lag45_vis lag45_ASTD lag45_Td lag45_temp-Td lag45_sst-Td lag50_u lag50_v lag50_temp lag50_sst lag50_qff
    lag50_rh lag50_vis lag50_ASTD lag50_Td lag50_temp-Td lag50_sst-Td lag55_u lag55_v lag55_temp lag55_sst
    lag55_qff lag55_rh lag55_vis lag55_ASTD lag55_Td lag55_temp-Td lag55_sst-Td lag60_u lag60_v lag60_temp
    lag60_sst lag60_qff lag60_rh lag60_vis lag60_ASTD lag60_Td lag60_temp-Td lag60_sst-Td lag65_u lag65_v
    lag65_temp lag65_sst lag65_qff lag65_rh lag65_vis lag65_ASTD lag65_Td lag65_temp-Td lag65_sst-Td lag70_u
    lag70_v lag70_temp lag70_sst lag70_qff lag70_rh lag70_vis lag70_ASTD lag70_Td lag70_temp-Td lag70_sst-Td
    lag75_u lag75_v lag75_temp lag75_sst lag75_qff lag75_rh lag75_vis lag75_ASTD lag75_Td lag75_temp-Td
    lag75_sst-Td lag80_u lag80_v lag80_temp lag80_sst lag80_qff lag80_rh lag80_vis lag80_ASTD lag80_Td
    lag80_temp-Td lag80_sst-Td lag85_u lag85_v lag85_temp lag85_sst lag85_qff lag85_rh lag85_vis lag85_ASTD
    lag85_Td lag85_temp-Td lag85_sst-Td lag90_u lag90_v lag90_temp lag90_sst lag90_qff lag90_rh lag90_vis
    lag90_ASTD lag90_Td lag90_temp-Td lag90_sst-Td lag95_u lag95_v lag95_temp lag95_sst lag95_qff lag95_rh
    lag95_vis lag95_ASTD lag95_Td lag95_temp-Td lag95_sst-Td lag100_u lag100_v lag100_temp lag100_sst lag100_qff
    lag100_rh lag100_vis lag100_ASTD lag100_Td lag100_temp-Td lag100_sst-Td lag105_u lag105_v lag105_temp lag105_sst
    lag105_qff lag105_rh lag105_vis lag105_ASTD lag105_Td lag105_temp-Td lag105_sst-Td lag110_u lag110_v lag110_temp
    lag110_sst lag110_qff lag110_rh lag110_vis lag110_ASTD lag110_Td lag110_temp-Td lag110_sst-Td lag115_u lag115_v
    lag115_temp lag115_sst lag115_qff lag115_rh lag115_vis lag115_ASTD lag115_Td lag115_temp-Td lag115_sst-Td std_ws
    std_ASTD std_rh reduce_fog_count BRD DAESAN DBD DJD GHD INCHEON JD
    KOEM PTDJ SD YHD YJD YPD hour month label""".split()
    #filename label
    return df.loc[train_mask, usecols], df.loc[val_mask, usecols], df.loc[test_mask, usecols]

    
@hydra.main(config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:

    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    ds = pd.read_parquet(cfg.ds_path)
    train_data, val_data, test_data = get_train_val_test_with_format(ds, pred_hour=cfg.pred_hour)
    
    label_name = f'label'
    run_type = {
        'train':Run_type.Train,
        'val':Run_type.Val, 
        'test':Run_type.Test
    }[cfg.run_type.lower()]


    if run_type== Run_type.Train:
        predictor = TabularPredictor(
            label='label', 
            path = os.path.join(os.getcwd(), 'tempdir'),
            eval_metric='f1_macro',
            sample_weight='balance_weight',
            verbosity=4
            )

        hyperparameters = {
        'GBM': {},
        'CAT': {},
        'XGB': {},
        'RF': {}
        }

        # hyperparameters = {
        # 'GBM' : {'n_estimators':np.arange(100,500,50)},
        # }
        

        results = {}
        predictor.fit(
            train_data,
            val_data, # if use bag or stack, don't specify val_data
            # ag_args_fit={'num_gpus': 1},
            # hyperparameter_tune=True,
            ag_args_fit={
                # 'num_gpus': 1,
                'num_cpus':50
                },
            # hyperparameter_tune=True,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs='auto',
            time_limit=cfg.time_limit,
        )

        results['fit_summary'] = predictor.fit_summary()  # display detailed summary of fit() process

        # predictor.class_labels_internal_map
        # predictor.delete_models(models_to_keep='best') # ensemble model contains best models
        topk_models = predictor.get_model_names()[:4]
        [ensemble_model] = predictor.fit_weighted_ensemble(topk_models)
        results['ensemble_model_names'] = topk_models

        results['pre_distill_perf'] = {
            'test': calc_metrics(
                test_data.label,
                predictor.predict(test_data,model=ensemble_model),
                binary=False),

            'val': calc_metrics(
                val_data.label,
                predictor.predict(val_data,model=ensemble_model),
                binary=False)
            }

        # kv = predictor.refit_full()
        predictor.delete_models(models_to_keep=[ensemble_model])
        distilled_model_names = predictor.distill(
            time_limit=cfg.distill_time_limit,
            # augment_method=None if cfg.distill_time_limit < 600 else 'spunge',
            augment_method = None,
            hyperparameters={'RF': {
                'max_depth':10,
                'max_features': 17,
                'n_estimators': 50
                }})

        model_to_deploy = distilled_model_names[0]
        predictor.delete_models(models_to_keep=model_to_deploy, dry_run=False)
        predictor.save()
       
        results['test_leaderboard'] = predictor.leaderboard(test_data)

        results['post_distill_perf'] = {
            'test': calc_metrics(
                test_data.label,
                predictor.predict(test_data,model=model_to_deploy),
                binary=False),
            'val': calc_metrics(
                val_data.label,
                predictor.predict(val_data,model=model_to_deploy),
                binary=False)
            }

        distill_model_path = Path(predictor.path) / 'models' / model_to_deploy /'model.pkl'
        distill_model = joblib.load(distill_model_path)

        clr = distill_model.model
        num_features = test_data.shape[1] - 1 # colums size - label
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        onx = convert_sklearn(clr, initial_types=initial_type)

        Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.model_path, "wb") as f:
            f.write(onx.SerializeToString())

        predictor.delete_models(models_to_keep = [], dry_run=False)
        del predictor        
        joblib.dump(results, cfg.train_result_path)

    # 항, 시간(input 시간), obs, pred
    sess = rt.InferenceSession(cfg.model_path)
    test_result = {}
    # yhat = predictor.predict(test_data)
    # proba = predictor.predict_proba(test_data)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
    proba = sess.run([label_name], {input_name: test_data.iloc[:,:-1].values.astype(np.float32)})[0]
    yhat = np.argmax(proba, axis=1)
    proba = pd.DataFrame(proba, index=test_data.index)
    obs_pred = test_result['obs_pred'] = test_data.label.to_frame(name='obs').assign(yhat=yhat).join(proba)
    test_result['test_metrics'] = calc_metrics(obs_pred.obs, obs_pred.yhat, binary=False)
    test_result['nia_metrics'] = calc_metrics_nia(obs_pred.obs, obs_pred.yhat, binary=False)
    joblib.dump(test_result, cfg.test_result_path)
    print(test_result['test_metrics'])

    

if __name__ =='__main__':
    main()






