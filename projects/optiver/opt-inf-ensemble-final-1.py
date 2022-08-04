#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import json
from functools import reduce
import zipfile as zf

import psutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from opt_utils import * 
from opt_fe import * 


# In[2]:


import shutil
for file in os.listdir('../input/tabnet-op-151-nb-150'): 
    print(file)
    if file.startswith('oof'):
        shutil.copy(f'../input/tabnet-op-151-nb-150/{file}', file)
    if file.startswith('tab'): 
        z = zf.ZipFile(f'{file}.zip', 'w')  
        for f in os.listdir(os.path.join('../input/tabnet-op-151-nb-150', file)): 
            shutil.copy(f'../input/tabnet-op-151-nb-150/{file}/{f}', f)
            z.write(f)
        z.close()


# In[3]:


dart_models_path = '../input/opt-train-dart-op-146'
dart_175_models_path = '../input/opt-train-dart-op-175-fold-0'
tab_models_path = '../input/opt-train-tabnet-147'
tab_181_models_path = '../input/opt-train-tabnet-181' 
tab_183_models_path = '../input/opt-train-tabnet-183' 

tab_nested_models_path = '.'
dart_nested_models_path = '../input/opt-train-dart-op-157-cons'
dart_175_nested_models_path = '../input/opt-train-dart-op-175-cons'

lgb_l2_models_path = '../input/opt-lgb-stacking-op-162'
dart_l2_models_path = '../input/opt-train-dart-stacking-op-163'



tar = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv').target
oof_dart = np.load(f'{dart_models_path}/oof_predictions.npy')
oof_dart_175 = np.load(f'{dart_175_models_path}/oof_predictions.npy')
oof_tab = np.load(f'{tab_models_path}/oof_predictions.npy')
oof_tab_181 = np.load(f'{tab_181_models_path}/oof_predictions.npy')
oof_tab_183 = np.load(f'{tab_183_models_path}/oof_predictions.npy')

oof_tab_nested = np.load(f'{tab_nested_models_path}/oof_predictions.npy')
oof_dart_nested = np.load(f'{dart_nested_models_path}/oof_predictions.npy')
oof_dart_175_nested = np.load(f'{dart_175_nested_models_path}/oof_predictions.npy')

oof_lgb_l2 = np.load(f'{lgb_l2_models_path}/oof_predictions.npy')
oof_dart_l2 = np.load(f'{dart_l2_models_path}/oof_predictions.npy')


oof_preds = [oof_dart, 
             oof_dart_175, 
             oof_tab, 
             oof_tab_181, 
             oof_tab_183, 
             oof_tab_nested, 
             oof_dart_nested, 
             oof_dart_175_nested,
             oof_lgb_l2, 
             oof_dart_l2, 
            ]

print(f'oof_dart score: {rmspe(tar, oof_dart)}')
print(f'oof_dart_175 score: {rmspe(tar, oof_dart_175)}')
print(f'oof_tab score: {rmspe(tar, oof_tab)}')
print(f'oof_tab_181 score: {rmspe(tar, oof_tab_181)}')
print(f'oof_tab_183 score: {rmspe(tar, oof_tab_183)}')

print(f'oof_tab_nested score: {rmspe(tar, oof_tab_nested)}')
print(f'oof_dart_nested score: {rmspe(tar, oof_dart_nested)}')
print(f'oof_dart_175_nested score: {rmspe(tar, oof_dart_175_nested)}')

print(f'oof_lgb_l2 score: {rmspe(tar, oof_lgb_l2)}')
print(f'oof_dart_l2 score: {rmspe(tar, oof_dart_l2)}')


def signed_power(x, p=2): 
        return np.sign(x) * np.abs(x) ** p
    
def get_ens_weights(preds):
    def minimize_arit(W):
        y_pred = sum([W[i] * preds[i] for i in range(len(preds))])
        return rmspe(tar, y_pred)

    W0 = minimize(minimize_arit, [1./len(preds)] * len(preds), options={'gtol': 1e-6, 'disp': True}).x
    print('Weights arit:',W0)
    
    def minimize_geom(W): 
        y_pred = [signed_power(preds[i], W[i]) for i in range(len(preds))]
        y_pred = reduce(lambda x, y: x * y, y_pred)
        return rmspe(tar, y_pred)

    W1 = minimize(minimize_geom, [1./len(preds)] * len(preds), options={'gtol': 1e-6, 'disp': True}).x
    print('Weights geom:',W1)
    
    return W0, W1

W0, W1 = get_ens_weights(oof_preds)

y_pred_arit = sum([W0[i] * oof_preds[i] for i in range(len(oof_preds))])
y_pred = [signed_power(oof_preds[i], W1[i]) for i in range(len(oof_preds))]
y_pred_geom = reduce(lambda x, y: x * y, y_pred)

print(f'arithmetic min/max: {y_pred_arit.min(), y_pred_arit.max()}')
print(f'geometric min/max: {y_pred_geom.min(), y_pred_geom.max()}')
print(f'ensemble score: {rmspe(tar, np.clip((y_pred_arit + y_pred_geom) / 2, 0, 1))}')


# In[4]:


def add_feat(train): 
    train['real_vol_ratio_5_10'] = (train[[f'real_vol_min_{i}' for i in range(1, 6)]].sum(axis=1) / train[[f'real_vol_min_{i}' for i in range(6, 11)]].sum(axis=1)).clip(0, 100)
    return train

with open(f'{dart_models_path}/cfg.json', 'r') as f: 
    cfg = json.load(f)
    
# cfg['path_models'] = dart_models_path
cfg['preprocessor_func'] = [p13, add_feat]
cfg["rerun"] = False
cfg["use_all"] = False
# make_submission(cfg)


# In[5]:


test = make_features(cfg['path_data_raw'], 'test', cfg['preprocessor_func'])
test_sub = test.copy()
if cfg['encode_time_cols']: 
    test = encode_cols(test, cfg["encode_time_cols"], funcs=cfg['encode_funcs'], on='time_id')
if cfg['encode_stock_cols']: 
    test = encode_cols(test, cfg["encode_stock_cols"], funcs=cfg['encode_funcs'], on='stock_id')


# In[6]:


dart_preds = evaluate(
    test, dart_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])

dart_175_preds = evaluate(
    test, dart_175_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])

cfg['use_all'] = True
dart_nested_preds = evaluate(
    test, dart_nested_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])

dart_175_nested_preds = evaluate(
    test, dart_175_nested_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])

test_tmp = test.copy()

drop_cols = [c for c in cfg['drop_cols'] if c in test.columns]
test = test.drop(drop_cols, axis = 1)

test.replace([np.inf, -np.inf], np.nan,inplace=True)
test = test.fillna(test.mean()).fillna(0)
test = test.values

tab_preds = evaluate_tabnet_models(
    test, tab_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all']
)

tab_181_preds = evaluate_tabnet_models(
    test, tab_181_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all']
)

tab_183_preds = evaluate_tabnet_models(
    test, tab_183_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all']
)

tab_nested_preds = evaluate_tabnet_models(
    test, tab_nested_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all']
)

test = test_tmp
test['dart_preds'] = dart_nested_preds
test['tab_preds'] = tab_nested_preds

test = encode_cols(test, 
                  ['dart_preds', 'tab_preds'], 
                  funcs=cfg['encode_funcs'], 
                  on='time_id')
test = encode_cols(test, 
                  ['dart_preds', 'tab_preds'], 
                  funcs=cfg['encode_funcs'])

lgb_l2_preds = evaluate(
    test, lgb_l2_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])

dart_l2_preds = evaluate(
    test, dart_l2_models_path, cfg['prefix'], 
    cfg['drop_cols'], cfg['rerun'], cfg['use_all'])


# In[7]:


preds = [dart_preds, 
         dart_175_preds, 
         tab_preds, 
         tab_181_preds, 
         tab_183_preds, 
         tab_nested_preds, 
         dart_nested_preds, 
         dart_175_nested_preds, 
         lgb_l2_preds, 
         dart_l2_preds]
y_pred = [signed_power(preds[i], W1[i]) for i in range(len(preds))]
y_pred = reduce(lambda x, y: x * y, y_pred)
y_pred += sum([W0[i] * preds[i] for i in range(len(preds))])
y_pred /= 2
test_sub['target'] = y_pred.clip(0, 1)
test_sub[['row_id', 'target']].to_csv('submission.csv',index = False)


# In[8]:


df = pd.read_csv('submission.csv')
df


# In[9]:


df.loc[0, :].isna().sum()

