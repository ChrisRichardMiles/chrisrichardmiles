import os ########## This is to show you how to track everything with 
import time ######## neptune.ai. It is even set up so that you track 
import json ######## all the awesome stuff during training, but without 
import pandas as pd # using up your neptune tracking time. To use neptune,
import numpy as np # you need a free account, the api_token from neptune.ai
import lightgbm as lgb # put into kaggle secrets in the "Add-ons" dropdown 
# from lightgbm.callback import # at the top. 
from opt_utils import rmspe
# opt_utils is my own module 
os.system('pip install neptune-client')
os.system('pip install neptune-lightgbm')
import neptune.new as neptune
from neptune.new.integrations.lightgbm import create_booster_summary
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
NEPTUNE_API_TOKEN = user_secrets.get_secret("NEPTUNE_API_TOKEN")

cfg = {
    "path_features": '../input/generate-train-features-script/p3_train.pkl', # Used in train mode
    "path_models": '',
    "path_data_raw": '../input/optiver-realized-volatility-prediction/',
    "neptune_project": 'chrisrichardmiles/optiver',
    "drop_cols": ['row_id', 'time_id', 'stock_id', 'target'], 
    "prefix": '',
    "script_name": 'Lightgbm with neptune.ai integration_v0',
    "rerun": True, 
    "neptune_description": '', 
    "neptune_run_name": '',
    "lgb_params": {
        # https://lightgbm.readthedocs.io/en/latest/index.html
        "boosting_type": "gbdt",
        "objective": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 255,
        "feature_fraction": 0.8,
        "bagging_fraction": .5, # Select bagging_fraction of rows every bagging_freq of iterations.
        "bagging_freq": 1,      # This speeds up training and underfits. Need both set to do anything.
        "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
        "seed": 42,
        "verbose": -1, 
    },
    "no_log": ["no_log", "path_models", "path_data_raw", "neptune_project"]
}
with open('cfg.json', 'w') as f: 
    json.dump(cfg, f)

def main(): 
    train = pd.read_pickle(cfg['path_features'])
    drop_cols = [c for c in cfg['drop_cols'] if c in train.columns]
    x = train.drop(drop_cols, axis = 1)
    y = train['target']
    
    oof_predictions = np.zeros(x.shape[0]) # Create out of folds array
    scores = [] # Keep track of scores for each fold and all oof at the end
    best_iterations = []
    training_best_scores = []
    valid_best_scores = []
    best_score_diffs = []
    dict_eval_logs = [] # For experimentation tracking
    booster_summaries = [] # For experimentation tracking
    
    for fold in range(5): # Your features need to have a fold column 
        trn_ind = x.fold != fold
        val_ind = x.fold == fold
        
        print(f'Training fold {fold}')
        x_train, x_val = x[trn_ind].drop('fold', axis=1), x[val_ind].drop('fold', axis=1)
        y_train, y_val = y[trn_ind], y[val_ind]
        
        train_weights = 1 / np.square(y_train) # Root mean squared percentage error weights
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights)
        val_dataset = lgb.Dataset(x_val, y_val, weight=val_weights, reference=train_dataset)
        
        dict_eval_log = {}
        model = lgb.train(params = cfg['lgb_params'], 
                          train_set = train_dataset, 
                          valid_sets = [val_dataset, train_dataset], 
                          valid_names = ['valid', 'train'], 
                          feval = feval_rmspe,
                          callbacks=[record_evaluation(dict_eval_log)],
                          verbose_eval = 50)
        
        model.save_model(os.path.join(cfg['path_models'], f'{cfg["prefix"]}lgb_fold_{fold}.txt'))
        y_pred = model.predict(x_val)
        oof_predictions[val_ind] = y_pred
        scores.append(round(rmspe(y_val, y_pred), 3))
        
        booster_summary = create_booster_summary(
            booster=model,
            log_importances=True,
            max_num_features=25,
            log_trees_as_dataframe=False, 
            log_pickled_booster=True, 
            y_true=y_val, 
            y_pred=y_pred, 
        )
        train_score = model.best_score['train']['RMSPE']
        valid_score = model.best_score['valid']['RMSPE']
        best_iterations.append(model.best_iteration)
        training_best_scores.append(round(train_score, 3))
        valid_best_scores.append(round(valid_score, 3))
        best_score_diffs.append(round(valid_score - train_score, 3))
        
        booster_summaries.append(booster_summary)
        dict_eval_logs.append(dict_eval_log)
        del booster_summary, dict_eval_log
    
    rmspe_score = round(rmspe(y, oof_predictions), 3)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    print(f'Our cv fold scores are {scores}')
    np.save('oof_predictions', oof_predictions)
    
    run = neptune.init(
            project=cfg['neptune_project'],
            api_token=NEPTUNE_API_TOKEN,
            name=cfg['neptune_run_name'],    
            description=cfg['neptune_description'],
            tags=[cfg['path_features'], cfg['prefix']],
            source_files=['cfg.json'],
    )
    run['feat_id'] = os.path.split(cfg['path_features'])[1].split('_')[0]
    run['cfg'] = cfg
    run['RMSPE'] = rmspe_score
    run['RMSPE_oof_scores'] = scores
    run['best_iterations'] = best_iterations
    best_iterations_mean = round(np.mean(best_iterations), 3)
    run['best_iterations_mean'] = best_iterations_mean
    run['training_best_scores'] = training_best_scores
    run['valid_best_scores'] = valid_best_scores
    run['best_score_diffs'] = best_score_diffs
    run['best_score_diffs_mean'] = round(np.mean(best_score_diffs), 3)
    
    # Logs for each folds model
    for fold in range(5):
        run[f'lgbm_summaries/fold_{fold}'] = booster_summaries[fold]
        dict_eval_log = dict_eval_logs[fold]
        for valid_set, odict in dict_eval_log.items():
            for metric, log in odict.items():
                for val in log:
                    run[f'eval_logs/{fold}_{valid_set}_{metric}'].log(val)
    run.stop()
    
    if cfg['rerun']: 
        print(f'retraining model with all data for {best_iterations_mean} iterations')
        x_train = x.drop('fold', axis=1)
        y_train = y
        
        train_weights = 1 / np.square(y_train) # Root mean squared percentage error weights
        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights)
        
        params = cfg['lgb_params'].copy()
        params['n_estimators'] = int(best_iterations_mean) # lgbm needs int here
        params['early_stopping_rounds'] = 0 # No valid set to stop with
        model = lgb.train(params = params, 
                          train_set = train_dataset)
        model.save_model(os.path.join(cfg['path_models'], f'{cfg["prefix"]}lgb.txt'))
    
if __name__ == '__main__': 
    main()

# %% [code]
