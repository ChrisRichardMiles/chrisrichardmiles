# %%
import os
import time
import sys
import json
from itertools import product
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# os.system('pip install ../input/pytorchtabnet/pytorch_tabnet-3.1.1-py3-none-any.whl')
from pytorch_tabnet.tab_model import TabNetRegressor

################## Making features and submissions ###################
def read_train_or_test(DATA_RAW, train_or_test): 
    if train_or_test == 'test': 
        return pd.read_csv(os.path.join(DATA_RAW, 'test.csv')).set_index('row_id')
    else: 
        return make_folds(DATA_RAW).set_index('row_id')
    
def make_features(DATA_RAW, train_or_test, preprocessor_func, testing_func=False):
    if type(preprocessor_func) == list: 
        df_fe = make_features(DATA_RAW, train_or_test, preprocessor_func[0], testing_func)
        for f in preprocessor_func[1:]: 
            df_fe = f(df_fe)
            return df_fe
    df_fe = read_train_or_test(DATA_RAW, train_or_test)
    global func
    def func(x): return preprocessor_func(DATA_RAW, x, train_or_test)
    if testing_func: 
        df_agg = pd.concat(pool_func(func, df_fe.stock_id.unique()[:4]))
    else: 
        df_agg = pd.concat(pool_func(func, df_fe.stock_id.unique()))
    return df_fe.join(df_agg).reset_index()

def make_train(cfg): 
    train = make_features(cfg['path_data_raw'], 'train', cfg['preprocessor_func'], cfg['testing_func'])
    train.to_pickle(f'{cfg["preprocessor_func"].__name__}_train.pkl')
    
def make_submission(cfg): 
    test = make_features(cfg['path_data_raw'], 'test', cfg['preprocessor_func'])
    if cfg['encode_time_cols']: 
        test = encode_cols(test, cfg["encode_time_cols"], funcs=cfg['encode_funcs'], on='time_id')
    if cfg['encode_stock_cols']: 
        test = encode_cols(test, cfg["encode_stock_cols"], funcs=cfg['encode_funcs'], on='stock_id')
#         train = pd.read_pickle(cfg['path_enc_train'])
#         enc_cols = [c for c in train.columns if c.startswith('stock_id_')]
#         stock = train.groupby('stock_id')[enc_cols].first()
#         test = test.set_index('stock_id').join(stock).reset_index()
    if type(cfg['path_models']) == str: 
        cfg['path_models'] = [cfg['path_models']]
        cfg['masks'] = [[True] * len(test)]
    
    for mask, path in zip(cfg['masks'], cfg['path_models']): 
        test.loc[mask, 'target'] = evaluate(
            test, path, cfg['prefix'], cfg['drop_cols'], cfg['rerun'], cfg['use_all']
        )[mask]
    
    test[['row_id', 'target']].to_csv('submission.csv',index = False)
    
def evaluate(test, path_models='', prefix='', drop_cols=[], rerun=False, use_all=False): 
    drop_cols = [c for c in drop_cols if c in test.columns]
    x_test = test.drop(drop_cols, axis = 1)
    test_predictions = np.zeros(x_test.shape[0]) # Create test array to store predictions
    if use_all: 
        n_models = 0
        for file in os.listdir(path_models): 
            if 'lgb' in file and file.startswith(prefix): 
                model = lgb.Booster(model_file=os.path.join(path_models, file))
                test_predictions += model.predict(x_test)
                n_models += 1
        return test_predictions / n_models
    if rerun: 
        model = lgb.Booster(model_file=os.path.join(path_models, f'{prefix}rerun_lgb_{best_iter}.txt'))
        return model.predict(x_test) 
    for fold in range(5):
        model = lgb.Booster(model_file=os.path.join(path_models, f'{prefix}lgb_fold_{fold}.txt'))
        test_predictions += model.predict(x_test) / 5
    return test_predictions

def evaluate_tabnet_models(x_test, path_models='', prefix='', drop_cols=[], rerun=False, use_all=False): 
    test_predictions = np.zeros(x_test.shape[0]) # Create test array to store predictions
    model = TabNetRegressor()
    if use_all: 
        n_models = 0
        for file in os.listdir(path_models): 
            if 'tab' in file and file.startswith(prefix): 
                model.load_model(os.path.join(path_models, file))
                test_predictions += model.predict(x_test).flatten()
                n_models += 1
        return test_predictions / n_models
    if rerun: 
        model.load_model(os.path.join(path_models, f'{prefix}rerun_tab_{best_iter}.zip'))
        return model.predict(x_test) .flatten()
    for fold in range(5):
        model.load_model(os.path.join(path_models, f'{prefix}tab_fold_{fold}.zip'))
        test_predictions += model.predict(x_test).flatten() / 5
    return test_predictions



######################### Helper functions ###################
def encode_cols(df, cols, funcs=['mean', 'std'], on='stock_id', shake=False, shake_std=False): 
    if not cols or not funcs: return df
    tmp = df.groupby(on)[cols].agg(funcs)
    tmp.columns = ['_'.join(c) for c in tmp.columns]
    tmp = tmp.add_prefix(on + '_')
    tmp =  df.join(tmp, on=on)
    if shake: 
        for c in tmp.columns: 
            if c.startswith(on + '_'):
                c_mean = tmp[c].mean()
                tmp[c] = tmp[c] + np.random.normal(scale=abs(c_mean * shake), size=len(tmp))
                
    if shake_std: 
        for c in tmp.columns: 
            if c.startswith(on + '_'):
                c_std = tmp[c].std()
                tmp[c] = tmp[c] + np.random.normal(scale=c_std * shake_std, size=len(tmp))
                
    return tmp 

def get_dumb_features(lgb_booster, importance_type='both', prefix='dummy'):
    if importance_type == 'both':
        s = get_dumb_features(lgb_booster, 'split', prefix)
        g = get_dumb_features(lgb_booster, 'gain', prefix)
        return list(set(s).union(set(g)))
    fi = lgb_booster.feature_importance(importance_type)
    fi = pd.DataFrame(list(zip(lgb_booster.feature_name(), fi))).set_index(0)[1]
    dummy_importance = []
    for col in fi.index: 
        if col.startswith(prefix): 
            dummy_importance.append(fi[col])
    if not dummy_importance: return [] 
    mx = max(dummy_importance)
    return dummy_importance + fi[fi < mx].index.tolist()

def get_top_features(lgb_booster, importance_type='both', n_features=25): 
    if importance_type == 'both':
        return list(set(get_top_features(lgb_booster, 'split', n_features)).union(
            set(get_top_features(lgb_booster, 'gain', n_features))))
    f_imp = lgb_booster.feature_importance(importance_type)
    f_name = lgb_booster.feature_name()
    return [name for _, name in sorted(zip(f_imp, f_name), reverse=True)[:n_features]]

def add_percentile(df, col): 
    """"""
    if type(col) == list: 
        for c in col: df = add_percentile(df, c)
    else: 
        df[col + '_percentile'] = (df[col].rank(pct=True) * 100).astype(int)
    return df

def pool_func(function, input_list: list, verbose=False, n_cpu=99):
    """Uses the Pool function from the package 'multiprocessing'
    to run `function` over the list `input_list`.  The `function`
    should only take """

    n_cpu = min(n_cpu, cpu_count())
    if verbose:
        print('#############################################')
        print('Pooling function: ')
        if hasattr(function, '__name__'):
            print(function.__name__)
        print(f'{n_cpu} of {cpu_count()} cpus used')
        print('Number of function calls: ', len(input_list))

    start = time.time()
    pool = Pool(n_cpu)
    res = pool.map(function, input_list)
    pool.close()
    pool.join()

    if verbose:
        print('Time taken:',
              round((time.time() - start) / 60, 2),
              'minutes')
    return res if res else []

def fix_offsets(data_df):
    
    offsets = data_df.groupby(['time_id']).agg({'seconds_in_bucket':'min'})
    offsets.columns = ['offset']
    data_df = data_df.join(offsets, on='time_id')
    data_df.seconds_in_bucket = data_df.seconds_in_bucket - data_df.offset
    
    return data_df

def ffill(data_df):
    """To fill missing seconds in book. From user https://www.kaggle.com/slawekbiel"""
    data_df=data_df.set_index(['time_id', 'seconds_in_bucket'])
    data_df = data_df.reindex(pd.MultiIndex.from_product([data_df.index.levels[0], np.arange(0,600)], names = ['time_id', 'seconds_in_bucket']), method='ffill')
    return data_df.reset_index()

def load_bt(DATA_RAW, stock_id, train_or_test='train', book_only=False, trade_only=False, add_stock_id=False, reindex_path=None):
    """Loads the book and trade data into a single dataframe."""
    
    book = pd.read_parquet(os.path.join(DATA_RAW, f'book_{train_or_test}.parquet/stock_id={stock_id}'))
    book = fix_offsets(book)
    book = ffill(book)
    if add_stock_id: book['stock_id'] = stock_id
    if book_only: return book
    trade =  pd.read_parquet(os.path.join(DATA_RAW, f'trade_{train_or_test}.parquet/stock_id={stock_id}'))
    if add_stock_id: trade['stock_id'] = stock_id
    if trade_only: return trade
    return book.merge(trade, on=['time_id', 'seconds_in_bucket'], how='outer')

def add_wap(df):
    """Adds the weighted average price to a book df."""
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1']+ df['ask_size1'])
    return df

def add_wap_2(df):
    """Adds the weighted average price to a book df."""
    df['wap_2'] = (df['bid_price_mean'] * df['sum_ask'] + df['ask_price_mean'] * df['sum_bid']) / (df['sum_bid']+ df['sum_ask'])
    return df

def count_unique(x): return len(np.unique(x))

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def vol_log_return(x): 
    return realized_volatility(log_return(x))

def mean_decay(x, decay=.9, step=-1, axis=0): 
    """Returns sum with exponential decay, step = -1
    for the right end of the array to matter the most."""
    weights = np.power(decay, np.arange(x.shape[axis])[::step]).astype(np.float32)
    return np.sum(weights * x, axis=axis) / weights.sum()

def get_mean_decay(decay=.9, step=-1): 
    return lambda x: mean_decay(x, decay, step)

def first(x): return x.values[0]
def last(x): return x.values[-1]
def max_sub_min(x): return np.max(x) - np.min(x)

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def make_folds(DATA_RAW, DATA_INTERIM=None):
    train = pd.read_csv(os.path.join(DATA_RAW, 'train.csv'))
    g = GroupKFold()
    for fold, (_, test_idx) in enumerate(g.split(train, groups=train['time_id'])):
        train.loc[test_idx, 'fold'] = fold
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    if not DATA_INTERIM: return train
    train.to_csv(os.path.join(DATA_INTERIM, 'folds.csv'), index=False)
    
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


################### Preprocessing functions ##################

def p1(stock_id, train_or_test):
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['minute'] = np.ceil((df.seconds_in_bucket + .1) / 120).astype(int)

    df_agg = pd.DataFrame()
    df_agg['real_vol'] = df.groupby('time_id')['log_return'].agg(realized_volatility)
    df_agg['is_pos_return_sum'] = df.groupby('time_id')['is_pos_return'].agg(sum).astype(np.float32)
    df_agg['is_neg_return_sum'] = df.groupby('time_id')['is_neg_return'].agg(sum).astype(np.float32)

    ############ Realized volume for each minute ############
    for m in range(1, 6): 
        mask = (df.seconds_in_bucket > 120 * m - 120) & (df.seconds_in_bucket < 120 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)
#         for minute in df.minute.unique():
#             tmp = df[df['minute'] == minute]
#             df_agg[f'real_vol_min_{minute}'] = tmp.groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in df.minute.unique()]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] =  mean_decay(x, decay, step)
    df_agg['end_beg_decay_ratio'] = df_agg['real_vol_mean_decay_0.85_-1'] / df_agg['real_vol_mean_decay_0.85_1']

    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg

def make_fe1(DATA_RAW,train_or_test):
    if train_or_test == 'test': 
        df_fe = pd.read_csv(os.path.join(DATA_RAW, 'test.csv')).set_index('row_id')
    else: 
        df_fe = make_folds(DATA_RAW).set_index('row_id')
    func = p1_train if train_or_test == 'train' else p1_test
    df_all = pd.concat(pool_func(func, df_fe.stock_id.unique()))

    ################### Merge with train or test file ###############
    return df_fe.join(df_all.set_index('row_id')).reset_index()


def p3(DATA_RAW, stock_id, train_or_test):
    
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']

    agg_dict = {
        'log_return': [realized_volatility, 'count', np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'bid_ask_ratio': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],

    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip'}
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]

    ############ Realized volume for each minute ############
    for m in range(1, 6): 
        mask = (df.seconds_in_bucket > 120 * m - 120) & (df.seconds_in_bucket < 120 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 6)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] =  mean_decay(x, decay, step, axis=1)
    df_agg['end_beg_decay_ratio'] = df_agg['real_vol_mean_decay_0.85_-1'] / df_agg['real_vol_mean_decay_0.85_1']
    
    df_agg = df_agg.astype('float32')
    df_agg['no_book'] = (df_agg['log_return_count'] == 0).astype(int)
    df_agg['no_book'] = df_agg['no_book'].astype('category')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')

def p4(DATA_RAW, stock_id, train_or_test):
    """Adding more mean decay based on minute splits, since they ranked 
    huge on gain importance and 
    including more stats around spread and size since they 
    ranked high on split importance."""
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']

    agg_dict = {
        'log_return': [realized_volatility, 'count', np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'bid_ask_ratio': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],


    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    ############ Realized volume for each minute ############
    for m in range(1, 6): 
        mask = (df.seconds_in_bucket >= 120 * m - 120) & (df.seconds_in_bucket < 120 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 6)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] =  mean_decay(x, decay, step, axis=1)
#     df_agg['end_beg_decay_ratio'] = df_agg['real_vol_mean_decay_0.85_-1'] / df_agg['real_vol_mean_decay_0.85_1'] # replaced by next code
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
    
    df_agg = df_agg.astype('float32')
    df_agg['no_book'] = (df_agg['log_return_count'] == 0).astype(int)
    df_agg['no_book'] = df_agg['no_book'].astype('category')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')




def p5(DATA_RAW, stock_id, train_or_test):
    """Same as p4 except Im goin to use 10 minutes 
    instead of 5."""
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']

    agg_dict = {
        'log_return': [realized_volatility, 'count', np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'bid_ask_ratio': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],


    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] =  mean_decay(x, decay, step, axis=1)
#     df_agg['end_beg_decay_ratio'] = df_agg['real_vol_mean_decay_0.85_-1'] / df_agg['real_vol_mean_decay_0.85_1'] # replaced by next code
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
    
    df_agg = df_agg.astype('float32')
    df_agg['no_book'] = (df_agg['log_return_count'] == 0).astype(int)
    df_agg['no_book'] = df_agg['no_book'].astype('category')
    
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')

###################################################################################
# def first(x): return x.values[0]
# def last(x): return x.values[-1]

def p6(DATA_RAW, stock_id, train_or_test):
    """Keeping all from p5. 
    Adding price min, max, std, max - min, last, first, last - first"""
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']

    agg_dict = {
        'log_return': [realized_volatility, 'count', np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [np.std, np.mean, np.min, np.max, first, last, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'bid_ask_ratio': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [np.mean, np.sum, np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],


    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] =  mean_decay(x, decay, step, axis=1)
#     df_agg['end_beg_decay_ratio'] = df_agg['real_vol_mean_decay_0.85_-1'] / df_agg['real_vol_mean_decay_0.85_1'] # replaced by next code
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
            
    df_agg['wap_max-min'] = df_agg['wap_amax'] - df_agg['wap_amin']
    df_agg['wap_last-first'] = df_agg['wap_last'] - df_agg['wap_first']

    df_agg = df_agg.astype('float32')
    df_agg['no_book'] = (df_agg['log_return_count'] == 0).astype(int)
    df_agg['no_book'] = df_agg['no_book'].astype('category')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')

#####################################################################333
#####################################################################333
#####################################################################333
# def max_sub_min(x): np.max(x) - np.min(x)

def p7(DATA_RAW, stock_id, train_or_test):
    """Keeping all from p7. 
    Making last - first absolute value 
    Making sure all aggregations include std and max_sub_min
    Adding original bid/ask ratio and changing name of bid/ask to total_bid_ask
    Adding sum_ask_sub_sum_bid
    """
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df.bid_size1 / df.ask_size1
    df['total_bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']
    df['sum_ask_sub_sum_bid'] = df['sum_ask'] - df['sum_bid']

    agg_dict = {
        'log_return': [max_sub_min, np.std, realized_volatility, 'count', get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [max_sub_min, np.std, np.mean, np.min, np.max, first, last, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'total_bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask_sub_sum_bid': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] = mean_decay(x, decay, step, axis=1)
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
            
    df_agg['wap_last-first'] = (df_agg['wap_last'] - df_agg['wap_first']).abs()

    df_agg = df_agg.astype('float32')
    df_agg['no_book'] = (df_agg['log_return_count'] == 0).astype(int)
    df_agg['no_book'] = df_agg['no_book'].astype('category')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')

#################################################################################
#################################################################################



def p8(DATA_RAW, stock_id, train_or_test):
    """First features with ffill implemented. p7 had no cv improvement, but it
    widened the train, val score, so soon I will be looking to cull  
    features, keeping only those features that already show up high on lgbm 
    gain or split importance. The new features are realized volatility on 
    almost everything. Keeping other aggregations sparce for now to be simple.
    """
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['sum_bid'] = (df.bid_size1 + df.bid_size2)
    df['sum_ask'] = (df.ask_size1 + df.ask_size2)
    df['bid_ask_ratio'] = df.bid_size1 / df.ask_size1
    df['total_bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask']
    df['sum_ask_sub_sum_bid'] = df['sum_ask'] - df['sum_bid']
    df['price_wap_diff'] = df['price'] - df['wap']

    agg_dict = {
        'log_return': [max_sub_min, np.std, realized_volatility, 'count', get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'wap': [max_sub_min, np.std, np.mean, np.min, np.max, first, last, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'price_wap_diff': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_2': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'total_bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask_sub_sum_bid': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread_2_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
    }
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] = mean_decay(x, decay, step, axis=1)
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
            
    df_agg['wap_last-first'] = (df_agg['wap_last'] - df_agg['wap_first']).abs()
    
    df_agg = df_agg.join(df.groupby(['time_id'])['wap', 'bid_price1', 'ask_size1', 'ask_price1', 'bid_size1', 
                            'sum_bid', 'sum_ask', 'spread_pct', 'spread_2_pct', 'spread', 'spread_2', 
                            'bid_ask_ratio', 'total_bid_ask_ratio', 'sum_bid_ask', 'sum_ask_sub_sum_bid', 
                            'size', 'price'
                           ].agg(vol_log_return).add_suffix('_real_vol'))
    
    df_agg['dummy1'] = np.random.normal(size=(len(df_agg)))
    df_agg['dummy2'] = np.random.normal(size=(len(df_agg)))
    df_agg['dummy3'] = np.random.normal(size=(len(df_agg)))

    df_agg = df_agg.astype('float32')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')

# %%

# %% [code] {"jupyter": {"outputs_hidden": false}}
