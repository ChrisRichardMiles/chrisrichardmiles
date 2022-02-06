# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['mkdirs_data', 'cp_tree', 'make_unique', 'make_unique_path', 'save_file', 'load_file', 'get_file_cols_dict',
           'fe_dict', 'load_features', 'pool_func', 'reduce_mem_usage', 'merge_by_concat', 'get_memory_usage',
           'sizeof_fmt', 'time_taken']

# Cell
import os
import shutil
import json
import gc
import sys
import time
import logging
from itertools import chain
from typing import Union
from importlib import import_module
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import psutil
from fastcore.script import call_parse, Param

# Cell
@call_parse
def mkdirs_data(data_dir_name: Param('Name of data folder', str)='data') -> None:
    """Initializes the data directory structure"""
    os.makedirs(f'{data_dir_name}/raw', exist_ok=True)
    os.makedirs(f'{data_dir_name}/interim', exist_ok=True)
    os.makedirs(f'{data_dir_name}/features', exist_ok=True)
    os.makedirs(f'{data_dir_name}/models', exist_ok=True)

@call_parse
def cp_tree(dir1: Param('path to directory to copy', str),
            dir2: Param('path to endpoint', str)):
    shutil.copytree(dir1, dir2, dirs_exist_ok=True)

# Cell
def make_unique(name: str, names: "list or set or dict") -> str:
    """Returns name with (x)_ prefix if `name` already in `names`.
    This is useful when you want to make sure you don't save over
    existing data with the same key `name`.
    """
    if name in names:
        x = 1
        while f'({x})_' + name in names: x += 1
        name = f'({x})_' + name
    return name

# Cell
def make_unique_path(path):
    """Returns path with prefix '(n)_' before the last element in
    path if it is a duplicate.
    """
    pre_path, file_name = os.path.split(path)
    file_name = make_unique(file_name, os.listdir(pre_path or '.'))
    return os.path.join(pre_path, file_name)

# Cell
def save_file(df: pd.DataFrame,
              path: str,
              usecols: list=None,
              save_index: bool=False,
              save_dtypes: bool=True,
              pickle: bool=False) -> None:
    """Saves `df` to `path` with dtypes as top column if `save_dtypes`
    is set to True. Load a files in this structure with `load_file`
    """
    if pickle:
        usecols = usecols if usecols else list(df)
        path_dir = os.path.split(path)[0] if path.endswith('.csv') else path # For M5 project maintenence
        for col in list(df):
            df[[col]].to_pickle(os.path.join(path_dir, col + '.pkl'))
        return

    path = make_unique_path(path)
    if save_dtypes:
        df_tmp = df.iloc[[0], :]
        if usecols: df_tmp = df_tmp.loc[:, usecols]
        if save_index:
            df_tmp.reset_index(inplace=True)
        df_dtypes = df_tmp.dtypes.to_frame().T
        df_dtypes.to_csv(path, index=False)
        df.to_csv(path, mode='a', index=save_index, header=False,
                  columns=usecols)
    else:
        df.to_csv(path, index=save_index, columns=usecols)

def load_file(path: str, load_dtypes=True, usecols: list=None) -> pd.DataFrame:
    """Loads a file into a DataFrame from `path` with dtypes
    taken from the top column if `load_dtypes` is set to True.
    Loads a files in the structure created with `save_file`.
    """
    if path.endswith('pkl'):
        df = pd.read_pickle(path)
        return df[usecols] if usecols else df

    if load_dtypes:
        dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
        return pd.read_csv(path, skiprows=[1], dtype=dtypes, usecols=usecols)
    else:
        return pd.read_csv(path, usecols=usecols)

# Cell
def get_file_cols_dict(path: str='.',
                       path_json: str='',
                       ignore_cols: list=['index']):
    """Explores `path` and returns a dictionary of file names and their columns
    for each file in `path`. Only file names that end with
    '.csv' and '.pkl' will be considered. Pickle file names
    will go in the 'pickles' key of the returned dictionary.
    Csv files will see their file name saved as a key with
    a list of their column names saved as the corresponding
    value.
    """

    d = {}
    for file in sorted(os.listdir(path)):
        if file.endswith('.csv'):
            cols = pd.read_csv(os.path.join(path, file), nrows=0).columns.tolist()
            d[file] = [c for c in cols if c not in ignore_cols]
        if file.endswith('.pkl'):
            d.setdefault('pickles', []).append(file)
    if path_json:
        with open(path_json, 'w') as path_json:
            json.dump(d, path_json, indent=0)
    return d

@call_parse
def fe_dict(path: Param('path to directory with files', str)='data/features',
            path_json: Param('path to json for saving dict', str)='fe_dict.json'):
    get_file_cols_dict(path, path_json)

# Cell
def load_features(path_features: Union[list, str],
                  dict_features: Union[dict, str]=None,
                  shift_index: int=0,
                  reindex_with: "list like"=None,
                  shift_prefix: Union[str, bool]='shift',
                  load_dtypes: bool=True,
                  features: list=None,
                  pickle: bool=True) -> pd.DataFrame:
    """Loads the features selected in `dict_features` into a dataframe.
    `dict_features` Must be a module that is located in the working
    directory.

    Parameters
    ----------
    path_features: Union[list, str]
        path to the folder that holds the features files

    dict_features: Union[dict, str]
        dict or path to the json that holds the feature dictionary. Set this
        parameter to None if you want to load all csv files, optionally
        filtered by `features` list.

    shift_index: int=0
        used to shift columns of files starting with `shift_prefix` when training
        for prediction periods past day 1.

    shift_prefix: Union[str, bool]='shift'
        The prefix of files that should have their index shifted for
        proper lag alignment in time series prediction.
        Set this to the boolean True to shift index of all files.

    reindex_with: "list like"=None
        Use anything that works with df.reindex(reindex_with). This is used when you
        only need rows for a subset of the orginal data.

    load_dtype: bool=True
        This will use the first row for dtypes

    features: list=None
        An explicit list of features that you want. Only these will be loaded
        if provided.
    """

    if type(path_features) == list:
        df = pd.DataFrame()
        for pf in path_features:
            args = (pf, dict_features, shift_index, reindex_with, shift_prefix, load_dtypes, features, load_all)
            df = pd.concat([df, load_features(*args)], axis=1)
        return df if df else None

    if type(dict_features) == str:
        with open(dict_features, 'r') as file:
            dict_features = json.load(file)

    if pickle: # Added for faster feature loading
        if not features:
            features = list(chain(*dict_features.values()))
        files = [f for f in features if f in path_features]
        df = pd.DataFrame()
        for file in files:
            path = os.path.join(path_features, file)
            df_tmp = pd.read_pickle(file)
            if file.startswith(shift_prefix) and shift_index:
                df_tmp.index = df_tmp.index + shift_index
            if type(reindex_with) != None:
                df_tmp = df_tmp.reindex(reindex_with)
            df = pd.concat([df, df_tmp])

    if not dict_features:
        dict_features = get_file_cols_dict(path_features)

    # Filter dict to keep only keys that are in `path_features`
    dict_features = {k: v for k, v in dict_features.items() if k in os.listdir(path_features)}

    dfs = []
    for file, f_list in dict_features.items():
        if features: f_list = [f for f in f_list if f in features]
        path = os.path.join(path_features, file)
        df = load_file(path, load_dtypes, f_list)
        if 'index' in df.columns: df.set_index('index', inplace=True)
        if file.startswith(shift_prefix) and shift_index:
            df.index = df.index + shift_index
        if type(reindex_with) != None: df = df.reindex(reindex_with)
        dfs.append(df)
    if not dfs:
        logging.info("No data was loaded")
        print("No data was loaded")
    return pd.concat(dfs, axis=1) if dfs else None

# Cell
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

# Cell
def reduce_mem_usage(df, verbose=True):
    """Converts numeric columns to smallest datatype that preserves information"""

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Cell
def merge_by_concat(df1, df2, merge_on):
    if type(merge_on) == str: merge_on = [merge_on]
    merged_df = df1[merge_on]
    merged_df = merged_df.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_df) if col not in merge_on]
    df1 = pd.concat([df1, merged_df[new_columns]], axis=1)
    return df1

# Cell
def get_memory_usage():
    """Returns RAM usage in gigabytes

    Explanation of code
    -------------------
    # getpid: gets the process id number.
    # psutil.process gets that process with a certain pid.
    # .memory_info() describes notebook memory usage.
    # [0] gets the rss resident state size of (process I think) in bytes.
    # /2.**30 converts output from bytes to gigabytes
    """
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2)

def sizeof_fmt(num, suffix='B'):
    """Reformats `num`, which is num bytes"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# Cell
def time_taken(start_time: float=0, time_elapsed: float=None):
    """Returns a string with the time elapsed from `start_time`
    in a nice format. If `time_elapsed` is provided, we ignore
    the start time.

    `start_time` should come from by calling the time module:
    start_time = time.time()
    """

    import time
    if not time_elapsed:
        time_elapsed = int(time.time() - start_time)
    else:
        time_elapsed = int(time_elapsed)

    m, s = divmod(time_elapsed, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d: return f'Time taken: {d} days {h} hours {m} minutes {s} seconds'
    if h: return f'Time taken: {h} hours {m} minutes {s} seconds'
    if m: return f'Time taken: {m} minutes {s} seconds'
    if s: return f'Time taken: {s} seconds'
    return 'Time taken: 0 seconds'