# %% [code]
import numpy as np 
import pandas as pd 
from opt_utils import * 

from functools import reduce
def count_unique(series):
    return len(np.unique(series))


def rv_99(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.99, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_95(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.95, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_90(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.90, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_85(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.85, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_80(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.80, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_75(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.75, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_70(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.70, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_65(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.65, np.arange(n)[::-1]).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))
 
    
def rv_99_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.99, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_95_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.95, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_90_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.90, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_85_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.85, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_80_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.80, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_75_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.75, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_70_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.70, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))

def rv_65_flip(series_log_return): 
    n = len(series_log_return)
    weights = np.power(.65, np.arange(n)).astype(np.float32)
    return np.sqrt(np.sum(weights * series_log_return**2))


def p5_v2(DATA_RAW, stock_id, train_or_test):
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
        'log_return': [realized_volatility, 'count', np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1),
                       rv_99, rv_95, rv_90, rv_85, rv_80, rv_75, rv_70, rv_65, 
                       rv_99_flip, rv_95_flip, rv_90_flip, rv_85_flip, 
                       rv_80_flip, rv_75_flip, rv_70_flip, rv_65_flip],
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

####################################################################################
####################################################################################

# from functools import reduce
# def count_unique(series):
#     return len(np.unique(series))
def tendency(price, vol):    
        df_diff = np.diff(price)
        val = (df_diff/price[1:])*100
        power = np.sum(val*vol[1:])
        return(power)

def p10(DATA_RAW, stock_id, train_or_test):
    """Adding to p8.
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
    df['sum_ask_sub_sum_bid_scaled'] = (df['sum_ask'] - df['sum_bid']) / min(df['sum_ask'] - df['sum_bid'])
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
        'sum_ask_sub_sum_bid_scaled': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_bid_ask': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread_2_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'bid_price1': [vol_log_return], 
        'ask_size1': [vol_log_return], 
        'ask_price1': [vol_log_return], 
        'bid_size1': [vol_log_return], 
        'sum_bid': [vol_log_return], 
        'sum_ask': [vol_log_return], 
        'spread_pct': [vol_log_return], 
        'spread_2_pct': [vol_log_return], 
        'spread': [vol_log_return], 
        'spread_2': [vol_log_return], 
        'bid_ask_ratio': [vol_log_return], 
        'total_bid_ask_ratio': [vol_log_return], 
        'sum_bid_ask': [vol_log_return], 
        'sum_ask_sub_sum_bid': [vol_log_return], 
        'price': [vol_log_return], 
        'seconds_in_bucket':[count_unique],
        'size':[np.sum, vol_log_return, np.mean, np.std, np.max, np.min],
        'order_count':[np.mean,np.sum,np.max],
    }
    dfs = []
    for secs in [0, 150, 350, 450]: 
        tmp = df[df.seconds_in_bucket > secs]
        tmp = tmp.groupby(['time_id']).agg(agg_dict).rename(
            columns={'<lambda_0>': 'mean_decay', 
                     '<lambda_1>': 'mean_decay_flip', 
                     '<lambda_2>': 'mean_decay_95', 
                     '<lambda_3>': 'mean_decay_flip_95',
                    }
        )
        tmp.columns = ['_'.join(c) + f'_{secs}' for c in tmp.columns]
        dfs.append(tmp)
    
    df_agg = reduce(lambda a, b: a.join(b), dfs)

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
            
    df_agg['wap_last-first'] = (df_agg['wap_last_0'] - df_agg['wap_first_0']).abs()
    
    df_agg['dummy1'] = np.random.normal(size=(len(df_agg)))
    df_agg['dummy2'] = np.random.normal(size=(len(df_agg)))
    df_agg['dummy3'] = np.random.normal(size=(len(df_agg)))

    df_agg = df_agg.astype('float32')
    ################# Adding 'row_id' column ##################
    df_agg.reset_index(inplace=True)
    df_agg['time_id'] = df_agg.time_id.apply(lambda x: f"{stock_id}-{x}")
    df_agg.rename({'time_id': 'row_id'}, axis=1, inplace=True)
    return df_agg.set_index('row_id')


####################################################################################
####################################################################################
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def p11(DATA_RAW, stock_id, train_or_test):
    """Copied exactly from highest scoring notebook
    """
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    df['log_return3'] = df.groupby(['time_id'])['wap3'].apply(log_return)
    df['log_return4'] = df.groupby(['time_id'])['wap4'].apply(log_return)
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.std],
        'wap2': [np.sum, np.std],
        'wap3': [np.sum, np.std],
        'wap4': [np.sum, np.std],
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
        'wap_balance': [np.sum, np.max],
        'price_spread':[np.sum, np.max],
        'price_spread2':[np.sum, np.max],
        'bid_spread':[np.sum, np.max],
        'ask_spread':[np.sum, np.max],
        'total_volume':[np.sum, np.max],
        'volume_imbalance':[np.sum, np.max],
        "bid_ask_spread":[np.sum,  np.max],
    }
    create_feature_dict_time = {
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)
    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)
    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id__100'], axis = 1, inplace = True)
    
    
    # Create row_id so we can merge
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    tmp = df_feature.copy()
#     return df_feature

# Function to preprocess trade data (for each stock id)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    df['amount']=df['price']*df['size']
    # Dict for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum, np.max],
        'order_count':[np.sum,np.max],
        'amount':[np.sum,np.max],
    }
    create_feature_dict_time = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.sum],
    }
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    

    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)
    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)
    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)
    
    def tendency(price, vol):    
        df_diff = np.diff(price)
        val = (df_diff/price[1:])*100
        power = np.sum(val*vol[1:])
        return(power)
    
    lis = []
    for n_time_id in df['time_id'].unique():
        df_id = df[df['time_id'] == n_time_id]        
        tendencyV = tendency(df_id['price'].values, df_id['size'].values)      
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max =  np.sum(np.diff(df_id['price'].values) > 0)
        df_min =  np.sum(np.diff(df_id['price'].values) < 0)
        # new
        abs_diff = np.median(np.abs( df_id['price'].values - np.mean(df_id['price'].values)))        
        energy = np.mean(df_id['price'].values**2)
        iqr_p = np.percentile(df_id['price'].values,75) - np.percentile(df_id['price'].values,25)
        
        # vol vars
        
        abs_diff_v = np.median(np.abs( df_id['size'].values - np.mean(df_id['size'].values)))        
        energy_v = np.sum(df_id['size'].values**2)
        iqr_p_v = np.percentile(df_id['size'].values,75) - np.percentile(df_id['size'].values,25)
        
        lis.append({'time_id':n_time_id,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max,'df_min':df_min,
                   'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
    
    df_lr = pd.DataFrame(lis)
        
   
    df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')
    
    # Merge all
    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id','time_id__100'], axis = 1, inplace = True)
    
    
    df_feature = df_feature.add_prefix('trade_')
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    
    return df_feature.set_index('row_id').join(tmp.set_index('row_id')).reset_index()

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_400', 'log_return2_realized_volatility_400', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_200', 'log_return2_realized_volatility_200', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_400', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_200']


    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the stock id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    
    df['size_tau'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique'] )
    df['size_tau_500'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique_500'] )
    df['size_tau_400'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique_400'] )
    df['size_tau_300'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique_300'] )
    df['size_tau_200'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique_200'] )
    df['size_tau_100'] = np.sqrt( 1/ df['trade_seconds_in_bucket_count_unique_100'] )

    df['size_tau2'] = np.sqrt( 1/ df['trade_order_count_sum'] )
    df['size_tau2_500'] = np.sqrt( 0.16/ df['trade_order_count_sum'] )
    df['size_tau2_400'] = np.sqrt( 0.33/ df['trade_order_count_sum'] )
    df['size_tau2_300'] = np.sqrt( 0.5/ df['trade_order_count_sum'] )
    df['size_tau2_200'] = np.sqrt( 0.66/ df['trade_order_count_sum'] )
    df['size_tau2_100'] = np.sqrt( 0.83/ df['trade_order_count_sum'] )

    # delta tau
    df['size_tau2_d'] = df['size_tau2_400'] - df['size_tau2']
    return df

###############################################################################
###############################################################################
###############################################################################


def p12(DATA_RAW, stock_id, train_or_test):
    """Copied exactly from highest scoring notebook
    """
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    df['log_return3'] = df.groupby(['time_id'])['wap3'].apply(log_return)
    df['log_return4'] = df.groupby(['time_id'])['wap4'].apply(log_return)
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.std],
        'wap2': [np.sum, np.std],
        'wap3': [np.sum, np.std],
        'wap4': [np.sum, np.std],
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
        'wap_balance': [np.sum, np.max],
        'price_spread':[np.sum, np.max],
        'price_spread2':[np.sum, np.max],
        'bid_spread':[np.sum, np.max],
        'ask_spread':[np.sum, np.max],
        'total_volume':[np.sum, np.max],
        'volume_imbalance':[np.sum, np.max],
        "bid_ask_spread":[np.sum,  np.max],
    }
    create_feature_dict_time = {
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)
    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)
    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)

    # Merge all
    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id__100'], axis = 1, inplace = True)
    
    
    # Create row_id so we can merge
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    tmp = df_feature.copy()
#     return df_feature

# Function to preprocess trade data (for each stock id)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    df['amount']=df['price']*df['size']
    # Dict for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum, np.max],
        'order_count':[np.sum,np.max],
        'amount':[np.sum,np.max],
    }
    create_feature_dict_time = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.sum],
    }
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict,seconds_in_bucket, add_suffix = False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    

    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict,seconds_in_bucket = 0, add_suffix = False)
    df_feature_500 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 500, add_suffix = True)
    df_feature_400 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 400, add_suffix = True)
    df_feature_300 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 300, add_suffix = True)
    df_feature_200 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 200, add_suffix = True)
    df_feature_100 = get_stats_window(create_feature_dict_time,seconds_in_bucket = 100, add_suffix = True)
    
    def tendency(price, vol):    
        df_diff = np.diff(price)
        val = (df_diff/price[1:])*100
        power = np.sum(val*vol[1:])
        return(power)
    
    lis = []
    for n_time_id in df['time_id'].unique():
        df_id = df[df['time_id'] == n_time_id]        
        tendencyV = tendency(df_id['price'].values, df_id['size'].values)      
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max =  np.sum(np.diff(df_id['price'].values) > 0)
        df_min =  np.sum(np.diff(df_id['price'].values) < 0)
        # new
        abs_diff = np.median(np.abs( df_id['price'].values - np.mean(df_id['price'].values)))        
        energy = np.mean(df_id['price'].values**2)
        iqr_p = np.percentile(df_id['price'].values,75) - np.percentile(df_id['price'].values,25)
        
        # vol vars
        
        abs_diff_v = np.median(np.abs( df_id['size'].values - np.mean(df_id['size'].values)))        
        energy_v = np.sum(df_id['size'].values**2)
        iqr_p_v = np.percentile(df_id['size'].values,75) - np.percentile(df_id['size'].values,25)
        
        lis.append({'time_id':n_time_id,'tendency':tendencyV,'f_max':f_max,'f_min':f_min,'df_max':df_max,'df_min':df_min,
                   'abs_diff':abs_diff,'energy':energy,'iqr_p':iqr_p,'abs_diff_v':abs_diff_v,'energy_v':energy_v,'iqr_p_v':iqr_p_v})
    
    df_lr = pd.DataFrame(lis)
        
   
    df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')
    
    # Merge all
    df_feature = df_feature.merge(df_feature_500, how = 'left', left_on = 'time_id_', right_on = 'time_id__500')
    df_feature = df_feature.merge(df_feature_400, how = 'left', left_on = 'time_id_', right_on = 'time_id__400')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_200, how = 'left', left_on = 'time_id_', right_on = 'time_id__200')
    df_feature = df_feature.merge(df_feature_100, how = 'left', left_on = 'time_id_', right_on = 'time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500','time_id__400', 'time_id__300', 'time_id__200','time_id','time_id__100'], axis = 1, inplace = True)
    
    
    df_feature = df_feature.add_prefix('trade_')
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_feature[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_feature[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_feature[f'real_vol_mean_decay_{decay}_{step}'] = mean_decay(x, decay, step, axis=1)
        
    return df_feature.set_index('row_id').join(tmp.set_index('row_id')).reset_index()

#############################################################
#############################################################
#############################################################
# These functions were created here but moved to opt_utils.py
# def add_wap_2(df): 
#     """Adds the weighted average price to a book df."""
#     df['wap_2'] = (df['bid_price_mean'] * df['sum_ask'] + df['ask_price_mean'] * df['sum_bid']) / (df['sum_bid']+ df['sum_ask'])
#     return df

# def count_unique(x): return len(np.unique(x))

def p13(DATA_RAW, stock_id, train_or_test):
    """Starting with features from 8, removing a few that were obviously not good. Adding normalization to
    size features so they are more comparable accross different stocks. """
    
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    size_sum = df['size'].sum() * 600 / (len(df)) # To normalize size features, getting mean per time bucket 
    order_sum = (df['order_count'].sum() * 600 / (len(df))) # To normalize order_count features, getting mean per time bucket
    df['size_norm'] = df['size'] / size_sum
    df['order_norm'] = df['order_count'] / order_sum
    
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['bid_price_diff'] = df.groupby('time_id')['bid_price1'].diff()
    df['ask_price_diff'] = df.groupby('time_id')['ask_price1'].diff()
    
    df['sum_bid'] = (df.bid_size1 + df.bid_size2) / size_sum 
    df['sum_ask'] = (df.ask_size1 + df.ask_size2) / size_sum
    df['ask_price_mean'] = (df.ask_price1 * df.ask_size1 + df.ask_price2 * df.ask_size2) / df['sum_ask']
    df['bid_price_mean'] = (df.bid_price1 * df.bid_size1 + df.bid_price2 * df.bid_size2) / df['sum_bid']
    add_wap_2(df)
    df['log_return_2'] = df.groupby(['time_id'])['wap_2'].apply(log_return)
    df['abs_log_return_2'] = df['log_return_2'].abs()
    df['is_pos_return_2'] = (df['log_return_2'] > 0).astype(int)
    df['is_neg_return_2'] = (df['log_return_2'] < 0).astype(int)
    
    df['size_spread'] = (df['sum_ask'] - df['sum_bid']) / size_sum
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask'] / size_sum
    df['bid_ask_ratio'] = df.bid_size1 / df.ask_size1
    df['total_bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['price_wap_diff'] = df['price'] - df['wap']
    df['price_wap_diff_2'] = df['price'] - df['wap_2']
    df['abs_price_wap_diff'] = (df['price'] - df['wap']).abs()
    df['abs_price_wap_diff_2'] = (df['price'] - df['wap_2']).abs()
    
    df['order_size_sqaure_weighted'] = ((df['size'] / df['order_count']) ** 2) * df['size']

    agg_dict = {
        'log_return': [max_sub_min, np.std, realized_volatility, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'log_return_2': [max_sub_min, np.std, realized_volatility, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return_2': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return_2': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return_2': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        
        'sum_bid': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size_spread': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'sum_bid_ask': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'size_norm': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'order_norm': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
#         'total_bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        
        'wap': [max_sub_min, np.std, first, last,],
        'wap_2': [max_sub_min, np.std, first, last,],
        'price_wap_diff': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'price_wap_diff_2': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_price_wap_diff': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_price_wap_diff_2': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_2': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread_2_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'order_count': ['sum'], 
        'order_size_sqaure_weighted': ['sum'], 
        'bid_price_diff': [count_unique], 
        'ask_price_diff': [count_unique], 
    }
    
    
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]
    
    # Variance = E[X^2] - E[X]^2, normalized by the average 
    df_agg['mean_squares'] = df_agg['order_size_sqaure_weighted_sum'] / df_agg['order_count_sum']
    df_agg['order_size_mean'] = df_agg['size_sum'] / df_agg['order_count_sum']
    df_agg['order_size_var_norm'] = (df_agg['mean_squares'] - df_agg['order_size_mean'] ** 2) / df_agg['order_size_mean']
    
    df_agg['ratio_sales_ask_bid_size'] = df_agg['size_sum'] / df_agg['sum_bid_ask_mean']

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] = mean_decay(x, decay, step, axis=1)
        
    ############ Realized volume for each minute using wap 2 ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}_2'] = df[mask].groupby('time_id')['log_return_2'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}_2' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}_2'] = mean_decay(x, decay, step, axis=1)
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
            
    df_agg['wap_last-first'] = (df_agg['wap_last'] - df_agg['wap_first']).abs()
    df_agg['wap_2_last-first'] = (df_agg['wap_2_last'] - df_agg['wap_2_first']).abs()
    
    df_agg = df_agg.join(df.groupby(['time_id'])['bid_price1', 'ask_price1', 'bid_size1', 
                            'sum_bid', 'sum_ask', 'spread_pct', 'spread_2_pct', 'spread', 'spread_2', 'size_spread',
                            'bid_ask_ratio', 'total_bid_ask_ratio', 'sum_bid_ask',  
                            'size', 'price', 'wap_2'
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




def p14(DATA_RAW, stock_id, train_or_test):
    """Starting with features from 8, removing a few that were obviously not good. Adding normalization to
    size features so they are more comparable accross different stocks.
    """
    
    df = load_bt(DATA_RAW, stock_id, train_or_test)
    size_sum = df['size'].sum() * 600 / (len(df)) # To normalize size features, getting mean per time bucket 
    order_sum = (df['order_count'].sum() * 600 / (len(df))) # To normalize order_count features, getting mean per time bucket
    df['size_norm'] = df['size'] / size_sum
    df['order_norm'] = df['order_count'] / order_sum
    
    df = add_wap(df)
    df['log_return'] = df.groupby(['time_id'])['wap'].apply(log_return)
    df['abs_log_return'] = df['log_return'].abs()
    df['is_pos_return'] = (df['log_return'] > 0).astype(int)
    df['is_neg_return'] = (df['log_return'] < 0).astype(int)
    df['spread_pct'] = (df.ask_price1 - df.bid_price1) / df.wap
    df['spread_2_pct'] = (df.ask_price2 - df.bid_price2) / df.wap
    df['spread'] = (df.ask_price1 - df.bid_price1) 
    df['spread_2'] = (df.ask_price2 - df.bid_price2) 
    df['bid_price_diff'] = df.groupby('time_id')['bid_price1'].diff()
    df['ask_price_diff'] = df.groupby('time_id')['ask_price1'].diff()
    
    df['sum_bid'] = (df.bid_size1 + df.bid_size2) / size_sum 
    df['sum_ask'] = (df.ask_size1 + df.ask_size2) / size_sum
    df['ask_price_mean'] = (df.ask_price1 * df.ask_size1 + df.ask_price2 * df.ask_size2) / df['sum_ask']
    df['bid_price_mean'] = (df.bid_price1 * df.bid_size1 + df.bid_price2 * df.bid_size2) / df['sum_bid']
    add_wap_2(df)
    df['log_return_2'] = df.groupby(['time_id'])['wap_2'].apply(log_return)
    df['abs_log_return_2'] = df['log_return_2'].abs()
    df['is_pos_return_2'] = (df['log_return_2'] > 0).astype(int)
    df['is_neg_return_2'] = (df['log_return_2'] < 0).astype(int)
    
    df['size_spread'] = (df['sum_ask'] - df['sum_bid']) / size_sum
    df['sum_bid_ask'] = df['sum_bid'] + df['sum_ask'] / size_sum
    df['bid_ask_ratio'] = df.bid_size1 / df.ask_size1
    df['total_bid_ask_ratio'] = df['sum_bid'] / df['sum_ask']
    df['price_wap_diff'] = df['price'] - df['wap']
    df['price_wap_diff_2'] = df['price'] - df['wap_2']
    df['abs_price_wap_diff'] = (df['price'] - df['wap']).abs()
    df['abs_price_wap_diff_2'] = (df['price'] - df['wap_2']).abs()
    
    df['order_size_sqaure_weighted'] = ((df['size'] / df['order_count']) ** 2) * df['size']

    agg_dict = {
        'log_return': [max_sub_min, np.std, realized_volatility, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'log_return_2': [max_sub_min, np.std, realized_volatility, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'is_pos_return_2': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)], 
        'is_neg_return_2': [np.std, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_log_return_2': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        
        'sum_bid': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'sum_ask': [max_sub_min, np.std, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size_spread': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'sum_bid_ask': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'size': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'size_norm': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'order_norm': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'price': [max_sub_min, np.std, np.mean, np.sum, count_unique, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
#         'total_bid_ask_ratio': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        
        'wap': [max_sub_min, np.std, first, last,],
        'wap_2': [max_sub_min, np.std, first, last,],
        'price_wap_diff': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'price_wap_diff_2': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_price_wap_diff': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'abs_price_wap_diff_2': [max_sub_min, np.std, np.mean, np.min, np.max, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_2': [max_sub_min, np.std, np.mean, np.sum, get_mean_decay(.99, -1), get_mean_decay(.99, 1), get_mean_decay(.95, -1), get_mean_decay(.95, 1)],
        'spread_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'spread_2_pct': [max_sub_min, np.std, np.mean, get_mean_decay(.99, -1), get_mean_decay(.99, 1)],
        'order_count': ['sum'], 
        'order_size_sqaure_weighted': ['sum'], 
        'bid_price_diff': [count_unique], 
        'ask_price_diff': [count_unique], 
    }
    
    
    df_agg = df.groupby(['time_id']).agg(agg_dict).rename(
        columns={'<lambda_0>': 'mean_decay', 
                 '<lambda_1>': 'mean_decay_flip', 
                 '<lambda_2>': 'mean_decay_95', 
                 '<lambda_3>': 'mean_decay_flip_95',
                }
    )
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]
    
    for m in range(1, 3): 
        mask = (df.seconds_in_bucket >= 300 * m - 300) & (df.seconds_in_bucket < 300 * m)
        df_agg_tmp = df[mask].groupby(['time_id']).agg(agg_dict).rename(
            columns={'<lambda_0>': 'mean_decay', 
                     '<lambda_1>': 'mean_decay_flip', 
                     '<lambda_2>': 'mean_decay_95', 
                     '<lambda_3>': 'mean_decay_flip_95',
                    }
        )
        df_agg_tmp.columns = ['_'.join(c) + f'_{300 * m}' for c in df_agg_tmp.columns]
        df_agg = pd.concat([df_agg, df_agg_tmp], axis=1)
        
    for col in df_agg.columns: 
        if col.endswith('_600'): 
            df_agg[col[:-4] + '_half_momentum'] = df_agg[col] / df_agg[col.replace('_600', '_300')]
    
    # Variance = E[X^2] - E[X]^2, normalized by the average 
    df_agg['mean_squares'] = df_agg['order_size_sqaure_weighted_sum'] / df_agg['order_count_sum']
    df_agg['order_size_mean'] = df_agg['size_sum'] / df_agg['order_count_sum']
    df_agg['order_size_var_norm'] = (df_agg['mean_squares'] - df_agg['order_size_mean'] ** 2) / df_agg['order_size_mean']
    
    df_agg['ratio_sales_ask_bid_size'] = df_agg['size_sum'] / df_agg['sum_bid_ask_mean']

    ############ Realized volume for each minute ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}'] = df[mask].groupby('time_id')['log_return'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}'] = mean_decay(x, decay, step, axis=1)
        
    ############ Realized volume for each minute using wap 2 ############
    for m in range(1, 11): 
        mask = (df.seconds_in_bucket >= 60 * m - 60) & (df.seconds_in_bucket < 60 * m)
        df_agg[f'real_vol_min_{m}_2'] = df[mask].groupby('time_id')['log_return_2'].agg(realized_volatility)

    ######### Decay sum of realized volume per minute ########
    cols = [f'real_vol_min_{minute}_2' for minute in range(1, 11)]
    x = df_agg[cols].values
    for decay, step in product((.99, .95, .9, .85, .75, .65, .55, .45), (1, -1)): 
        df_agg[f'real_vol_mean_decay_{decay}_{step}_2'] = mean_decay(x, decay, step, axis=1)
    
    for c1, c2 in zip(df_agg.columns, df_agg.columns[1:]): 
        if 'mean_decay_flip' in c2: 
            pre, suf = c2.split('mean_decay_flip')
            df_agg[pre + 'momentum' + suf] = df_agg[c1] / df_agg[c2]
        if 'vol_mean_decay' in c2 and '-1' in c2: 
            pre, suf = c2.split('vol_mean_decay')
            df_agg[pre + 'momentum' + suf] = df_agg[c2] / df_agg[c1]
            
    df_agg['wap_last-first'] = (df_agg['wap_last'] - df_agg['wap_first']).abs()
    df_agg['wap_2_last-first'] = (df_agg['wap_2_last'] - df_agg['wap_2_first']).abs()
    
    df_agg = df_agg.join(df.groupby(['time_id'])['bid_price1', 'ask_price1', 'bid_size1', 
                            'sum_bid', 'sum_ask', 'spread_pct', 'spread_2_pct', 'spread', 'spread_2', 'size_spread',
                            'bid_ask_ratio', 'total_bid_ask_ratio', 'sum_bid_ask',  
                            'size', 'price', 'wap_2'
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






