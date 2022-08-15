# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + _cell_guid="d6c21b6a-db9b-4a95-a5b0-d96b77cc1397" _uuid="228575b3-52c6-4082-9106-13a5af713005" papermill={"duration": 2.737639, "end_time": "2021-08-28T09:54:27.805811", "exception": false, "start_time": "2021-08-28T09:54:25.068172", "status": "completed"} tags=[]
from opt_utils import * 
from opt_fe import * 
cfg = {
    "path_data_raw": 'input/',
    "preprocessor_func": p13, 
    "testing_func": False, 
}
DATA_RAW = 'input/'

# + papermill={"duration": 7718.362104, "end_time": "2021-08-28T12:03:06.177313", "exception": false, "start_time": "2021-08-28T09:54:27.815209", "status": "completed"} tags=[]
make_train(cfg)

# + papermill={"duration": 0.285087, "end_time": "2021-08-28T12:03:06.547628", "exception": false, "start_time": "2021-08-28T12:03:06.262541", "status": "completed"} tags=[]
train = pd.read_csv(os.path.join(DATA_RAW, 'train.csv'))

# + papermill={"duration": 1.871318, "end_time": "2021-08-28T12:03:08.503735", "exception": false, "start_time": "2021-08-28T12:03:06.632417", "status": "completed"} tags=[]
df = pd.read_pickle('p13_train.pkl')
train['row_id'] = train.stock_id.astype(str) + '-' + train.time_id.astype(str)
assert all(train.row_id == df.row_id)

# + papermill={"duration": 0.093389, "end_time": "2021-08-28T12:03:08.681766", "exception": false, "start_time": "2021-08-28T12:03:08.588377", "status": "completed"} tags=[]
print('shape of training data: ', df.shape)

# + papermill={"duration": 2.13227, "end_time": "2021-08-28T12:03:10.899464", "exception": false, "start_time": "2021-08-28T12:03:08.767194", "status": "completed"} tags=[]
print('Features sorted by correlation with target')
display(df.corrwith(train['target']).sort_values(key=lambda x: abs(x), ascending=False))

# + papermill={"duration": 0.106307, "end_time": "2021-08-28T12:03:11.672635", "exception": false, "start_time": "2021-08-28T12:03:11.566328", "status": "completed"} tags=[]
print('rmspe score just using the weighted average price:')
display(rmspe(train['target'], df['wap_2_real_vol']))
print('rmspe score just using the log_return_realized_volatility feature')
display(rmspe(train['target'], df['log_return_realized_volatility']))

# + papermill={"duration": 0.093088, "end_time": "2021-08-28T12:03:12.048594", "exception": false, "start_time": "2021-08-28T12:03:11.955506", "status": "completed"} tags=[]

