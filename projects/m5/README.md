
# Welcome to chrisrichardmiles.m5

> A submodule built for the M5-Accuracy competition on Kaggle. Chris
> Miles placed 77th in the competition

This package is built with [nbdev](https://nbdev.fast.ai/), so the
source code, testing, and documentation are all built in jupyter
notebooks. Please read through the notebooks or online documentation
sequentially for more information about the solution.

## Create submission from scratch

Requirements: 16 of RAM and pip installed \### 1. Install package with
pip `pip install chrisrichardmiles`

### 2. Create data folders, download data, and unzip files

-   If you have your kaggle api info in root/.kaggle/kaggle.json then
    run:

<!-- -->

    crm_download_kaggle_data --comp_name m5-forecasting-accuracy

-   Otherwise, you must run:

<!-- -->

    crm_mkdirs_data
    cd data/raw

-   Now manully download the data zipfile from
    [kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/data) and
    upload it into the data/raw folder.

<!-- -->

    unzip * 
    cd ../..

### 3. Detect out of stock days for products and change sales to NaN.

    crm_m5_make_oos_data

### 4. Create features for training

    crm_m5_fe

### 5. Train models and create submission

    crm_m5_lgb_daily

## Some lessons I learned

### RAM issues

-   Computing rolling window statistics can be very expensive, but can
    be ok if we do processing in smaller sections like we do with
    `n_splits` in the `rolling_window` statistics calculations.
-   Datatype matter: float32 and float16 datatypes can save a lot of
    RAM, but be careful for float16, which might cause accuracy problems
    or completely break a process, like when trying to use
    `StandardScaler` with float16 datatypes resulted in all zeros.
-   Be careful with Pandas DataFrames. I had a lot of problems where I
    was unmindfully creating large copies of data, such as a saving
    function, looking something like `df[cols].to_csv(....)` which was
    making an entirely new dataframe in memory before saving. This was
    remedied by using the `usecols` param in `df.to_csv`
-   Be careful with DataFrame names. It seems that it is better to keep
    the same name of a DataFrame when doing things like concating an
    existing df with new data. Its like pandas will use memory more
    intelligently. So this:
    `df2 = pd.concat([df1, pd.read_csv('data.csv')]; del df2` should be
    replaced with `df1 = pd.concat([df1, pd.read_csv('data.csv')]`. It
    seems like it should work the same, but my experience with RAM
    crashes seems to indicate the second method is much better.

### Organization is important

The first iteration of this project was code spread accross hundreds of
kaggle notebooks. I ran experiments by running a notebook and just
looking at the result. It always felt ok while I was doing it, and it
worked out ok, but I didn’t have the results saved in a proper manner.
Now I use neptune.ai to track experiments.
