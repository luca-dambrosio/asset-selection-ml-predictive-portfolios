import numpy as np
import pandas as pd
import random
random.seed(28)
import matplotlib.pyplot as plt
import warnings
# Disable the specific PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def add_dummies(df,col_name):
    """
    Parameters -->  df : dataframe to process
                    col_name: categorical variable column
    Returns -->     df: dataframe with added dummies and dropped col_name 

    """

    return pd.concat([df,pd.get_dummies(df[col_name], drop_first = True)], axis = 1).drop(col_name,axis=1)

def cap_outliers(series, lower_percentile=1, upper_percentile=99):
    """
    Caps outliers in a pandas Series based on specified percentiles.
    
    Any value below the lower percentile is capped to the lower bound, 
    and any value above the upper percentile is capped to the upper bound.
    
    Parameters:
    - series: The pandas Series to modify.
    - lower_percentile: The lower percentile (default is 1).
    - upper_percentile: The upper percentile (default is 99).
    
    Returns:
    - pandas Series with outliers capped.
    """
    series = series.copy()  # Create a copy of the series to avoid SettingWithCopyWarning
    
    lower_bound = series.quantile(lower_percentile / 100)
    upper_bound = series.quantile(upper_percentile / 100)
    
    series.loc[series < lower_bound] = lower_bound
    series.loc[series > upper_bound] = upper_bound
    
    return series

def min_max_normalize(df, column_name):
    """
    Performs min-max normalization on a specified column of a DataFrame.
    
    For each observation in the column:
        normalized_value = (value - min(column)) / (max(column) - min(column))
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to normalize.
        column_name (str): The name of the column to normalize.
    
    Returns:
        pd.Series: The normalized column as a pandas Series.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    
    normalized_column = (df[column_name] - col_min) / (col_max - col_min)
    return normalized_column

def preprocess(df=pd.read_csv("GFD_final.csv")):
    """
    Preprocesses the input DataFrame by performing several data cleaning and transformation steps:
    - Reads the input df
    - Drops rows with missing values in the target columns 'ret' and 'y'.
    - Fills missing values in categorical columns with 'missing' and applies one-hot encoding.
    - Drops features that have more than 500 stocks missings and drops stocks that have missing features
    - Imputes missing values in continuous columns with the median for that specific stock/feature.
    - Caps outliers in continuous columns to the 1% and 99% percentiles.
    - Applies min-max normalization to continuous columns and target variables.
    - Returns the cleaned and preprocessed DataFrame.

    Parameters:
    df (str): The path to the CSV file to be processed (default: "GFD_final.csv").

    Returns:
    pandas.DataFrame: The cleaned and preprocessed DataFrame.
    """

    # Remove observations that have missing returns
    df = df.dropna(subset=["y", "ret"])

    # Inspecting the columns 1 by 1 by hand to check whether continuous/ordinal or categorical
    categ = ["size_grp","ff49"]
    returns = ["ret","y"]
    ids = ["id", "eom","excntry","gvkey","permno"]
    conti = list(set(df.columns).difference(set(categ)).difference(set(returns)).difference(set(ids)))

    # Adding category in column ff49 for missings
    df[categ[1]] = df[categ[1]].fillna('missing')

    # One-hot encoding
    df = add_dummies(df, categ[0])
    df = add_dummies(df, categ[1])


    # Check for columns where all values in a group are NaN
    all_nan = df.groupby('id').apply(lambda group: group.isna().all())
    # We drop features for which 500 or more stocks had no information for that feature
    missing_cols = all_nan.sum().to_dict()
    missing_500 = {k:v for k,v in missing_cols.items() if v<=500}
    missing =  list(missing_500.keys())
    # Manually printing because we want to remove this features independently for each subset of the data
    missing = ['niq_su', 'saleq_su', 'ni_inc8q', 'resff3_6_1', 'resff3_12_1', 'ret_60_12', 'sale_gr3', 'ival_me', 'capex_abn', 
           'ppeinv_gr1a', 'capx_gr2', 'capx_gr3', 'debt_gr3', 'inv_gr1', 'ope_be', 'ope_bel1', 'f_score', 'pi_nix', 'rd_me', 
           'rd_sale', 'emp_gr1', 'rd5_at', 'dsale_dinv', 'dgp_dsale', 'dsale_dsga', 'sale_emp_gr1', 'ocfq_saleq_std', 'ni_ar1', 
           'ni_ivol', 'earnings_variability', 'aliq_mat', 'seas_2_5an', 'seas_2_5na', 'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 
           'seas_11_15na', 'seas_16_20an', 'seas_16_20na', 'beta_60m', 'betabab_1260d', 'corr_1260d', 'qmj', 'qmj_growth']
    df = df.drop(columns = missing).reset_index(drop=True)
    non_missing_sub = set(list(missing_cols.keys())).difference(set(missing))

    # Filtering out features with too many missings
    conti_features = list(set(conti).intersection(set(non_missing_sub))) + ["id"]
    df[conti_features] = df[conti_features].groupby('id').apply(lambda group: group.fillna(group.median())).reset_index(drop=True)

    # Filtering out obs with missings for the whole stock
    missing_obs = df[df.isna().any(axis=1)].index.to_list()
    df = df.drop(missing_obs).reset_index(drop=True)

    # We want to leave the following column unchanged from now on:
    # Remove specific items from conti_features
    for col in ["id", "date", "RF"]:
        conti_features.remove(col)
    
    # Caping outliers
    for col in conti_features:
        df[col] = cap_outliers(df[col])

    # We apply the min-max normalization using our custom function
    for col in conti_features:
        df[col] =  min_max_normalize(df, col)

    return df

