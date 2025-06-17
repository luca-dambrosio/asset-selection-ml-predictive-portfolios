import pandas as pd
import numpy as np
import os

def save_data(data):
    """
    Splits a dataset into chunks of 50,000 rows and saves each chunk into CSV files,
    organizing the files into folders. Each folder, named 'BATCH_{batch_number}', contains 
    up to 4 datasets. The function ensures that there are a maximum of 10 folders.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset to be split and saved.
    """
        
    length = data.shape[0]
    i = 0
    span = 50000
    iter = 0
    batch_size = 4
    max_batches = 10

    while i < length and iter < batch_size * max_batches:
        # Determine batch number based on the dataset number
        batch_number = iter // batch_size + 1
        batch_folder = f"BATCH_{batch_number}"

        # Create the folder if it does not exist
        os.makedirs(batch_folder, exist_ok=True)

        # Define the subset of data
        if i + span > length:
            subset = data.iloc[i:, :].copy()
            subset.to_csv(os.path.join(batch_folder, f"GFD_{iter}.csv"), index=False)
            break
        else:
            subset = data.iloc[i:i + span, :].copy()
            subset.to_csv(os.path.join(batch_folder, f"GFD_{iter}.csv"), index=False)
            i += span
            iter += 1

def load_data():
    """
    Returns: Pandas Dataframe.

    This function collects the chunks in which we 
    subdivided the data and concatenates them.
    This is fundamental to avoid RAM overloads.
    """

    count = 0
    # Iterate through the batches we defined
    for batch in range(1,11):
        left = count
        right = count + 3

        # Get each file within a batch (4 per batch)
        for file in range(left,right + 1):
            dir = f"BATCH_{batch}/GFD_{file}.csv" # define directory
            new_df = pd.read_csv(dir)

            if count == 0:
                df = new_df.copy()
                count += 1
                continue

            else:
                df = pd.concat([df,new_df],axis = 0, ignore_index = True)
                print(f"Dataframes Loaded: {count + 1}/40", end = "\r")
                count += 1
    #df.drop("Unnamed: 0", axis = 1, inplace = True)
    df.eom = pd.to_datetime(df.eom)
    return df

def train_test_split(X,y_col, start_val, end_val, start_test, end_test):

    """
    Splits the input data (X) into training, validation, and test sets based on the specified years to put in each set.

    Parameters:
    -----------
    X : pandas.DataFrame
        The input dataset, which must contain a column `eom` of datetime type and a target column specified by `y_col`.

    y_col : str
        The name of the target variable column in the dataset. This column will be split into `y_train`, `y_val`, and `y_test`.

    start_val : int
        The starting year for the validation period. All rows with a year before this value are assigned to the "Train" set.

    end_val : int
        The ending year for the validation period. All rows with a year between `start_val` and `end_val` (inclusive) are assigned to the "Val" set.

    start_test : int
        The single year for the test period. All rows with this exact year are assigned to the "Test" set.

    Returns:
    --------
    X_train : pandas.DataFrame
        The training set, containing the features of the data excluding the target column (`y_col`).

    X_val : pandas.DataFrame
        The validation set, containing the features of the data excluding the target column (`y_col`).

    X_test : pandas.DataFrame
        The test set, containing the features of the data excluding the target column (`y_col`).

    y_train : pandas.Series
        The target variable values for the training set.

    y_val : pandas.Series
        The target variable values for the validation set.

    y_test : pandas.Series
        The target variable values for the test set.
    """

    # Set the conditions for the split
    conditions = [(X.eom.dt.year < start_val),
                   (X.eom.dt.year >= start_val) & (X.eom.dt.year <= end_val),
                   (X.eom.dt.year >= start_test) & (X.eom.dt.year <= end_test)]
    
    #Create and map a flag to define in which set each observation will fall
    mappings = ["Train","Val","Test"]
    X["split"] = np.select(conditions, mappings, "Exclude")

    # Store each subset
    X_train = X.query("split == 'Train'").reset_index(drop = True)
    X_val = X.query("split == 'Val'").reset_index(drop = True)
    X_test = X.query("split == 'Test'").reset_index(drop = True)

    #Extract and drop the target variable
    y_train = X_train[y_col].reset_index(drop = True)
    y_val = X_val[y_col].reset_index(drop = True)
    y_test = X_test[y_col].reset_index(drop = True)

    X_train.drop(y_col, axis = 1, inplace = True)
    X_val.drop(y_col, axis = 1, inplace = True)
    X_test.drop(y_col, axis = 1, inplace = True)


    return X_train, X_val, X_test, y_train, y_val, y_test