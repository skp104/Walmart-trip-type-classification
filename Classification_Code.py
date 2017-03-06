from __future__ import print_function
import time
import os
import pandas as pd
import numpy as np
from numpy import float64
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Code to read data from a location on the PC or Web
start_time = time.time()

def get_walmart_data():
    """Get the data, from local csv or pandas repo."""
    if os.path.exists("train2.csv"):
        print("-- train.csv found locally")
        df = pd.read_csv("train2.csv")

    else:
        print("File Not Found")
        fn = "https://onedrive.live.com//redir?resid=CA4A8A4256D7DBE8!390&authkey=!AAexODgwK8DSWBY&ithint=file%2ccsv"
        df = pd.read_csv(fn, error_bad_lines=False, sep='\t')
    return df


df = get_walmart_data()

def encode_target1(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: float64(n) for n, name in enumerate(targets)}
    df_mod["Target1"] = df_mod[target_column].replace(map_to_int)
    return df_mod, targets


def encode_target2(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: float64(n) for n, name in enumerate(targets)}
    df_mod["Target2"] = df_mod[target_column].replace(map_to_int)
    return df_mod, targets


df2, targets1 = encode_target1(df, "DepartmentDescription")

df3, targets2 = encode_target2(df2, "Weekday")

features1 = list(df3.columns[1:9])

# features2 = list((df3.columns[1],df3.columns[4],df3.columns[6],df3.columns[7],df3.columns[8]))

features2 = list((df3.columns[3], df3.columns[4], df3.columns[5], df3.columns[6], df3.columns[7], df3.columns[8]))

# print(features1)
# print(features2)

y = df3["TripType"]
X = df3[features2]
X = np.array(X)
y = np.array(y)

dt = DecisionTreeClassifier(min_samples_split=2000, random_state=99)

dt.fit(X, y)

def get_walmart_test_data():
    """Get the data, from local csv or pandas repo."""
    if os.path.exists("test.csv"):
        print("-- test.csv found locally")
        dftest = pd.read_csv("test.csv")


    else:
        print("File Not Found")
        fn = "https://onedrive.live.com//redir?resid=CA4A8A4256D7DBE8!390&authkey=!AAexODgwK8DSWBY&ithint=file%2ccsv"
        dftest = pd.read_csv(fn, error_bad_lines=False, sep='\t')
    return dftest


dftest = get_walmart_test_data()

dftest2, targetstest1 = encode_target1(dftest, "DepartmentDescription")

dftest3, targetstest2 = encode_target2(dftest2, "Weekday")

# print(list(dftest3.columns))

featurestest2 = list((
                     dftest3.columns[0], dftest3.columns[2], dftest3.columns[3], dftest3.columns[5], dftest3.columns[6],
                     dftest3.columns[7]))

# print(featurestest2)

DATA = df3[featurestest2]
print("The predicted TripType classes are as below")
print(dt.predict(DATA))
y_pred = dt.predict(DATA)
print ("The Predicted Trip Type classes for the test data have been listed in predictions.txt file in the current directory")
np.savetxt('predictions.txt',y_pred,delimiter=' ')

# print("--- %s seconds ---" % (time.time() - start_time))