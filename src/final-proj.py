# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.covariance import EmpiricalCovariance


rnd = 42

# Import Data File
sgcredit_df = pd.read_table("../data/SouthGermanCredit/SouthGermanCredit.asc", header=0, sep=None)

# Change column names from German to English

sgcredit_df = sgcredit_df.rename(columns=({"laufkont" : "status", "laufzeit" : "duration", "moral" : "credit history", "verw" : "purpose", "hoehe" : "amount", 
                    "sparkont" : "savings", "beszeit" : "employment duration", "rate" : "installment rate",
                    "famges" : "personal status sex", "buerge" : "other debtors",
                    "wohnzeit" : "present residence", "verm" : "property",
                    "alter" : "age", "weitkred" : "other installment plans",
                    "wohn" : "housing", "bishkred" : "number credits",
                    "beruf" : "job", "pers" : "people liable", "telef" : "telephone", "gastarb" : "foreign worker",
                    "kredit" : "credit risk"}))

# Call scaler and one hot encoder
scaler = StandardScaler()
ohe = OneHotEncoder()

# Use scaler and transform continuous columns
sgcredit_df["age"] = scaler.fit_transform(sgcredit_df["age"].values.reshape(-1,1))
sgcredit_df["amount"] = scaler.fit_transform(sgcredit_df["amount"].values.reshape(-1,1))
sgcredit_df["duration"] = scaler.fit_transform(sgcredit_df["duration"].values.reshape(-1,1))

# set Target and Features
X=sgcredit_df.drop("credit risk", axis=1)
y=sgcredit_df[["credit risk"]]

# Get training set and test set data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rnd)


def get_dummies_pipe(col):
    """ 
    Take column name from training data as string to one hot encode
    """
    df = pd.get_dummies(X_train[col], prefix=col)
    return df 

def get_dummies_pipe_test(col):
    """ 
    Take column name from test data as string to one hot encode
    """
    df = pd.get_dummies(X_test[col], prefix=col)
    return df 

def merge_one_hot_column(to_df, merge_df):
    """ 
    Takes two dataframes and merges
    """
    to_df = to_df.merge(merge_df, left_index=True, right_index=True)

def scale_transform(df, col):
    """
    Takes Dataframe and column name (str) applies scaler and transforms
    """
    df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

def onehotadjust (df, col):
    """process column with two variables to one hot binary"""

    df[col] = df[col].apply(lambda x : x -1)





def preprocesspipe(df):

    onehotadjust(df, "telephone")
    onehotadjust(df, "foreign worker")
    df["2-3 credits"] = df["number credits"].map(lambda x: 1.0 if x== 2 else 0.0)
    df["4-5 credits"] = df["number credits"].map(lambda x: 1.0 if x== 3 else 0.0)
    df["6+ credits"] = df["number credits"].map(lambda x: 1.0 if x== 4 else 0.0)














