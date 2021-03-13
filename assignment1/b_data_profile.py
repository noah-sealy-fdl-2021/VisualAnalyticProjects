from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


'''
NOTE: I added some print statements to help looking at the functions output for texting
I commented them out as they cause quite a lot of clutter while using them for larger functions
BUT feel free to comment any of the print statements back in while testing if it helps :)
'''


def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    # print("Max value in " + column_name + ": " + str(df[column_name].max()))
    return df[column_name].max()


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    # print("Min value in " + column_name + ": " + str(df[column_name].min()))
    return df[column_name].min()


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    # print("Mean for " + column_name + ": " + str(df[column_name].mean()))
    return df[column_name].mean()


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    # print("Amount of missing values in " + column_name + ": " + str(df[column_name].isnull().sum()))
    return df[column_name].isnull().sum()


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    # print("Amount of duplicates in " + column_name + ": " + str(df[column_name].duplicated().sum()))
    return df[column_name].duplicated().sum()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.select_dtypes(include="number").columns:
        cols.append(col)
    # print("Numeric Data Types (in df): " + str(cols))
    return cols


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.select_dtypes(include="bool").columns:
        cols.append(col)
    # print("Binary Data Types (in df): " + str(cols))
    return cols


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    """"
    CITATION
    this Stack Overflow post helped with this method
    the user's name was 'cs95'
    I fully acknowledge and give credit to them for the code found from their section in this article
    The article can be found at the link below
    https://stackoverflow.com/questions/45836794/selecting-string-columns-in-pandas-df-equivalent-to-df-select-dtypes
    """
    query = (df.applymap(type) == str).all(0)
    string_df = df[df.columns[query]]
    # print("Categorical Data Types (in df): " + str(cols))
    return list(string_df)


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    # print("Correlation between " + col1 + " and " + col2 + ": " + str(df[col1].corr(df[col2])))
    cor = df[col1].corr(df[col2])
    print(cor)
    return cor


if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))
    #a = pandas_profile(df)
    #assert get_column_max(df, df.columns[0]) is not None
    #assert get_column_min(df, df.columns[0]) is not None
    #assert get_column_mean(df, df.columns[0]) is not None
    #assert get_column_count_of_nan(df, df.columns[0]) is not None
    #assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    #assert get_numeric_columns(df) is not None
    #assert get_binary_columns(df) is not None
    #assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
    print("ok")
