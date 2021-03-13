import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


'''
NOTE: I added some print statements to help looking at the functions output for texting
I commented them out as they cause quite a lot of clutter while using them for larger functions
BUT feel free to comment any of the print statements back in while testing if it helps :)
'''


class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    """
    CITATION
    this Youtube video helped with this method
    the user's name was Corey Schafer
    I fully acknowledge and give credit to them for the code found from their section in this article
    The article can be found at the link below
    https://www.youtube.com/watch?v=Lw2rlcxScZY
    """
    if must_be_rule == must_be_rule.MUST_BE_POSITIVE:
        #print("must be positive")
        df.at[df[column] < 0, column] = np.nan
        #print(df[column])
        return df
    elif must_be_rule == must_be_rule.MUST_BE_NEGATIVE:
        #print("must be negative")
        df.at[df[column] > 0, column] = np.nan
        #print(df[column])
        return df
    elif must_be_rule == must_be_rule.MUST_BE_GREATER_THAN:
        #print("must be greater than " + str(must_be_rule_optional_parameter))
        df.at[df[column] < must_be_rule_optional_parameter, column] = np.nan
        #print(df[column])
        return df
    elif must_be_rule == must_be_rule.MUST_BE_LESS_THAN:
        #print("must be less than " + str(must_be_rule_optional_parameter))
        df.at[df[column] > must_be_rule_optional_parameter, column] = np.nan
        #print(df[column])
        return df
    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """
    """
    CITATION
    this Machine Learning Mastery post helped with this method
    the user's name was Jason Brownlee
    I fully acknowledge and give credit to them for the code from their section in this article
    The article can be found at the link below
    https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
    """

    nums_cols = get_numeric_columns(df)
    bin_cols = get_binary_columns(df)
    cat_cols = get_text_categorical_columns(df)

    if column in nums_cols:
        std_dev = np.std(df[column])
        upper = get_column_mean(df, column) + 3 * std_dev
        lower = get_column_mean(df, column) - 3 * std_dev
        for i in range(len(df[column])):
            item = df.iloc[i][column]
            if (item > upper) or (item < lower):
                df.at[i, column] = np.nan
        return df
    elif column in bin_cols:
        return df
    elif column in cat_cols:
        return df

    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    nums_cols = get_numeric_columns(df)
    bin_cols = get_binary_columns(df)
    cat_cols = get_text_categorical_columns(df)
    if column in nums_cols:
        missing = df[column].isnull()
        df.at[missing, column] = 0
        return df
    elif list(column) in bin_cols:
        missing = df[column].isnull()
        df.at[missing, column] = False
        return df
    elif list(column) in cat_cols:
        missing = df[column].isnull()
        df.at[missing, column] = "N/A"
        return df
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    """"
    CITATION
    this Stack Overflow post helped with this method
    the user's name was 'Cina'
    I fully acknowledge and give credit to them for the code from their section in this article
    The article can be found at the link below
    https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    """
    # min max normalization method is used
    # I chose this method as I knew some values may be missing or np.nan
    if is_numeric_dtype(df_column):
        df_norm = abs((df_column - df_column.max()) / (df_column.max() - df_column.min()))
        return df_norm
    return df_column


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    # z-score standardization will be used here
    # this will standardize the scores, setting the population mean to 0, and the standard deviation to 1
    df_norm = (df_column - df_column.mean())/df_column.std()
    #print(df_norm)
    #print("Standardized mean: " + str(df_norm.mean())) # should = ~0
    #print("Standardized std_dev: " + str(df_norm.std())) # should = ~1
    return df_norm


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    if distance_metric == DistanceMetric.EUCLIDEAN:
        euc = pow(pow(df_column_1 - df_column_2, 2), 1/2)
        #print(euc)
        return euc
    elif distance_metric == DistanceMetric.MANHATTAN:
        man = abs(df_column_1 - df_column_2)
        #print(man)
        return man
    return df_column_1


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    # I used element wise Jaccard's distance
    # The formulas were found in the lecture slides
    jaccard = 1 - ((df_column_1 & df_column_2) / 1 + 1 - (df_column_1 & df_column_2))
    #print(jaccard)
    return jaccard


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    #assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    #assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    #assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    #assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    #assert fix_outliers(df, 'c') is not None
    #assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    #assert standardize_column(df.loc[:, 'a']) is not None
    #assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    #assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    #assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
