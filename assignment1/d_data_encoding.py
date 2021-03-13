import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


'''
NOTE: I added some print statements to help looking at the functions output for texting
I commented them out as they cause quite a lot of clutter while using them for larger functions
BUT feel free to comment any of the print statements back in while testing if it helps :)
'''


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    """
    This method should generate a (sklearn version of a) label encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    encoder = LabelEncoder()
    encoder.fit_transform(df_column)
    #print("Encoded! " + str(list(df_column)) + " --> " + str(list(encoder.transform(df_column))))
    return encoder


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    """
    This method should generate a (sklearn version of a) one hot encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    one_hot = OneHotEncoder()
    col = np.array(df_column)
    col = col.reshape(-1, 1)
    one_hot.fit(col)
    return one_hot


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should replace the column of df with the label encoder's version of the column
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to replace the column
    :return: The df with the column replaced with the one from label encoder
    """
    le.fit_transform(df[column])
    df[column] = le.transform(df[column])
    #print(df)
    return df


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    """
    This method should replace the column of df with all the columns generated from the one hot's version of the encoder
    Feel free to do it manually or through a sklearn ColumnTransformer
    :param df: Dataset
    :param column: column to be replaced
    :param ohe: the one hot encoder to be used to replace the column
    :param ohe_column_names: the names to be used as the one hot encoded's column names
    :return: The df with the column replaced with the one from label encoder
    """
    """
    CITATION
    this Stack Overflow post helped with this method
    the user's name was 'Amine'
    I fully acknowledge and give credit to them for the code found from their section in this article
    The article can be found at the link below
    https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
    """
    transformed = ohe.transform(np.array(df[column]).reshape(-1, 1)).toarray()
    # turn new data into a pd.df
    ohe_df = pd.DataFrame(transformed, columns=ohe_column_names)
    # merge the two data frames, dropping the one we don't need
    data = pd.concat([df, ohe_df], axis=1).drop([column], axis=1)
    #print(data)
    return data


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_label_encoder
    The column of df should be from the label encoder, and you should use the le to revert the column to the previous state
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to revert the column
    :return: The df with the column reverted from label encoder
    """
    df[column] = le.inverse_transform(df[column])
    #print(df)
    return df


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_one_hot_encoder
    The columns (one of the method's arguments) are the columns of df that were placed there by the OneHotEncoder.
    You should use the ohe to revert these columns to the previous state (single column) which was present previously
    :param df: Dataset
    :param columns: the one hot encoded columns to be replaced
    :param ohe: the one hot encoder to be used to revert the columns
    :param original_column_name: the original column name which was used before being replaced with the one hot encoded version of it
    :return: The df with the columns reverted from the one hot encoder
    """
    """
    CITATION
    this Stack Overflow post helped with this method
    the user's name was 'Amine'
    I fully acknowledge and give credit to them for the code found from their section in this article
    The article can be found at the link below
    https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
    """
    # transform data back
    original = ohe.inverse_transform(df[columns])
    # make new pd.df
    inverse_df = pd.DataFrame(original, columns=[original_column_name])
    # merge the two data frames, dropping the ones we no longer need
    data = pd.concat([df, inverse_df], axis=1).drop(columns, axis=1)
    #print(data)
    return data


if __name__ == "__main__":
    """"
    NOTE: I switched around the order of these so the label encoder does its thing, and then the one hot encoder does
    I also commented out some of the duplicate function calls
    I found with the duplicates there would be some weird behaviour that did not show off the full functionality -
    of the encoders
    They still work if you have them all uncommented though, just doesn't show them off fully :)
    """
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    le = generate_label_encoder(df.loc[:, 'c'])
    assert le is not None
    #assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_label_encoder_with_original_column(replace_with_label_encoder(df, 'c', le), 'c', le) is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    assert ohe is not None
    #assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    assert replace_one_hot_encoder_with_original_column(replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())),
                                                        list(ohe.get_feature_names()),
                                                        ohe,
                                                        'c') is not None
    print("ok")
