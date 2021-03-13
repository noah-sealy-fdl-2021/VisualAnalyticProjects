import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 4 columns:
    three numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    print("loading datasets from csv files...")
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)
    print("fixing missing values, outliers, and standardizing data...")
    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])
    print("calculating numeric distances...")
    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    print("calculating mean...")
    df['numeric_mean'] = distances.mean(axis=1)
    print("one hot encoding categorical columns...")
    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))
    print("done :)")
    print(df)
    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    Also, for this example, consider that all petal_widths should be between 0.0 and 1.0, replace the wong_values
    of that column with the mean of that column. Also include a new (binary) column called "large_sepal_lenght"
    saying whether the row's sepal_length is larger (true) or not (false) than 5.0
    :return: A dataframe with the above conditions.
    """
    print("loading dataset...")
    df = read_dataset(Path('..', '..', 'iris.csv'))

    """
    thought a comment here may be useful as the ordering of this answer differs from that of the question
    I chose to implement the large_sepal_length first, because we need to look at sepal lengths
    before they are normalized
    this is because we need to analyse their true length values
    rather than values that are between 0 and 1
    """
    print("generating large sepal length column...")
    large_sepal_length = []
    for i in range(len(df["sepal_length"])):
        if df.loc[i, "sepal_length"] > 5.0:
            large_sepal_length.append(True)
        else:
            large_sepal_length.append(False)
    df["large_sepal_length"] = large_sepal_length

    """
    I then chose to replace petal_widths missing values with their means before normalization
    as after normalization, all values will be between 0 and 1 regardless
    Once again, I think it's important to take a look at the true values here
    I also take the mean of the petal_widths are setting values > 1 or < 0 to np.nan
    this ensures the true values to be between 0 and 1, those which were np.nan will now be the mean
    """
    print("fixing wrong and missing petal_width values (replacing them w/ mean)...")
    df = fix_numeric_wrong_values(df, "petal_width", WrongValueNumericRule.MUST_BE_LESS_THAN, 1)
    df = fix_numeric_wrong_values(df, "petal_width", WrongValueNumericRule.MUST_BE_GREATER_THAN, 0)
    mean = get_column_mean(df, "petal_width")
    missing = df["petal_width"].isnull()
    df.at[missing, "petal_width"] = mean

    # normalization and encoding now goes along as normal
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    print("normalizing numeric columns...")
    for nc in numeric_columns:
        df.loc[:, nc] = normalize_column(df.loc[:, nc])

    print("label encoding categorical columns...")
    for cc in categorical_columns:
        le = generate_label_encoder(df.loc[:, cc])
        df = replace_with_label_encoder(df, cc, le)

    print("done :)")
    print(df)
    return df


def process_amazon_video_game_dataset():
    """
    Now use the rating_Video_Games dataset following these rules:
    1. The rating has to be between 1.0 and 5.0
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I don't care about who voted what, I only want the average rating per product,
        therefore replace the user column by counting how many ratings each product had (which should be a column called count),
        and the average rating (as the "review" column).
    :return: A dataframe with the above conditions. The columns at the end should be: asin,review,time,count
    """
    print("loading dataset...")
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    print("fixing wrong values in review columns (replacing them with mean)...")
    # used the same logic as the making sure the petal_widths were b/w 0.0 and 1.0
    df = fix_numeric_wrong_values(df, "review", WrongValueNumericRule.MUST_BE_LESS_THAN, 5.0)
    df = fix_numeric_wrong_values(df, "review", WrongValueNumericRule.MUST_BE_GREATER_THAN, 1.0)
    mean = get_column_mean(df, "review")
    missing = df["review"].isnull()
    df.at[missing, "review"] = mean

    """"
    CITATION
    this CMSDK post from July 2019 helped with this method
    I fully acknowledge and give credit to them for the code found from this article
    The article can be found at the link below
    https://cmsdk.com/python/pandas-converting-row-with-unix-timestamp-in-milliseconds-to-datetime.html
    """
    print("converting milliseconds to datetime in time column...")
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    """"
    CITATION
    this Stack Overflow post helped with this method
    the user's name was 'unutbu'
    I fully acknowledge and give credit to them for the code found in their section of the article
    The article can be found at the link below
    https://stackoverflow.com/questions/17709270/create-column-of-value-counts-in-pandas-dataframe
    """
    print("replacing user count with review count column...")
    df['user'] = df.groupby(['asin'])['user'].transform('count')
    df.rename(columns={'user': 'count'}, inplace=True)

    print("replacing review with average review per product...")
    # here I assume average rating is [product_rating] / rating_count
    df['review'] = df.groupby(['asin'])['review'].transform('sum') / df.groupby(['asin'])['review'].transform('count')

    print("done :)")
    #print(df)
    # for the new df, I figured I would not combined rows w/ the same asin value because the time was unique

    return df


def process_amazon_video_game_dataset_again():
    """
    Now use the rating_Video_Games dataset following these rules (the third rule changed, and is more open-ended):
    1. The rating has to be between 1.0 and 5.0, drop any rows not following this rule
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I just want to know more about the users, therefore show me how many reviews each user has,
        and a statistical analysis of each user (average, median, std, etc..., each as its own row)
    :return: A dataframe with the above conditions.
    """
    print("loading dataset...")
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    print("dropping 'wrong value' rows...")
    # used the same logic as the making sure the petal_widths were b/w 0.0 and 1.0
    df = fix_numeric_wrong_values(df, "review", WrongValueNumericRule.MUST_BE_LESS_THAN, 5.0)
    df = fix_numeric_wrong_values(df, "review", WrongValueNumericRule.MUST_BE_GREATER_THAN, 1.0)
    df = df[df['review'].notnull()]

    """"
    CITATION
    this CMSDK post from July 2019 helped with this method
    I fully acknowledge and give credit to them for the code found from this article
    The article can be found at the link below
    https://cmsdk.com/python/pandas-converting-row-with-unix-timestamp-in-milliseconds-to-datetime.html
    """

    print("converting milliseconds to datetime in time column...")
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    """"
    CITATION
    this Pandas Documentation for the groupby function really helped
    I'm not quite sure if I need to cite documentation but I thought I would to be safe -
    as this specific page helped me a lot
    I fully acknowledge and give credit to them for the code found from this article
    The documentation can be found at the link below
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html?highlight=transform#pandas.core.groupby.DataFrameGroupBy.transform
    """
    print("creating columns for statistical analysis...")
    # the number of reviews the user has made
    df['user_count'] = df.groupby(['user'])['review'].transform('count')
    # the user's average review score
    df['user_average'] = df.groupby(['user'])['review'].transform('sum') / df.groupby(['user'])['review'].transform('count')
    # the user's median review score
    #df['user_median'] = df.groupby(['user'])['review'].transform('median')
    # the std dev of the user's average review score
    #df['user_std'] = df.groupby(['user'])['review'].transform('std')

    print("done :)")
    print(df)
    return df


def process_life_expectancy_dataset():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    print("loading datasets from csv files...")
    expectancy = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    geography = read_dataset(Path('..', '..', 'geography.csv'))
    # for each column, I've decided to set outliers and missing values to the mean
    num_cols = get_numeric_columns(expectancy)

    # print("fixing missing values and outliers in expectancy...")
    # for i in range(len(num_cols)):
    #     # set outliers to nan
    #     expectancy = fix_outliers(expectancy, num_cols[i])
    #     # if anything is less than 5, set it to mean
    #     expectancy = fix_numeric_wrong_values(expectancy, num_cols[i], WrongValueNumericRule.MUST_BE_GREATER_THAN, 5)
    #     mean = get_column_mean(expectancy, num_cols[i])
    #     missing = expectancy[num_cols[i]].isnull()
    #     expectancy.at[missing, num_cols[i]] = mean

    # note: instead of 28 years, I just use the whole data set
    # this may take longer to process, but it may give us a better idea of the data as a whole
    # I think it's good to keep all the data as we do not have a specific future goal yet
    print("reformatting expectancy...")
    countries = []
    years = []
    expectancies = []
    for i in range(len(expectancy['country'])):
        for n in range(len(num_cols)):
            countries.append(expectancy['country'][i])
            years.append(num_cols[n])
            expectancies.append(expectancy.loc[i, num_cols[n]])
    new_expectancy = pd.DataFrame()
    new_expectancy['name'] = countries # name the column name so it will match w/ geography dataset
    new_expectancy['year'] = years
    new_expectancy['expectancy'] = expectancies

    print("merging new expectancy and geography...")
    merged = pd.merge(new_expectancy, geography, how="outer", on='name')

    """"
    CITATION
    this Stack Overflow post helped me with this method
    the user's name is 'cs95'
    I fully acknowledge and give credit to them for the code found from their section in this article
    The documentation can be found at the link below
    https://stackoverflow.com/questions/16616141/keep-certain-columns-in-a-pandas-dataframe-deleting-everything-else#16616454
    """

    print("dropping all columns except country, continent, year, value, and latitude...")
    keep = ['name', 'four_regions', 'year', 'expectancy', 'Latitude', 'Longitude']
    drop = merged.columns.difference(keep)
    merged = merged.drop(drop, axis=1)

    # I'm assuming negative latitude means south, as it will be south of the equator
    # and positive means north, as it will be north of the equator
    # print("changing latitude to categorical and putting it through label encoder...")
    # cat_lat = []
    # for i in range(len(merged['Latitude'])):
    #     if merged.loc[i, 'Latitude'] > 0:
    #         cat_lat.append("North")
    #     else:
    #         cat_lat.append("South")
    # merged = merged.drop('Latitude', axis=1)
    # merged['latitude'] = cat_lat
    #
    # le = generate_label_encoder(merged.loc[:, 'latitude'])
    # merged = replace_with_label_encoder(merged, 'latitude', le)

    # print("one hot encoding continent...")
    # ohe = generate_one_hot_encoder(merged['four_regions'])
    # merged = replace_with_one_hot_encoder(merged, 'four_regions', ohe, ohe.get_feature_names())

    print("done :)")
    #print(merged)
    return merged


if __name__ == "__main__":
    assert process_iris_dataset() is not None
    assert process_iris_dataset_again() is not None
    assert process_amazon_video_game_dataset() is not None
    assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
