from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_column_mean
from assignments.assignment1.c_data_cleaning import fix_nans
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, fix_outliers, fix_nans, normalize_column
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset_again, \
    process_life_expectancy_dataset

"""
Classification is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Classification
focuses in results of the categorical type.
"""

'''
NOTE: I added some print statements to help looking at the functions output for texting
I commented them out as they cause quite a lot of clutter while using them for larger functions
BUT feel free to comment any of the print statements back in while testing if it helps :)
'''


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_classifier(X: pd.DataFrame, y: pd.Series, set: str = None) -> Dict:
    """
    Simple method to create and train a random forest classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    if set == 'Amazon':
        model = RandomForestClassifier(n_estimators=5)
    else:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a classification on the iris dataset with random forest
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    rf = simple_random_forest_classifier(X, y_encoded)

    print(rf['accuracy'])
    return rf


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a classification on the iris dataset, but reusing
    the existing code from assignment1. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        # Notice that I am now passing though all columns.
        # If your code does not handle normalizing categorical columns, do so now (just return the unchanged column)
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    X, y = df.iloc[:, :4], df.iloc[:, 4]
    le = generate_label_encoder(y)

    # Be careful to return a copy of the input with the changes, instead of changing inplace the inputs here!
    y_encoded = replace_with_label_encoder(y.to_frame(), column='species', le=le)
    rf = simple_random_forest_classifier(X, y_encoded['species'])

    '''
    !!Explanation!!
    Both the classifier in this function and the one in the last yield just about the same score on average
    I believe this is because the two datasets are essentially the same at this point:
    They both have label encoded classes
    The only difference is this function removed nans and outliers, which the dataset does not possess many of anyway
    And also normalizes the dataset, which from what my understanding might not actually change the values 
    in relation to other values. This normalization may just make the model in this function more efficient!
    Due to this potential boost in efficiency due to normalization, I would choose this function's model over the last 
    '''
    print(rf['accuracy'])
    return rf


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Use the same random forest method.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_iris_dataset()
    X, y = df.iloc[:, :5], df.iloc[:, 5:]
    rf = simple_random_forest_classifier(X, y)

    '''
    !!!Explanation!!!
    There are not too many differences present, as the datasets are the same.
    The datasets are quite balanced, and the train and test are properly split so we can rule out model 
    over fitting for the most part.
    Although the labels are encoded in different ways, their meanings are not changed between models.
    The only notable difference is that the process_iris_dataset() classifier has a slightly lower score on average.
    I believe this is because the process_iris_dataset() has an additional numeric mean column. 
    This may provide extra noise to the dataset, which results in the classifier being slightly worse!
    I think this adds noise as the mean of each column doesn't really provide any new information that may benefit
    this specific classification task.
    To combat this, I believe running some feature selection and decsriptive analysis on the dataset, and 
    dropping a few of the less relevant columns may improve the model.
    A feature selection method that may prove useful here is the Pandas correlation function "corr()" - to find the 
    strength of the correlation between each feature and the target label. 
    '''
    print(rf['accuracy'])
    return rf


def decision_tree_classifier(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_classifier" but using the technique we saw in class: decision trees
    (you can use sklearn to help you).
    Optional: also optimise the parameters of the model to maximise accuracy
    :param X: Input dataframe
    :param y: Label data
    :return: model, accuracy and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    # max_features = 1
    # max_depth = 2
    # max_leaf_nodes = 2
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model
    accuracy = model.score(X_test, y_test)
    return dict(model=model, accuracy=accuracy, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the iris dataset again task of e_experimentation using the
    decision_tree classifier AND random_forest classifier. Return the one with highest score.
    Discuss (1 sentence) what you found different between the two models and scores.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset()
    X, y = df.iloc[:, :5], df.iloc[:, 5:]

    rf = simple_random_forest_classifier(X, y)
    dt = decision_tree_classifier(X, y)
    print(rf)
    print(dt)
    '''
    !!!Explanation!!!
    I may be inclined to choose the decision tree here (in this specific case) over the random forest
    Though random forests are typically known to be more accurate, this is because they take the average of many
    decision trees, rather than just one. This makes the decision tree more efficient in time and space as it requires
    only one tree, instead of many.
    In this specific instance, it seems that on average the decision tree is just as accurate as the random forest
    I believe this is due to the data set being both balanced and easily separable.
    Therefore I will take the decision tree over the random forest, 
    as the decision tree is yielding around the same accuracy on average, AND is more efficient.
    This is just for this specific function though, I think overall random forests are usually the way to go, 
    even if they require more time and resources to execute; they do solve a lot of accuracy issues the decision
    trees may have, such as overfitting.
    '''
    if rf['accuracy'] > dt['accuracy']:
        print('random forest wins')
        return rf
    else:
        print('decision tree wins')
        return dt


def train_amazon_video_game_again() -> Dict:
    """
    Run the result of the amazon dataset again task of e_experimentation using the
    decision tree classifier AND random_forest classifier. Return the one with highest score.
    The Label column is the user column. Choose what you wish to do with the time column (drop, convert, etc)
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score worse than the iris score (or why is it not worse) in your opinion?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_amazon_video_game_dataset_again()

    '''
    !!!Explanation!!!
    This is the most significant preprocess action I make
    I have decided to remove all rows that have labels that appear less than 10 times in the dataset
    I find this solves many of the issues I was having with this data set
    1. In classification, the model must train itself using the available labels in the training set, and then tests its
    performance predicting those labels with the testing set. I found as there are many unique instances in this dataset
    the model would evaluate instances that had labels which the model had not even seen before. This is problematic as 
    the model would essentially make a guess at the instance, and because it did not know the correct label, it would 
    always get it wrong. To fix the data set, it may be good to collect some data to help inflate those unique instances
    and thus balancing the dataset, or to somehow generalize labels so they are not so specific to a point where there
    are single instances with a unique label.
    2. This also significantly reduces the size of the data set, which allows the model to run efficiently without 
    sacrifices to the Decision Tree or Random Forest models. The data set is reduced to nearly half of what it used to 
    be when you remove unique instances, and even more when you only look at labels that appear at least 10 times.
    '''
    df = df.drop(df[df['user_count'] < 10].index)
    print(df)

    X, y = df.iloc[:, 1:], df.iloc[:, :1]

    '''
    !!!Explanation!!!
    I decided to drop the time column as I personally don't think it will have a correlation with the target labels.
    The time only seems to indicate the activity of the user, which is easily updates once the user reviews again.
    Thus, my theory is that the model might learn to check when a user is active, which could overfit the model if user
    activity is somewhat random.
    For example, if they reviewed a video game that came out today, after not reviewing one after 10 years,
    the model may not predict the user because it is biased to the activity dates.
    Sometimes sequels to games come out after a long, long time as any video game fan knows, and perhaps a player might
    want to review the newest sequel of a game series they used to like to review.
    I believe the model should be able to predict the user from other features relating to the users rating behaviours,
    but should be independent of time, as there are no set rules to when a user might review
    '''
    X = X.drop(['time'], axis=1)
    '''
    !!!Explanation!!!
    I decided to label encode the 'asin' data column. I believe this may be important to the models classification as
    there may be some sort of pattern between the user and the types of video games they review.
    For example, maybe user John only reviews Halo games, and never Call of Duty games.
    As this data type is a string, I needed some way to encode it. My first thought was one hot encoding but there are 
    many different 'asin' attributes, so to one hot encode that we would need to use A LOT of bits. Thus one hot 
    encoding seemed inefficient for space, thus label encoding these values seemed to be the next best option, as to the
    model the newly allocated numeric names to the 'asin' data will not change its meaning if patterns are present.
    '''
    le = LabelEncoder()
    X['asin'] = le.fit_transform(X['asin'])

    # this is here to convert shape to (n,) to prevent future warnings
    y = y.values.ravel()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    '''
    !!!Explanation!!!
    I used a special random forest compared to the others I've been using
    The default estimator size (number of trees in the forest) is 100 according to the scikit learn documentation.
    If I execute my code with that amount of estimators, my computer would run out of memory and the program crashes,
    thus after playing around with the hyper parameter of the random forest, I settled at 5 estimators. Once again, I'm
    sure the ideal number of estimators is more, but due to memory limitations I am using 5 estimators.
    '''
    rf = simple_random_forest_classifier(X, y_encoded, 'Amazon')
    print(rf)
    dt = decision_tree_classifier(X, y_encoded)
    print(dt)

    '''
    !!!Results!!!
    The decision tree is returning around a .5 accuracy score. 
    The random forest classifier is returning around the same accuracy score on average.
    This specific function takes a long time to run as there is a ton of data to be processed, even with the 
    preprocessing.

    I think there is room for overfitting here due to the duplicate values in the data set.
    This is an issue because these values may be ending up in both the training and the testing set, leading to a bias
    for that one set. It is difficult to compensate for these duplicates with the data we have, so I believe a solution
    to this may be to collect some more data relating to each specific row, perhaps more information relating to the
    users specific review for each review. These features may include some traits coming from the field of NLP, such as 
    semantic and sentiment analysis. Perhaps the model would be able to pick up on some patterns relating to how the 
    user writes, while also not being biased towards specific labels due to data duplication.
    '''
    if rf['accuracy'] > dt['accuracy']:
        print('random forest wins!')
        return rf
    else:
        print('decision tree wins!')
        return dt


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the column which has north/south. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    (check the c_regression examples to see example on how to do this one hot encoding)
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_life_expectancy_dataset()
    '''
    !!!Explanation!!!
    I dropped the year column as there are many and more Nan values within
    It is not really a value you can simply fix by average the columns that are not empty
    Logically that would not make sense, and I believe by doing that the year column would become misrepresented
    I do not predict this to affect accuracy all that much as year should not have that big of an impact on the 
    classification of the country being in the north or south, as this function is doing
    '''
    df = df.drop(['year'], axis=1)
    '''
    !!!Explanation!!!
    The expectancy column also has a lot of Nan values, so I decided to replace those Nans with the average of that 
    column. I believe this is appropriate as the life expectancy is probably around the same range for each country in
    this dataset, so taking the average of it is a good measure of the life expectancy for any country.
    Note: This hypothesis may not be great as the range of expectancy is quite large, from my preprocessing it will be 
    around 75 years; but given that some countries are developing, as well as the data being from many years ago,
    for now I believe the mean can still give a better representation than nothing!  
    '''
    mean = get_column_mean(df, 'expectancy')
    df['expectancy'].fillna(value=mean, inplace=True)
    X = df
    X = X.drop(['latitude'], axis=1)
    y = df['latitude']
    print(X)
    print(y)

    '''
    !!! Explanation !!!
    I decided to label encode the country name
    I could not leave them as strings as the model would not be able to read it, and I think one hot encoding the names
    would be very space innificient as there are many different country names, and we would need a lot of bits to 
    one hot encode them all!
    '''
    le = generate_label_encoder(X['name'])
    X['name'] = le.fit_transform(X['name'])

    rf = simple_random_forest_classifier(X, y)
    dt = decision_tree_classifier(X, y)

    '''
    !!!Explanation!!!
    Both the decision tree and the random forest are performing very well, both with ~.99 accuracy scores.
    From the results, both performed much better than any function we have classified before.
    I am inclined to believe that this data set has lead to some overfitting, due to an unbalanced dataset.
    The dataset for example, has the country Afghanistan many times, each attribute being the same as the year has been
    removed and many of the expectancy missing values are set to that columns mean.
    This introduces overfitting because the duplicate data instances may go into both the training and testing set,
    contamination!! This is not good as the model will be tested on things it already knows, giving it 100% on it 
    almost automatically... kind of like the model is cheating on a test. Given a completely brand new data set,
    I think the models performance would drop.

    Due to this data imbalance, I don't think this dataset is that great to run classification on, even with all of the 
    preprocessing. I believe a solution to this would be to of course balance out the data set, by collecting more 
    information about other countries that are less represented in the dataset, as well as add dimensions that are not 
    so redundant as missing or mean expectancies; perhaps more general features relating to the weather if we are still
    trying to predict if it is in the north or south.
    '''
    if rf['accuracy'] > dt['accuracy']:
        print('random forest wins')
        return rf
    else:
        print('decision tree wins')
        return dt


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are classification
    models, therefore it is useful only for categorical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """
    '''
    !!!My Goal!!!
    I will be using the dataset "Geography"
    With this dataset, I want to find out if we can fit a model to predict the World Bank Income Group of a country
    given a some geographical and bank related features
    To find this out, I will preprocess the data in the following ways:
        - Fix any missing data in the columns that are mentioned below
        - Extract and label encode the World Bank groups column into the labels vector 
        - Extract and one hot encode World bank region column into the features vector
        - Extract latitude into the features vector
        - Extract longitude into the features vector
    I will train both a Decision Tree and Random Forest to find my goal, and return the model with the greater accuracy
    '''
    df = pd.read_csv(Path('..', '..', 'geography.csv'))

    '''
    !!!Explanation!!!
    The only columns with Nans for the target features for this were from the Vatican, 
    so I replaced their null values with the values from Italy.
    I know they are technically separate, but until the data set can be filled we will simply consider them the same.
    '''
    df['World bank region'].fillna(value='Europe & Central Asia', inplace=True)
    df['World bank, 4 income groups 2017'].fillna('High Income', inplace=True)

    le = generate_label_encoder(df_column=df['World bank, 4 income groups 2017'])
    df = replace_with_label_encoder(df=df, column='World bank, 4 income groups 2017', le=le)

    ohe = generate_one_hot_encoder(df_column=df['World bank region'])
    df = replace_with_one_hot_encoder(df=df, column='World bank region', ohe=ohe,
                                      ohe_column_names=ohe.get_feature_names())

    columns = ['Latitude', 'Longitude', 'x0_East Asia & Pacific', 'x0_Europe & Central Asia',
               'x0_Latin America & Caribbean', 'x0_Middle East & North Africa', 'x0_North America',
               'x0_South Asia', 'x0_Sub-Saharan Africa']
    X = df[columns]
    y = df['World bank, 4 income groups 2017']

    dt = decision_tree_classifier(X=X, y=y)
    #print(dt)
    rf = simple_random_forest_classifier(X=X, y=y)
    #print(rf)
    '''
    !!!My Results!!!
    It seems that once again on average the Decision Tree and Random Forest are yielding similar results.
    Their accuracies are quite low, and range from around 50 to nearly 70 percent accuracy.
    I don't think a lot of overfitting is occurring here, as the datasets are well balanced, and properly split
    into training and testing.
    The data set does have a lack of columns that relate to the economy, wealth, or demographics of the country,
    So I believe that more data may improve the model to fit a mapping between the demographic and wealth data of a
    given country, and its income group (target label).
    Features that could be collected as additional data columns could include things such as average income, employment
    rate, tax information, and more!
    I believe although this model is just a start, it could be beneficial to companies who are figuring out economic
    policies or tax plans. I believe, the ability to use this model while trying to come up with plans to benefit a 
    country's economy could be useful, with enough relevant training and data :)
    '''
    if rf['accuracy'] > dt['accuracy']:
        #print('random forest wins')
        return rf
    else:
        #print('decision tree wins')
        return dt


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game_again() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None
