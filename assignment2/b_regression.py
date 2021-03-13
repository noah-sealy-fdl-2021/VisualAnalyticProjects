from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_iris_dataset_again, \
    process_amazon_video_game_dataset, process_life_expectancy_dataset

"""
Regression is a supervised form of machine learning. It uses labeled data, which is data with an expected
result available, and uses it to train a machine learning model to predict the said result. Regression
focuses in results of the numerical type.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_random_forest_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # If necessary, change the n_estimators, max_depth and max_leaf_nodes in the below method to accelerate the model training,
    # but don't forget to comment why you did and any consequences of setting them!
    model = RandomForestRegressor()  # Now I am doing a regression!
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)  # Use this line to get the prediction from the model

    # In regression, there is no accuracy, but other types of score. See the following link for one example (R^2)
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.score
    score = model.score(X_test, y_test)
    return dict(model=model, score=score, test_prediction=y_predict)


def simple_random_forest_on_iris() -> Dict:
    """
    Here I will run a regression on the iris dataset with a random-forest regressor
    Notice that my logic has changed. I am not predicting the species anymore, but
    am predicting the sepal_length. I am also removing the species column, and will handle
    it in the next example.
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    X, y = df.iloc[:, 1:4], df.iloc[:, 0]
    model = simple_random_forest_regressor(X, y)
    print(model)
    return model


def reusing_code_random_forest_on_iris() -> Dict:
    """
    Again I will run a regression on the iris dataset, but reusing
    the existing code from assignment1. I am also including the species column as a one_hot_encoded
    value for the prediction. Use this to check how different the results are (score and
    predictions).
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    '''
    !!!My Results!!!
    When comparing the two regression functions together, it seems that the raw data will typically have a higher 
    accuracy compared to the processed data, but they are essentially the same.
    I believe this is due to the same reasons I mentioned in the classification file, the data is essentially the same,
    the only preprocessing which was done fixed the outliers and removed the missing values in the dataset, but this
    dataset does not actually have too many outliers or nans, thus we are left with a very similar dataset.
    The normalization and one hot encoding changes in the dataset in a way that processing it with a model may be more
    efficient, but it will not actually change the meaning of the data! 
    So, although the datasets are essentially the same, I would still choose to use the preprocessed data, as the 
    normalization and one hot encoding may make the model process the data more efficiently compared to just the
    raw data.
    '''
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    model = simple_random_forest_regressor(X, y)
    print(model)
    return model


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def random_forest_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation and discuss (1 sentence)
    the differences from the above results. Also, as above, use sepal_length as the label column and
    the one_hot_encoder to transform the categorical column into a usable format.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    '''
    !!!Explanation!!!
    It seems that with this dataset, the model performs with much better accuracy compared to the dataset in the 
    previous function. I believe the key difference between this data set and the previous data set is the addition
    of the large petal length column. Although this column only added noise while predicting a feature like the 
    flowers species (discussed in the classification file), I believe it may be making a benefit to the model while
    trying to predict something like sepal_length. This may be due to the correlation between the sepal_length and 
    the large_sepal_length feature, as they are closely related.
    I think it makes sense that this data set would have a better accuracy as it provides more information into the 
    data we are trying to predict! I discussed this in a few places in the classification file, but I believe it is 
    generally important to include relevant data to what is being predicted in order to have a well trained model!
    We see that here, as the large sepal length feature is relevant to he length of the sepal.
    '''
    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    model = simple_random_forest_regressor(X, y)
    print(model)
    return model


def decision_tree_regressor(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Reimplement the method "simple_random_forest_regressor" but using
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.DecisionTreeRegressor.html
    Optional: also optimise the parameters of the model to minimise the R^2 score
    '''
    correction: we want R^2 to be maximized (see Leo's comment on this thread in the VA Teams)
    '''
    :param X: Input dataframe
    :param y: Label data
    :return: model, score and prediction of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    score = model.score(X_test, y_test)

    return dict(model=model, score=score, test_prediction=y_predict)


def train_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation, but now using the
    decision tree regressor AND random_forest regressor. Return the one with lowest R^2.
    Use the same label column and one hot encoding logic as before.
    Discuss (1 sentence) what you found different between the results.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_iris_dataset_again()

    ohe = generate_one_hot_encoder(df['species'])
    df = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    dt = decision_tree_regressor(X, y)
    print(dt)

    rf = simple_random_forest_regressor(X, y)
    print(rf)

    '''
    Given the results, it is evident that on average the Decision Tree and Random Forest regressors yield similar 
    accuracy scores, it is hard to say which is the clear winner.
    This is very similar to the instance of comparing the two on the processed iris set from the classification file,
    I believe the iris data set is quite well balanced, so the improvements and reduction in overfitting the 
    random forest model has to offer over the decision tree is somewhat mitigated. 
    I would say becuase of the this, the decision tree has an advantage over the random forest model, as it is more
    efficient in both time and space (uses only one tree compared to many).
    I would like to note as I did in the classification example, that I would usually choose a random forest over a 
    decision tree as they have many benefits related to improving accuracy through reducing overfitting, but in this
    case those issues are not present, thus the decision tree wins my pick!
    '''
    if rf['score'] > dt['score']:
        print('random forest wins!')
        return rf
    else:
        print('decision tree wins!')
        return dt


def train_amazon_video_game() -> Dict:
    """
    Run the result of the amazon dataset task of e_experimentation using the
    decision tree regressor AND random_forest regressor. Return the one with lowest R^2.
    The Label column is the count column
    Discuss (1 sentence) what you found different between the results.
    In one sentence, why is the score different (or the same) compared to the iris score?
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """
    df = process_amazon_video_game_dataset()
    print(df)

    '''
    !!!Explanation!!!
    I used this same logic in the classification file, but we would need many and more bits to represent a one hot
    encoding of all of the movie names, so instead I just label encode them, though they still represent the same data
    '''
    le = generate_label_encoder(df['asin'])
    df = replace_with_label_encoder(df, 'asin', le)

    '''
    !!!Explanation!!!
    Note: this is the same explanation I gave for dropping the time column in the classification file, but I 
    thought I would restate it here just in case.
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
    df = df.drop('time', axis=1)

    X, y = df.iloc[:, 1:], df.iloc[:, 0]

    rf = simple_random_forest_regressor(X, y)
    print(rf)

    dt = decision_tree_regressor(X, y)
    print(dt)

    '''
    !!!My Results!!!
    Using the original data with little processing other than encoding, we get a models that have scores of around 
    0.9999 on average. This may look very good but I don't think there is any way this could be realistic.
    I think there are two pretty big issues with this data set, both of which I discuss in the classification file
    but they seem even more apparent here so I will reiterate my analysis.
    Issue #1: Unique Labels.
        - A unique label refers to an instance of data in the data set that possesses a label, which no other instances
        in that dataset possess. I believe this is an issue in regression and classification as if that instance is put
        in a test set, the label will not be learned by the model. This becomes an issue as it will be automatically 
        wrongly predicted. 
        - I believe a solution to this would be to simply choose better labels; ones that can describe the data in a 
        more general way. I believe the count feature is problematic as it is somewhat trivial (you can just count the
        number of reviews for each user) and without an idea of that counting pattern, it will be somewhat of a guess!
    Issue #2: Data Duplication.
        - Although it seems this may contradict issue #1, I think there should be a nice mix of data that have the same 
        labels for supervised learning, but are different, I think this is the basis of a balanced data set.
        - The issue here is that a lot of data instances in this data set are extremely similar to others, apart from 
        the video game being reviewed. This can lead to overfitting of the model.
        - This leads to overfitting of the model as these duplicates can show up in both the training set AND the test
        set, leading to contamination of the sets. I've talked about this a lot throughout the assignment but the model
        will learn the instances in training, and then be tested on those same instances for the score function. This is
        really bad because the score now misrepresents the models ability to predict actually new data.
        - I don't think its as an easy fix as issue #1, but I think one solution may be more data collection; features
        that are not just unique to the class, but also unique to the individual instances must be collected! This data
        diversity in theory should reduce a lot of the contamination, and thus reduce the overfiting in the model.
    '''

    '''
    !!!Versus the Iris Data Set!!!
    I think the only thing to say here should be obvious from my responses to the iris function and this one. The big
    differences between these results is that there is evident overfitting going on with this data set, and not too much
    going on with the iris set. 
    I believe the biggest differences of these data sets (if we were to somehow standardize them relative to each other)
    is that the iris data set is a lot more balanced, and does not contain as much potential for train/test
    contamination. I believe if you balanced out the data and fix the duplication issues (as listed above) in the 
    Amazon data set, it could have the potential to have a model learn it just as well as it learns the iris data set.
    Although the Amazon data sets models had better scores, I believe that is just due to various instances of 
    overfitting, and thus the more balanced Iris data set produced better models.
    '''

    if rf['score'] < dt['score']:
        print('random forest wins!')
        return rf
    else:
        print('decision tree wins!')
        return dt


def train_life_expectancy() -> Dict:
    """
    Do the same as the previous task with the result of the life expectancy task of e_experimentation.
    The label column is the value column. Remember to convert drop columns you think are useless for
    the machine learning (say why you think so) and convert the remaining categorical columns with one_hot_encoding.
    Feel free to change your e_experimentation code (changes there will not be considered for grading
    purposes) to optimise the model (e.g. score, parameters, etc).
    """

    df = process_life_expectancy_dataset()

    '''
    !!!Explanation!!!
    This code below is interesting, and it aims to fix several Nan values that existed as expectancy values
    I originally intended to replace the expectancy with the mean of the column as I did in the classification file,
    BUT this cause the issue of a large number of data instances having the same target value, this resulted in 
    regressors with very low scores. This wasn't good so I needed to come up with a new solution and I decided to 
    drop each row which had a Nan expectancy value, and simply work with the data that we had.
    You'll see that this resulted in a model with a very high score, and I will discuss it more in the results section.

    Additionally this drops various instances which had year set to Nan, I did not think I could represent date 
    by replacing the Nan values with anything from the other instances of the Year column, so I would have dropped those
    values anyway.
    '''
    df['expectancy'].fillna(value=0, inplace=True)
    df = df[df['expectancy'] != 0]
    print(df)

    le = generate_label_encoder(df['name'])
    df = replace_with_label_encoder(df, 'name', le)
    print(df)

    X = df
    X = X.drop(['expectancy'], axis=1)
    y = df['expectancy']

    rf = simple_random_forest_regressor(X=X, y=y)
    print(rf)

    dt = decision_tree_regressor(X=X, y=y)
    print(dt)

    '''
    !!!My Results!!!
    The models perform very well! BUT I think this may be due to the model being overfit. It seems that there is some
    risk of contamination with this data set. There are many instances which are essentially the same, except for the 
    year value. These values would risk being split into both the training and testing dataset, and the model might
    train and learn one instance just to be tested on what is a very, very similar version of that instance.
    It is like the model is cheating! And I do not predict it would do so well on new, real life data.

    In order to reduce risk of overfitting, while also not removing any more instances, I would collect more data in
    order to include information specific to the year and the country, this way the model would have more information
    about the demographics and conditions the country held and could perhaps find patterns to predict life expectancy
    that way.

    To quickly conclude my observation on dropping expectancy values from before,
    I do not think setting the expectancy values should have been set to the mean as they were what was being predicted,
    and by extracting a value for the prediction that way causes the model to misrepresent the real data I think.
    In order to include the dropped data in future tests, we would simply have to find the actual data for those 
    instances, as I do not believe we should be using false values for such an important feature! To do this, we could
    search through government papers or statistical analyses for the target country. Once we have the complete data
    profile of that country, then I would feel more comfortable adding it back into the data set.
    '''

    if rf['score'] > dt['score']:
        print('random forest wins!')
        return rf
    else:
        print('decision tree wins!')
        return dt


def your_choice() -> Dict:
    """
    Now choose one of the datasets included in the assignment1 (the raw one, before anything done to them)
    and decide for yourself a set of instructions to be done (similar to the e_experimentation tasks).
    Specify your goal (e.g. analyse the reviews of the amazon dataset), say what you did to try to achieve the goal
    and use one (or both) of the models above to help you answer that. Remember that these models are regression
    models, therefore it is useful only for numerical labels.
    We will not grade your result itself, but your decision-making and suppositions given the goal you decided.
    Use this as a small exercise of what you will do in the project.
    """
    '''
    !!!My Goal!!!
    I will be using the dataset Iris Dataset
        I think this dataset is relatively simple, and very well balanced; making it super fun to use because
        the resulting models so far have seemed quite robust. 
    With this dataset, I want to run a regression which predicts the petal width of a given iris flower. 
    To find this out, I will preprocess the data in the following ways:
        - Extract petal_width for use as the target feature
        - Extract and one hot encode the species column to use for the feature vector
        - Extract the petal length column to use for the feature vector
        - Let's say for this specific problem, the iris flower researcher student forgot to measure any information
        regarding the sepal - oops! I want to see how robust of a model I can make using just those two features listed.
        - I think plan to make a 
    I will train a decision tree model for this problem, as we saw before I do not think I will need something as
    powerful as a random forest as this dataset is very well balanced.
    I do however want to see the difference in accuracy score when I include the sepal information into the dataset, 
    say the student's mentor went back to record that information to complete the data set (after firing the student).
    Thus I will use two decision trees and compare them at the end, to see if leaving out the sepal info will have any
    effect! I will return the one with the better score.
    Let's see if the mentor made the right decision in firing the student ;)
    '''
    df = pd.read_csv(Path('..', '..', 'iris.csv'))

    ohe = generate_one_hot_encoder(df_column=df['species'])
    df = replace_with_one_hot_encoder(df=df, column='species', ohe=ohe, ohe_column_names=ohe.get_feature_names())

    y = df['petal_width']

    # student collected features and model
    columns = ['petal_length', 'x0_setosa', 'x0_versicolor', 'x0_virginica']
    x_student = df[columns]
    dt_student = decision_tree_regressor(X=x_student, y=y)

    # mentor collected features and model
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'x0_setosa', 'x0_versicolor', 'x0_virginica']
    x_mentor = df[columns]
    dt_mentor = decision_tree_regressor(X=x_mentor, y=y)

    print(dt_student)
    print(dt_mentor)
    '''
    !!!My Results!!!
    On average, the models seem to have about the same score, around .88 to .93.
    I believe this is a real score as the dataset is balanced and there should not be any contamination between training
    and testing sets.
    With that being said, the students model has few features in its data set, with just about the same score. I believe
    this may support the theory that in order to predict petal width, data relating to the petal is very important.

    Although upon further manipulation of the dataset, it seems that another feature which is truly crucial for is the 
    species of the flower. The robustness of the model seems to be very dependent on the presence of this feature as 
    well. Thus the petal_length and species feature are very important for this problem! 

    I think I would still rehire the student, as they were able to make a model that scored just as well, but used a 
    smaller dataset (for this specific problem), and thus actually saved the lab a little bit of space, and perhaps a 
    whole lot of time collecting data on iris flowers. The work from the student also questioned the data set, and 
    allowed the "lab" to find out which features are the most important for this problem.
    I would rehire the student and give him a raise!!
    '''
    if dt_student['score'] > dt_mentor['score']:
        print('rehire the student!')
        return dt_student
    else:
        print('keep him fired!')
        return dt_mentor


if __name__ == "__main__":
    assert simple_random_forest_on_iris() is not None
    assert reusing_code_random_forest_on_iris() is not None
    assert random_forest_iris_dataset_again() is not None
    assert train_iris_dataset_again() is not None
    assert train_amazon_video_game() is not None
    assert train_life_expectancy() is not None
    assert your_choice() is not None
