from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from assignments.assignment1.b_data_profile import get_numeric_columns, get_text_categorical_columns
from assignments.assignment1.c_data_cleaning import normalize_column

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS, MeanShift
from sklearn.decomposition import PCA
from sklearn import metrics


"""
COMPETITION EXTRA POINTS!!
The below method should:
1. Handle any dataset (if you think worthwhile, you should do some preprocessing)
2. Generate a model based on the label_column and return the one with best score/accuracy

The label_column may indicate categorical column as label, numerical column as label or it can also be None
If categorical, run through these ML classifiers and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or a NaiveBayes
If numerical, run through these ML regressors and return the one with lowest R^2: 
    DecisionTree, RandomForestRegressor, KNeighborsRegressor or a Gaussian NaiveBayes
If None, run through at least 4 of the ML clustering algorithms in https://scikit-learn.org/stable/modules/clustering.html
and return the one with highest silhouette (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

Optimize all parameters for the given score above. Feel free to choose any method you wish to optimize these parameters.
Write as a comment why and how you decide your model_type/parameter combination.
This method should not take more than 10 minutes to finish in a desktop Ryzen5 (or Core i5) CPU (no gpu acceleration).  

We will run your code with a separate dataset unknown to you. We will call the method more then once with said dataset, measuring
all scores listed above. The 5 best students of each score type will receive up to 5 points (first place->5 points, second->4, and so on).
Finally, the top 5 students overall (with most points in the end) will be awarded a prize!
"""


def generate_model(df: pd.DataFrame, label_column: Optional[str]) -> Dict:

    num_cols = get_numeric_columns(df=df)
    cat_cols = get_text_categorical_columns(df=df)

    if label_column is None:
        print('clustering')

        '''
        !!!Preprocessing!!!
        This applies to all processing throughout this file.
        I tried to reduce the data sets to data that can run on these very general data sets.
        An example of this may be removing all text columns for clustering, as there are more cases where this is a 
        good idea than it is not.
        Similar to how you'll see me use my models, I don't think this file will work very well for any data set, but 
        perhaps ok for a lot of data sets. I discuss this more with the models but there is truly no free lunch.
        '''

        # drop categorical columns
        for i in cat_cols:
            df = df.drop(i, axis=1)

        # drop nans
        for i in df.columns:
            df[i].fillna(value=0, inplace=True)
            df = df[df[i] != 0]

        # scale everything down for PCA
        for i in num_cols:
            df[i] = normalize_column(df_column=df[i])

        '''
        !!!Explanation!!!
        Using PCA for dimensionality reduction!
        by using n_components = 0.90, we are keeping 90% of the datasets variance within the amount of features
        projected will have 
        '''
        pca = PCA(n_components=0.90)
        projected = pca.fit_transform(X=df)

        '''
        !!!Models!!!
        This applies for all of my model sections.
        THERE IS NO FREE LUNCH!! 
        Not expecting these models to run perfectly, or even well for every single data set you throw at them.
        Though I did try to make them capable of being as general as possible. 
        For example, this includes setting my trees to be very shallow!
        I think if I can make sure no model is becoming too specific (like a deep tree would), I can perhaps swing at
        least an average of a generally low score for all my models, which for the possibility of any data set ever
        being thrown at it, I would be happy with those results. 
        For this reason, I chose my models based on trying to cover as much ground as I can. For example, with 
        classification I know Naive Bayes may not be great in some cases, that is why the other models are there; but 
        for cases where Naive Bayes is very useful, it will be there to shine! I believe trying to cover as many
        general data sets as possible, the files may perform mediocre on any data set you may throw at it (within 
        reason lol), once again I don't think there could be a file such as this that could run 99% on any dataset...
        and if someone does find it I think instead of submitting it for marks they should sell it for billions ;)
        I think the keyword here is generality!!
        '''

        '''
        !!!DBScan!!!
        '''

        eps = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        mins = [10, 15, 20, 30]

        db_scores = []
        for i in eps:
            for n in mins:
                model = DBSCAN(eps=.2, min_samples=5)
                clusters = model.fit(projected)
                score = metrics.silhouette_score(projected, model.labels_)
                db_scores.append(dict(model=model, score=score))

        best_db = dict(model=None, score=0)
        for i in range(len(db_scores)):
            if db_scores[i]['score'] > best_db['score']:
                best_db['score'] = db_scores[i]['score']
                best_db['model'] = db_scores[i]['model']

        '''
        MeanShift
        '''

        ms_scores = []
        bands = [2, 4, 6, 8, 10]
        for i in bands:
            model = MeanShift(bandwidth=i)
            clusters = model.fit(projected)
            score = metrics.silhouette_score(projected, model.labels_)
            ms_scores.append(dict(model=model, score=score))

        best_ms = dict(model=None, score=0)
        for i in range(len(db_scores)):
            if ms_scores[i]['score'] > best_ms['score']:
                best_ms['score'] = ms_scores[i]['score']
                best_ms['model'] = ms_scores[i]['model']

        '''
        OPTICS
        '''

        o_scores = []
        eps = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        mins = [10, 15, 20, 30]
        for i in eps:
            for n in mins:
                model = OPTICS(min_samples=i, max_eps=n)
                clusters = model.fit(projected)
                score = metrics.silhouette_score(projected, model.labels_)
                o_scores.append(dict(model=model, score=score))

        best_o = dict(model=None, score=0)
        for i in range(len(db_scores)):
            if o_scores[i]['score'] > best_o['score']:
                best_o['score'] = o_scores[i]['score']
                best_o['model'] = o_scores[i]['model']

        '''
        Hierarchical 
        '''

        hier_scores = []
        aff = ['euclidean', 'cosine', 'l1', 'l2', 'manhatten']
        for i in aff:
            model = AgglomerativeClustering(affinity=i)
            clusters = model.fit(projected)
            score = metrics.silhouette_score(projected, model.labels_)
            hier_scores.append(dict(model=model, score=score))
        best_h = dict(model=None, score=0)

        for i in range(len(db_scores)):
            if hier_scores[i]['score'] > best_h['score']:
                best_h['score'] = hier_scores[i]['score']
                best_h['model'] = hier_scores[i]['model']

        '''
        Return Best Cluster!
        '''

        best_scores = np.array([best_db['score'], best_ms['score'], best_o['score'], best_h['score']])
        best = best_scores.max()

        if best == best_h['score']:
            print(best_h)
            return best_h
        elif best == best_ms['score']:
            print(best_ms)
            return best_ms
        elif best == best_o['score']:
            print(best_o)
            return best_o
        else:
            print(best_db)
            return best_db

    elif label_column in num_cols:
        print('regressor')

        '''
        !!!Processing!!!
        '''

        # label encode non numeric
        for i in cat_cols:
            le = LabelEncoder()
            df[i] = le.fit_transform(df[i])

        # replace nans with mean for each column
        for i in df.columns:
            df[i].fillna(value=df[i].mean(), inplace=True)

        # partition df to x and y
        y = df[label_column]
        df = df.drop(label_column, axis=1)
        x = df

        '''
        !!!Models!!!
        '''

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
        '''
        !!!Decision Tree Regressor!!!
        '''
        dt_scores = []
        depths = [4, 6, 8, 10]
        splits = [4, 6, 8, 10]
        impurity = [0.2, 0.3, 0.4]
        for i in depths:
            for n in impurity:
                for m in splits:
                    model = DecisionTreeRegressor(max_depth=i, min_impurity_decrease=n, min_samples_split=m)
                    model.fit(X_train, y_train)
                    y_predict = model.predict(X_test)
                    score = model.score(X_test, y_test)
                    dt_scores.append(dict(model=model, score=score))

        best_dt = dict(model=None, score=0)
        for i in range(len(dt_scores)):
            if dt_scores[i]['score'] > best_dt['score']:
                best_dt['score'] = dt_scores[i]['score']
                best_dt['model'] = dt_scores[i]['model']

        '''
        !!!Random Forest Regressor!!!
        '''

        rf_scores = []
        estimators = [50, 70, 100, 120]
        for q in estimators:
            for i in depths:
                for n in impurity:
                    for m in splits:
                        model = RandomForestRegressor(n_estimators=q, max_depth=i, min_impurity_decrease=n,
                                                      min_samples_split=m)
                        model.fit(X_train, y_train)
                        y_predict = model.predict(X_test)
                        score = model.score(X_test, y_test)
                        rf_scores.append(dict(model=model, score=score))


        best_rf = dict(model=None, score=0)
        for i in range(len(rf_scores)):
            if rf_scores[i]['score'] > best_rf['score']:
                best_rf['score'] = rf_scores[i]['score']
                best_rf['model'] = rf_scores[i]['model']

        '''
        !!!KNeighbours Regressor!!!
        '''

        neighs = [3, 6, 9, 12, 15]
        kn_scores = []
        for i in neighs:
            model = KNeighborsRegressor(n_neighbors=i)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            score = model.score(X_test, y_test)
            kn_scores.append(dict(model=model, score=score))

        best_kn = dict(model=None, score=0)
        for i in range(len(kn_scores)):
            if kn_scores[i]['score'] > best_kn['score']:
                best_kn['score'] = kn_scores[i]['score']
                best_kn['model'] = kn_scores[i]['model']

        '''
        !!!Return Best!!!
        '''

        best_scores = np.array([best_dt['score'], best_rf['score'], best_kn['score']])
        best = best_scores.max()

        print(best_scores)

        if best == best_kn['score']:
            print(best_kn)
            return best_kn
        elif best == best_dt['score']:
            print(best_dt)
            return best_dt
        else:
            print(best_rf)
            return best_rf

    elif label_column in cat_cols:

        print('label_column is categorical!')

        '''
        !!!Preprocessing!!!
        '''

        # label encode non numeric
        for i in cat_cols:
            le = LabelEncoder()
            df[i] = le.fit_transform(df[i])

        # replace nans with mean for each column
        for i in df.columns:
            df[i].fillna(value=df[i].mean(), inplace=True)

        # partition df to x and y
        y_encoded = df[label_column]
        df = df.drop(label_column, axis=1)
        x = df

        '''
        !!!Models!!!
        '''

        '''
        !!!Explanation and Citation!!!
        In order to average out multiple instances of the model, I use k-fold cross validation for training and testing
        of the model.
        I used the from cross_val_score function from scikit learn model selection library to do this
        I used this Youtube tutorial to learn how to effectively use this method.
        https://www.youtube.com/watch?v=gJo0uNL-5Qw.
        Thus I would like to cite the Youtube user "codebasics" from Jan. 26, 2019 for helping me figure out this 
        library.
        '''

        '''
        Decision Tree
        '''

        dt_scores = []
        depths = [4, 6, 8, 10]
        splits = [4, 6, 8, 10]
        impurity = [0.2, 0.3, 0.4]
        for i in depths:
            for n in impurity:
                for m in splits:
                    model = DecisionTreeClassifier(max_depth=i, min_impurity_decrease=n, min_samples_split=m)
                    score = np.average(cross_val_score(model, x, y_encoded))
                    dt_scores.append(dict(model=model, score=score))

        best_dt = dict(model=None, score=0)
        for i in range(len(dt_scores)):
            if dt_scores[i]['score'] > best_dt['score']:
                best_dt['score'] = dt_scores[i]['score']
                best_dt['model'] = dt_scores[i]['model']

        '''
        Random Forest
        '''

        rf_scores = []
        estimators = [50, 70, 100, 120]
        for q in estimators:
            for i in depths:
                for n in impurity:
                    for m in splits:
                        model = RandomForestClassifier(n_estimators=q, max_depth=i, min_impurity_decrease=n,
                                                       min_samples_split=m)
                        score = np.average(cross_val_score(model, x, y_encoded))
                        rf_scores.append(dict(model=model, score=score))

        best_rf = dict(model=None, score=0)
        for i in range(len(rf_scores)):
            if rf_scores[i]['score'] > best_rf['score']:
                best_rf['score'] = rf_scores[i]['score']
                best_rf['model'] = rf_scores[i]['model']

        '''
        K-Neighbours
        '''

        neighs = [3, 6, 9, 12, 15]
        kn_scores = []
        for i in neighs:
            model = KNeighborsClassifier(n_neighbors=i)
            score = np.average(cross_val_score(model, x, y_encoded))
            kn_scores.append(dict(model=model, score=score))

        best_kn = dict(model=None, score=0)
        for i in range(len(kn_scores)):
            if kn_scores[i]['score'] > best_kn['score']:
                best_kn['score'] = kn_scores[i]['score']
                best_kn['model'] = kn_scores[i]['model']

        '''
        GaussianNB
        '''

        gnb_scores = []
        model = GaussianNB()
        score = np.average(cross_val_score(model, x, y_encoded))

        gnb_scores.append(dict(model=model, score=score))
        best_gnb = gnb_scores[0]

        '''
        Return Best Classifier!
        '''

        best_scores = np.array([best_dt['score'], best_rf['score'], best_kn['score'], best_gnb['score']])
        best = best_scores.max()

        if best == best_gnb['score']:
            print(best_gnb)
            return best_gnb
        elif best == best_kn['score']:
            print(best_kn)
            return best_kn
        elif best == best_dt['score']:
            print(best_dt)
            return best_dt
        else:
            print(best_rf)
            return best_rf

    else:
        print('Error: label_column is not valid.')
        print('It must be relate to a column in df that is of categorical data type, numeric data type, ' +
              'or is simply None. Returning empty dict object.')
        return dict(model=None, final_score=None)


'''
!!!Explanation!!!
I used this main function for testing purposes
I left in some code, very general stuff for running the scripts, feel free to delete it or modify it as required :)
Thanks!
'''
if __name__ == "__main__":

    csv = ''
    df = pd.read_csv(Path('..', '..', csv))

    assert generate_model(df=df, label_column=None) is not None
