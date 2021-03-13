from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.d_data_encoding import fix_outliers, fix_nans, normalize_column, \
    generate_one_hot_encoder, replace_with_one_hot_encoder, generate_label_encoder, replace_with_label_encoder
from assignments.assignment1.e_experimentation import process_iris_dataset, process_amazon_video_game_dataset, \
    process_amazon_video_game_dataset_again, process_life_expectancy_dataset

"""
Clustering is a non-supervised form of machine learning. It uses unlabeled data
 through a given method, returns the similarity/dissimilarity between rows of the data.
 See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def simple_k_means(X: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(X)
    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(X, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    df = pd.read_csv(Path('..', '..', 'iris.csv'))
    for c in list(df.columns):
        df = fix_outliers(df, c)
        df = fix_nans(df, c)
        df[c] = normalize_column(df[c])

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(df.iloc[:, :4])

    ohe = generate_one_hot_encoder(df['species'])
    df_ohe = replace_with_one_hot_encoder(df, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(df['species'])
    df_le = replace_with_label_encoder(df, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    print(no_species_column['score'], no_binary_distance_clusters['score'], labeled_encoded_clusters['score'])
    ret = no_species_column
    if no_binary_distance_clusters['score'] > ret['score']:
        print('no binary distance')
        ret = no_binary_distance_clusters
    if labeled_encoded_clusters['score'] > ret['score']:
        print('labeled encoded')
        ret = labeled_encoded_clusters
    return ret


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(X: pd.DataFrame, eps: float, min_samples: float) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """

    '''
    !!!Explanation!!!
    I chose to use the DBScan model for my choice of a clustering algorithm.
    I choose this for two reasons:
    1. This algorithm was discussed in class, and seems very interesting! Thus I was somewhat inspired from that
    discussion to try it out :)
    2. I was especially excited to test out this algorithm in the Amazon Video Game Dataset, I found that particular
    data set struggled a lot in the Regression and Classification tasks due to some data set issues I've already 
    discussed in length in the previous files. I think that data set might be better suited for an unsupervised task.
    DBScan may work especially well for that unsupervised task as I've read in the documentation that DBScan is able to 
    handle large data sets with a lot of noise very well.
    Note about the above points, while writing them I have not implemented the below methods, so technically they are
    just hypotheses, I will probably not rewrite this part given the results, so it will be interesting to see if my
    predictions came true, and if the DBScan was able to work well with the Amazon data set.

    I also added some parameters to the model which would allow me to tune the model differently for each individual 
    function, these parameters include eps and min_samples, which are both hyper parameters for the DBScan model.
    '''

    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit(X)

    #print('Labels: ' + str(len(set(model.labels_))))
    #print(model.labels_)
    #print(model)
    #print(clusters)

    score = metrics.silhouette_score(X, model.labels_)

    return dict(model=model, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the result of the process iris again task of e_experimentation through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    df = process_iris_dataset()

    '''
    !!!Explanation!!!
    I remove the labels as this is unsupervised learning, thus we do not use labels.
    I found the score went up nearly 10% when I removed these labels, and just clustered based on the sepal and
    petal data.
    '''

    x = df.iloc[:, :4]

    '''
    !!!Explanation!!!
    I'm keeping this here just to show how I tested hyperparameters.
    I thought it would be an easier method than graphing the data to find for the elbow plot because we have more
    than 2 dimensions. This is essentially the same, just perhaps a little messier.
    The model that I return is the model that turned out to have the best results from these tests.
    I will discuss that model in My Results section.
    '''

    eps = [0.6, 0.7, 0.8, 0.9]
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i in eps:
        for n in mins:
            model = custom_clustering(X=x, eps=i, min_samples=n)
            #print('--------------')
            #print(model['model'])
            #print(model['score'])

    '''
    !!!My Results!!!
    My most successful model yields a score of around 0.58, identifying 4 labels (the 3 data labels and a noise label). 
    I figured this was an realistic score given the model.
    I believe this is a realistic result because DBScan is known to being effective by the concept mentioned in lecture
    called The Curse of Dimensionality. When more dimensions are added, the accuracy of the model is reduced. 
    From what I've read in the documentation and the lecture, I believe DBScan is especially effected by the Curse of
    Dimensionality, so a dataset like this with even just four dimensions and three known labels can be difficult to 
    cluster.
    I believe another issue here is that each data instance is pretty close together, regardless of label. The data
    points which were especially close regardless of label were the points that belonged in ___ and ____. Due to this,
    I understand how the DBScan clustering algorithm may have placed some of those super close points into incorrect 
    neighbourhoods. I believe for the score to increase, the exact perfect epsilon value would have to be found, so the
    DBScan model could define neighbourhoods (thus clusters) perfectly! Due to the high dimensionality of the data set,
    as discussed, this would require a lot of time and resources devoted to hyper parameter tuning, which might end with
    the realization that for this problem, a classification model might just be better! 
    '''
    return custom_clustering(X=x, eps=0.9, min_samples=3)  # 0.58


def cluster_amazon_video_game() -> Dict:
    """
    Run the result of the process amazon_video_game task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    '''
    !!!My Goals!!!
    This question was a bit vague to me, so I've decided that with my clusters I want to use this data set to begin to
    find a method of investigating any trends or interesting behaviours between a products average review, and how many
    users have reviewed that given product. I think this method would be valuable to companies, which I discuss more in
    the My Results section of this function. The TA, Asal, helped me with the thought process of the overall idea of this
    function as well as a few of the techniques within, so I must cite her here and thank her for her help :)
    '''

    df = process_amazon_video_game_dataset()

    '''
    !!!Explanation!!!
    This data set holds data regarding the number of reviews and the average reviews PER product. That means each data
    attribute is specific to that particular product. That also means that each row of the data set involving that 
    product, or with the same 'asin' id (which is the field we drop the duplicates by) are the same! Thus there is a
    ton of data duplication in this data set, and thus we can remove the redundant data and save a bunch of space.
    When we drop the duplicates, the rows go from around one million to around fifty thousand! This saves a lot of 
    time and space for our clustering models.
    '''

    df = df.drop_duplicates(subset=['asin'])

    '''
    !!!Explanation!!!
    Here I drop both the 'time' and 'asin' columns from the data set. 
    Obviously, the 'time' column has no relevance to the data now that we've dropped duplicates.
    As for the 'asin' column, I do not think it is very important here as it does not make much sense to use it for 
    what I am trying to investigate. I want to look at if the average rating is affected by the amount of times users
    rate it, and the movie id itself is not very relevant, if we are looking generally for all movies; especially if
    we need to normalize each column to ready it for PCA, thus the 'asin' column gets dropped!
    '''

    df = df.drop('time', axis=1)
    df = df.drop('asin', axis=1)

    '''
    !!!Explanation!!!
    I decided to use the PCA dimensionality reduction technique as DBScan falls victim to the Curse of Dimensionality,
    I decided to bring my data set down to 1 dimension. From what I understand PCA projects this 2d data to a one 
    dimensional plane, choosing priority on the more important "component". 

    I needed to normalize my data as well to properly execute the PCA, which is shown below as well. The Z-score
    normalization method I use will in a way scale the data to be consistent with each other. In a way, they will now
    be able to "speak the same language". The data will be normalized to both fit a distribution of mean=0 and 
    std_dev=1. It is important for the data to be on the "same scale" as if not, the PCA algorithm will run the risk
    of denoting whichever column simply has the larger values as the "principal component".
    '''

    df['count'] = normalize_column(df_column=df['count'])
    df['review'] = normalize_column(df_column=df['review'])
    pca = PCA(n_components=1)
    projected = pca.fit_transform(X=df)

    '''
    !!!My Results!!!
    The commented out section of code is a sample of how I chose my eps and min_samples hyper parameters.
    The final model which I return is the one I chose to use. I am typically getting around a score of 0.75 with it.
    I believe this is a more believable result compared to the classification or regression, I think this data set
    is more suited for a clustering algorithm, especially one like DBScan which is good at handling noise. I think at
    the end of the day it is important to have a clear goal of what you want to do with a model, and without concrete
    labels or at least more features to avoid duplication, it was difficult to process this data with a supervised
    model, but at least clustering is able to show off some relationships that may be evident between review counts 
    and average rating for each product.

    I think an extension of this investigation may be looking into review trends for products on Amazon, both the 
    average rating and number of reviews, but also data such as total number of purchases per product. This would 
    perhaps show if customers are actually purchasing a product and leaving good reviews, or if the company is just
    hiring people to leave good reviews on their product on the Amazon website; which could perhaps lead to detection
    of phony reviews and scams. This may be valuable to a company as they can improve their overall quality assurance.
    '''

    '''
    eps = [0.0001, 0.0002, 0.000302, 0.0004, 0.0005]
    mins = [20, 30, 40, 50, 60, 100]
    for i in eps:
        for n in mins:
            model = custom_clustering(X=projected, eps=i, min_samples=n)
            print(model)
    '''

    model = custom_clustering(X=projected, eps=0.000302, min_samples=20)  # gives close to 0.76 score
    return model


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the result of the process amazon_video_game_again task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    '''
    !!!My Goal!!!
    Similar to the last function, I am going to try to develop this function into a more concrete idea in order to focus
    my investigation.
    This data set relates to user information, rather than product information, thus I would like to see if there are
    trends for each user review as the number of reviews the user gives differs. 
    I believe this can be valuable to a company as it gives the start to some user data analysis, once again a company 
    could use something like this in order to develop a method of tracking users, with additional purchasing data they
    may be able to see the difference of what they are reviewing and simply just purchasing, and the frequency of it. 
    This could give the company data to push specific products towards specific users, while also potentially gaining
    information about bots or scammer accounts within their site.
    I think we will need to use users that only have a certain amount of reviews and more, as too few reviews might not
    really amount to any useful data. This is a working hyper parameter, that perhaps the company would choose and tune,
    but for now I will only look at users who have more than 4 reviews. Conveniently, this cuts down the data size
    quite a lot, as I was having difficulties with too much data otherwise.
    Due to the nature of the investigation, I believe this function will be quite similar to the previous function, 
    thus I will be reusing a lot of the code found there, and refer to the previous function during my explanations.
    '''

    df = process_amazon_video_game_dataset_again()

    '''
    !!!Explanation!!!
    Dropping the user duplicates for the same reasons I dropped the asin duplicates previously.
    '''

    df = df.drop_duplicates(subset=['user'])

    '''
    !!!Explanation!!!
    Dropping columns that are not relevant to the investigation, or will not make sense for clustering.
    '''
    df = df.drop('user', axis=1)
    df = df.drop('asin', axis=1)
    df = df.drop('review', axis=1)
    df = df.drop('time', axis=1)

    '''
    !!!Explanation!!!
    As mentioned in the description, only choose users that have more than 4 reviews
    '''

    df = df[df['user_count'] > 4]
    print(df)

    '''
    !!!Explanation!!!
    Normalize then PCA, same reasons as previously.
    '''

    df['user_count'] = normalize_column(df_column=df['user_count'])
    df['user_average'] = normalize_column(df_column=df['user_average'])
    pca = PCA(n_components=1)
    projected = pca.fit_transform(X=df)

    '''
    !!!The Limitations!!!
    I believe for the scope of this specific investigation, we are limited from the features we have. We would be able
    to know more about trends of users if we had some insight into their purchase trends. That way we would be able to
    have a better idea if they were reviewing products that they bought, and what then we would be able to even relate 
    that data back to the previous function and find if a products average review is similar to the user review, and 
    then perhaps try to find out the reason why differences occur. With this data we would also be able to recommend 
    other products the company has to offer, based on those results.

    Another limitation is simply the computational power of my laptop! There is a lot of data here, and a lot that I had
    to cut out for efficiency of the model. Say we had more computers to stream the model on, maybe a few nice GPUs, we
    could process a lot more data a lot faster!
    '''

    '''
    !!!My Results!!!
    WIth the data that I was able to run the model on, using the parameters that I selected I was able to get the 
    running model to yield a score of 0.82. As mentioned before, I believe DBScan is a much better form of fitting
    to this data set compared to the supervised classifiers and regressors, as this data set has a lot of noise and 
    outliers, while also being very large with somewhat ambiguous labels in places.

    This seems to be good, and I think overall this is a good start to an 
    overall larger product of putting users into clusters based on activity, in order to extract information which
    can lead the company to recommend other products to the 'real' users, based on their previous purchases and reviews.
    The overall program could also spot 'fake' users, bots and scammers who may be hyping up a product even though they
    will not actually purchase it. I believe both of listed functionality is very good for a company who is trying to 
    sell products online as they will be able to learn more about their customers, and target their products depending
    on what they've learned. I believe overall these two functions begin to show the power of data science for 
    businesses and company's who are able to collect data on their customers.
    '''

    model = custom_clustering(X=projected, eps=0.000302, min_samples=20)
    print(model)

    return model


def cluster_life_expectancy() -> Dict:
    """
    Run the result of the process life_expectancy task of e_experimentation through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    '''
    !!!My Goal!!!
    I would like to cluster the year, expectancy, and latitude.
    I believe this may show some trends in age expectancy with respect to the location of each country
    I think it's important to consider location, as some parts of the world are much more developed than others,
    thus may have varying life expectancies.
    '''

    df = process_life_expectancy_dataset()

    '''
    !!!Explanation!!!
    As expectancy cannot be made up, I simply remove the entries where expectancy is Nan.
    This only reduces the data set by around 10 rows.
    '''

    df['expectancy'].fillna(value=0, inplace=True)
    df = df[df['expectancy'] != 0]

    '''
    !!!Explanation!!!
    As labels are not required for an unsupervised task such as DBScan clustering, I remove their columns from the 
    data set.
    '''

    df = df.drop(['x0_africa'], axis=1)
    df = df.drop(['x0_americas'], axis=1)
    df = df.drop(['x0_asia'], axis=1)
    df = df.drop(['x0_europe'], axis=1)

    '''
    !!!Explanation!!!
    Conceptually I do not think the name column makes much sense to include in the cluster.
    We are trying to cluster locations based on age expectancy, any information the algorithm needs relating to location
    will come from the latitude data. The name is simply a label that we put on the Latitude to make it easier, but 
    will mean little to the DBScan algorithm, thus I remove it. It does not provide any additional information that 
    may help this investigation.
    '''

    df = df.drop(['name'], axis=1)

    '''
    !!!Explanation!!!
    This investigation will be independent of time, as time for the most part is irrelevant.
    '''

    df = df.drop(['year'], axis=1)

    '''
    !!!Explanation!!!
    I use the same normalization and PCA technique as before. 
    The normalization is to standardize the data all onto the same scale (mean=0,std=1)
    The PCA is to reduce the dimensionality of the data, by projecting the data onto one dimension
    This will all make the model run more efficiently
    '''

    df['expectancy'] = normalize_column(df_column=df['expectancy'])
    df['latitude'] = normalize_column(df_column=df['latitude'])

    pca = PCA(n_components=1)
    projected = pca.fit_transform(X=df)

    '''
    !!!My Results!!!
    I was able to get working cluster which yielded the score of 0.99. The commented out code below is the method I
    used to find my hyper parameters, the model I've returned use the hyper parameters that seem to yield the best
    results.

    I believe that clustering data based on location is an effective way of showing trends and correlations in respect
    to geography. For example, if a few places are all clustered together, it may indicate that those places all have
    a lower or higher life expectancy. This could lead to an investigation on finding constant factors that may 
    correlate with either having a high or low average expectancy. This information could be extremely useful for 
    international policies, in order to increase overall life expectancy around the world.

    The limitation here is, as always, lack of data! I believe a few more fields representing metrics which represent
    demographic and economic details of the country could improve this investigation. I believe a lot of this data can 
    also be found in the geography data set provided in the assignment folder, a good start at least. I think if I were 
    to continue with this investigation, I would integrate that data into this and see if the clusters would yield 
    more information regarding the factors that affect life expectancy around the world.
    '''

    '''
    eps = [0.1, 0.001, 0.0001]
    mins = [5, 10, 15]
    for i in eps:
        for n in mins:
            model = custom_clustering(X=projected, eps=i, min_samples=n)
            print(model)
    '''

    model = custom_clustering(X=projected, eps=0.1, min_samples=15)
    return model


if __name__ == "__main__":
    # iris_clusters()
    # assert cluster_iris_dataset_again() is not None
    # assert cluster_amazon_video_game() is not None
    # assert cluster_amazon_video_game_again() is not None
    assert cluster_life_expectancy() is not None
