from myHelper import timing
from sklearn.neighbors import LocalOutlierFactor
from myHelper import my_ged
import numpy as np
from sklearn.model_selection import train_test_split

@timing.time_it
def ged_novelty_detection(graph_stream,
                          X_test,
                          X_train,
                          n_process= -1,
                          n_neighbors = 4):
    '''
    
    '''
    clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                             algorithm="brute",
                             metric=my_ged.ged_distance,
                             metric_params={"graph_stream": graph_stream},
                             contamination='auto', # the proportion of outliers in the data set float 
                             novelty=True, # novelty=True if you want to use LOF for novelty 
                                           # detection and predict on new unseen data
                                           # 
                                           # (novelty = true) you should only use predict,
                                           # decision_function and score_samples on new unseen 
                                           # data and not on the training set.  
                             n_jobs=n_process)
#     print("Train data shape =", X_train.shape)
#     print("Test data shape =", X_test.shape)

    # Fit the model using X as training data.
    clf.fit(X=X_train)

    prediction = clf.predict(X=X_test) # X.shape = (n_test_samples, n_features)
#     print("\n predictions are \n", prediction)

    # Observations having a negative_outlier_factor smaller than offset_ are detected as abnormal.
    # The offset is set to -1.5 (inliers score around -1), except when
    # a contamination parameter different than “auto” is provided.
#     print("\n Outlier threshold = ", -(clf.offset_))


    # Only available for novelty detection (when novelty is set to True).
    # The shift offset allows a zero threshold for being an outlier.
    # output of (decision_function) => negative values are outliers and non-negative ones are inliers
#     dec_func = clf.decision_function(X=X_test) # X.shape = (n_test_samples, n_features)
#     print("\n decision_function = ", dec_func)

    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
#     print("\n Outlier scores for train data (1 = normal, 1 >> abnormal)= \n",
#           -(clf.negative_outlier_factor_))

    # This scoring function is accessible through the score_samples method, 
    # while the threshold can be controlled by the contamination parameter.
#     scores = clf.score_samples(X=X_test) # X.shape = (n_test_samples, n_features)
#     print("\n Score of test samples:\n", scores)
    
    return (prediction)

@timing.time_it
def ged_outlier_detection(graph_stream,
                          X_train,
                          n_process= -1,
                          n_neighbors = 4):
    '''
    
    '''
    clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                             algorithm="brute",
                             metric=my_ged.ged_distance,
                             metric_params={"graph_stream": graph_stream},
                             contamination='auto', # the proportion of outliers in the data set float 
                             novelty=False, # novelty=True if you want to use LOF for novelty 
                                           # detection and predict on new unseen data
                                           # 
                                           # (novelty = true) you should only use predict,
                                           # decision_function and score_samples on new unseen 
                                           # data and not on the training set.  
                             n_jobs=n_process)

    # predict on training data
    # Fits the model to the training set X and predict the labels.
    outlier_prediction = clf.fit_predict(X=X_train)

    # Observations having a negative_outlier_factor smaller than offset_ are detected as abnormal.
    # The offset is set to -1.5 (inliers score around -1), except when
    # a contamination parameter different than “auto” is provided.
#     print("\n Outlier threshold = ", -(clf.offset_))

    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
#     print("\n Outlier scores (1 = normal, 1 >> abnormal)= \n \n", -(clf.negative_outlier_factor_))
    return (outlier_prediction)

def my_train_test_split(graph_stream,
                        true_labels,
                        test_size=0.33,
                        random_state=42):
    
    X = np.array(list(graph_stream.keys())).reshape(-1,1)
    y = true_labels
    (X_train, X_test, y_train, y_test) = train_test_split(X, 
                                                          y, 
                                                          test_size=0.33, 
                                                          random_state=42)
    print("X_train shape = ", X_train.shape)
    print("X_test shape = ", X_test.shape)
    return X_train, X_test, y_train, y_test

def get_all_data(graph_stream, true_labels):
    X_train = np.array(list(graph_stream.keys())).reshape(-1,1)
    print("X_train shape = ", X_train.shape)
    return X_train, true_labels