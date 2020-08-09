from myHelper import timing
from sklearn.neighbors import LocalOutlierFactor
from myHelper import my_ged
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from myHelper import My_confusion_matrix_plotter
from matplotlib import pyplot as plt

@timing.time_it
def ged_novelty_detection_cv(graph_stream,
                             graph_stream_labels,
                             n_process= -1):
    
    skf = StratifiedKFold(n_splits=5)
    X = np.array(list(graph_stream.keys())).reshape(-1,1)
    y = graph_stream_labels
    folds = list(skf.split(X, y))

    f1sc = make_scorer(f1_score)

    ps = {"contamination": np.linspace(0.000001, 0.25, 10),
          "n_neighbors": np.linspace(4, 10, 4, dtype = int)}
    lof = LocalOutlierFactor(algorithm="brute",
                             metric=my_ged.ged_distance,
                             metric_params={"graph_stream": graph_stream},
                             novelty=True,
                             n_jobs=-1)

    clf = GridSearchCV(estimator=lof,
                       param_grid=ps,
                       scoring=f1sc,
                       cv=folds,
                       verbose=10,
                       n_jobs=-1)

    clf.fit(X, y)


    sorted(clf.cv_results_.keys())

    optimal_lof = clf.best_estimator_

    # Make predictions with our optimized envelope fit
    y_pred = optimal_lof.predict(X)
 
    return (y_pred)

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


def anomaly_plotter(y_true, y_pred):
    n_errors = (y_pred != y_true).sum()
    print('\nNumber of errors = ', n_errors)
    print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

    # F1 Score
    #print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))
    precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, 
                                                                             y_pred, 
                                                                             average='binary')
    print("precision ", round((precision), 4))
    print("recall ", round((recall), 4))
    print("F1 score on Test", round((fbeta_score), 4))

    My_confusion_matrix_plotter.confusion_matrix_plotter(cm           = confusion_matrix(y_true, 
                                                                                         y_pred), 
                                                         normalize    = False,
                                                         target_names = ['Anonymous Heartbeat', 
                                                                         'Normal Heartbeat'],
                                                         title        = "Confusion Matrix")
    
    
def k_neigh_plotter(distance_matrix,
                    graph_stream_labels
                    X):
    minRE = 2
    maxRE = 100

    EpsF1 = []
    y_true = graph_stream_labels

    for TryRE in range(minRE,maxRE,1):
        clf = LocalOutlierFactor(n_neighbors=TryRE,
                                 algorithm="brute",
                                 metric="precomputed",
                                 contamination='auto', # the proportion of outliers in the data set float 
                                 novelty=False, # novelty=True if you want to use LOF for novelty 
                                                # detection and predict on new unseen data
                                 n_jobs=None)

        y_pred = clf.fit_predict(X=distance_matrix)
        n_errors = (y_pred != y_true).sum()
    #     print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

    #     X_scores = clf.negative_outlier_factor_

        precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, y_pred, average='binary')

    #     print("F1 score on test", round(fbeta_score,4), " with num neighbors ", TryRE)
        EpsF1.append([TryRE, round(fbeta_score,4)])

    EpsF1df = pd.DataFrame(EpsF1, columns = ['NumNeighb', 'F1'])

    EpsF1df.plot.line("NumNeighb","F1")
    plt.xlim(2, 40)
    plt.title("F1 vs NumNeighb")
    plt.show()