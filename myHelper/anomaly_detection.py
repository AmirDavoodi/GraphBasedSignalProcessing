from myHelper import timing
from sklearn.neighbors import LocalOutlierFactor
from myHelper import my_ged
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, fbeta_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from myHelper import My_confusion_matrix_plotter
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
from PyNomaly import loop

@timing.time_it
def ged_novelty_detection_cv(graph_stream,
                             graph_stream_labels,
                             n_process= -1):
    
    skf = StratifiedKFold(n_splits=5)
    X = np.array(list(graph_stream.keys())).reshape(-1,1)
    y = graph_stream_labels
    folds = list(skf.split(X, y))

    f1sc = make_scorer(f1_score)

    ps = {"contamination": np.linspace(0.000001, 0.25, 3),
          "n_neighbors": np.linspace(4, 200, 3, dtype = int)}
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

def vec_nov_det_cv_dis_mat(vec_dis_mat,
                           fix_heart_beats,
                           fix_hb_labels,
                           n_process= -1):
    
    skf = StratifiedKFold(n_splits=5)
    X = fix_heart_beats
    y = fix_hb_labels
    folds = list(skf.split(X, y))
    list(skf.split(fix_heart_beats, fix_hb_labels))

#     f1sc = make_scorer(f1_score)
#     fbet = make_scorer(fbeta_score, average='binary', beta=0.2)
    facur=make_scorer(accuracy_score)

    ps = {"contamination": np.linspace(0.000001, 0.5, 20, endpoint=False),
          "n_neighbors": np.linspace(4, 200, 10, dtype = int)}
#     ps = {"n_neighbors": np.linspace(4, 200, 50, dtype = int)}
    
    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
#                              contamination=0.03,
                             novelty=True,
                             n_jobs=n_process)

    clf = GridSearchCV(estimator=lof,
                       param_grid=ps,
                       scoring=facur,
                       cv=folds,
                       verbose=10,
                       n_jobs=-1)

    clf.fit(X=vec_dis_mat, y=y)

    sorted(clf.cv_results_.keys())
    optimal_lof = clf.best_estimator_
    print("\n\n This is optimal parameters:", optimal_lof, "\n\n")

    # Make predictions with our optimized envelope fit
    y_pred = optimal_lof.predict(X=vec_dis_mat)
    
    # Observations having a negative_outlier_factor smaller than offset_ are detected as abnormal.
    # The offset is set to -1.5 (inliers score around -1), except when
    # a contamination parameter different than “auto” is provided.
    print("\n Outlier threshold -(clf.offset_) = ", -(optimal_lof.offset_), "\n\n")


    # Only available for novelty detection (when novelty is set to True).
    # The shift offset allows a zero threshold for being an outlier.
    # output of (decision_function) => negative values are outliers and non-negative ones are inliers
    dec_func = optimal_lof.decision_function(X=vec_dis_mat) # X.shape = (n_test_samples, n_features)
    print("\n decision_function = ", dec_func, "\n\n")

    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
    print("\n Outlier scores for train data (1 = normal, 1 >> abnormal)= \n",
          -(optimal_lof.negative_outlier_factor_))

    # This scoring function is accessible through the score_samples method, 
    # while the threshold can be controlled by the contamination parameter.
    y_scores = optimal_lof.score_samples(X=vec_dis_mat) # X.shape = (n_test_samples, n_features)
    print("\n Score of test samples:\n", y_scores)
 
    return (y_pred, y_scores)


def ged_nov_det_cv_dis_mat(ged_dis_mat,
                           graph_stream,
                           graph_stream_labels,
                           n_process= -1):
    
    skf = StratifiedKFold(n_splits=5)
    X = np.array(list(graph_stream.keys())).reshape(-1,1)
    y = graph_stream_labels
    folds = list(skf.split(X, y))

#     f1sc = make_scorer(f1_score)
#     fbet = make_scorer(fbeta_score, average='binary', beta=0.2)
    facur=make_scorer(accuracy_score)

    ps = {"contamination": np.linspace(0.000001, 0.5, 20, endpoint=False),
          "n_neighbors": np.linspace(4, 200, 10, dtype = int)}
#     ps = {"n_neighbors": np.linspace(4, 200, 50, dtype = int)}
    
    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
#                              contamination=0.05,
                             novelty=True,
                             n_jobs=n_process)

    clf = GridSearchCV(estimator=lof,
                       param_grid=ps,
                       scoring=facur,
                       cv=folds,
                       verbose=10,
                       n_jobs=-1)

    clf.fit(X=ged_dis_mat, y=y)

    sorted(clf.cv_results_.keys())
    optimal_lof = clf.best_estimator_
    print("\n\n This is optimal parameters:", optimal_lof, "\n\n")

    # Make predictions with our optimized envelope fit
    y_pred = optimal_lof.predict(X=ged_dis_mat)
    
    # Observations having a negative_outlier_factor smaller than offset_ are detected as abnormal.
    # The offset is set to -1.5 (inliers score around -1), except when
    # a contamination parameter different than “auto” is provided.
    print("\n Outlier threshold -(clf.offset_) = ", -(optimal_lof.offset_), "\n\n")


    # Only available for novelty detection (when novelty is set to True).
    # The shift offset allows a zero threshold for being an outlier.
    # output of (decision_function) => negative values are outliers and non-negative ones are inliers
    dec_func = optimal_lof.decision_function(X=ged_dis_mat) # X.shape = (n_test_samples, n_features)
    print("\n decision_function = ", dec_func, "\n\n")

    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
    print("\n Outlier scores for train data (1 = normal, 1 >> abnormal)= \n",
          -(optimal_lof.negative_outlier_factor_))

    # This scoring function is accessible through the score_samples method, 
    # while the threshold can be controlled by the contamination parameter.
    y_scores = optimal_lof.score_samples(X=ged_dis_mat) # X.shape = (n_test_samples, n_features)
    print("\n Score of test samples:\n", y_scores)
 
    return (y_pred, y_scores)

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
def ged_out_det_dis_mat(ged_dis_mat,
                        n_process= -1,
                        n_neighbors=4):
    '''
    --------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------
    '''
    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=n_neighbors,
                             contamination=0.03,
                             novelty=False,
                             n_jobs=n_process)
    y_pred = lof.fit_predict(X=ged_dis_mat)
    
    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
    # The higher the more normal.
    y_scores = (lof.negative_outlier_factor_)
#     print("\n Outlier scores for train data (1 = normal, 1 >> abnormal)= \n", y_scores)
    print("\n Outlier threshold -(clf.offset_) = ", -(lof.offset_), "\n\n")
    
    return y_pred, y_scores

@timing.time_it
def vec_out_det_dis_mat(vec_dis_mat,
                        n_process= -1,
                        n_neighbors=200):
    '''
    --------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------------
    '''
    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=n_neighbors,
                             contamination=0.2,
                             novelty=False,
                             n_jobs=n_process)
    y_pred = lof.fit_predict(X=vec_dis_mat)
    
    #  negative_outlier_factor_ndarray shape (n_samples,)
    #  negative_outlier_factor_ = -1 normal
    #  negative_outlier_factor_ << -1 abnormal 
    # The higher the more normal.
    y_scores = (lof.negative_outlier_factor_)
#     print("\n Outlier scores for train data (1 = normal, 1 >> abnormal)= \n", y_scores)
    
    return y_pred, y_scores

def combine_OD_dis(vec_dis_mat,
                   ged_dis_mat,
                   n_process= -1,
                   vec_n_neighbors=200,
                   vec_c =0.2,
                   ged_n_neighbors=4,
                   ged_c=0.03):
    vec_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=vec_n_neighbors,
                             contamination=vec_c,
                             novelty=False,
                             n_jobs=n_process)
    vec_y_pred = vec_lof.fit_predict(X=vec_dis_mat)
    vec_y_scores = (vec_lof.negative_outlier_factor_)
    vec_threshold = vec_lof.offset_
    
    ged_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=ged_n_neighbors,
                             contamination=ged_c,
                             novelty=False,
                             n_jobs=n_process)
    ged_y_pred = ged_lof.fit_predict(X=ged_dis_mat)
    ged_y_scores = (ged_lof.negative_outlier_factor_)
    ged_threshold = ged_lof.offset_
    
    return vec_y_pred, vec_y_scores, vec_threshold, ged_y_pred, ged_y_scores, ged_threshold
    
    
    
    
    

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


def anomaly_plotter(y_true, y_pred, y_scores):
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

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from 
    # prediction scores.
    print("(ROC AUC) Score = ", roc_auc_score(y_true, y_scores))

    My_confusion_matrix_plotter.confusion_matrix_plotter(cm           = confusion_matrix(y_true, 
                                                                                         y_pred), 
                                                         normalize    = False,
                                                         target_names = ['Anonymous Heartbeat', 
                                                                         'Normal Heartbeat'],
                                                         title        = "Confusion Matrix")
    
    
def k_neigh_plotter(distance_matrix, 
                    graph_stream_labels, 
                    n_process= None,
                    contamination=0.03):
    min_NN = 2
    max_NN = 1000

    NN_F1 = []
    y_true = graph_stream_labels
    
    for try_NN in  np.linspace(min_NN, max_NN, 100, dtype = int):
        clf = LocalOutlierFactor(n_neighbors=try_NN,
                                 algorithm="brute",
                                 metric="precomputed",
                                 contamination=contamination,
                                 novelty=False,
                                 n_jobs=n_process)

        y_pred = clf.fit_predict(X=distance_matrix)
        n_errors = (y_pred != y_true).sum()
    #     print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

    #     X_scores = clf.negative_outlier_factor_

        precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, 
                                                                                 y_pred,
                                                                                 average='binary')

    #     print("F1 score on test", round(fbeta_score,4), " with num neighbors ", try_NN)
        NN_F1.append([try_NN, round(fbeta_score,4)])

    NN_F1_df = pd.DataFrame(NN_F1, columns = ['Number of neighbors', 'F1 Score'])
    fig = px.line(NN_F1_df, 
                  x="Number of neighbors", 
                  y="F1 Score", 
                  title='F1 Score vs Number of neighbors (LOF)')
    fig.show()
    
def LOOP_k_neigh_plotter(ged_dis_mat, 
                         graph_stream_labels,
                         contamination = 0.19):
    min_NN = 2
    max_NN = 1000

    NN_F1 = []
    y_true = graph_stream_labels

    for try_NN in  np.linspace(min_NN, max_NN, 100, dtype = int):
        d, idx = my_ged.k_dist_idx(ged_dis_mat,try_NN)
        m = loop.LocalOutlierProbability(distance_matrix=d, 
                                         neighbor_matrix=idx, 
                                         extent=2, 
                                         n_neighbors=try_NN, 
                                         use_numba=True, 
                                         progress_bar=True).fit()
        scores = m.local_outlier_probabilities
        y_pred = [1 if(my_score<(1-contamination)) else -1 for my_score in scores]

        n_errors = (y_pred != y_true).sum()
    #     print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

        precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, 
                                                                                 y_pred, 
                                                                                 average='binary')

    #     print("F1 score on test", round(fbeta_score,4), " with num neighbors ", try_NN)
        NN_F1.append([try_NN, round(recall,4), round(precision,4), round(fbeta_score,4)])

    NN_F1_df = pd.DataFrame(NN_F1, columns = ['Number of neighbors',
                                              'Recall (Sensitivity)',
                                              'Precision',
                                              'F1 Score'])
    
    fig = px.line(NN_F1_df, 
                  x="Number of neighbors", 
                  y="F1 Score", 
                  title='F1 Score vs Number of neighbors (LoOP)')
    fig.show()
    
def contamination_plotter(distance_matrix, 
                          graph_stream_labels, 
                          n_process= None,
                          n_neighbors=4):
    min_c = 0.000001
    max_c = 0.5

    contamination_F1 = []
    y_true = graph_stream_labels

    for try_c in  np.linspace(min_c, max_c, 100):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                                 algorithm="brute",
                                 metric="precomputed",
                                 contamination=try_c, # the proportion of outliers in the data set float 
                                 novelty=False, # novelty=True if you want to use LOF for novelty 
                                                # detection and predict on new unseen data
                                 n_jobs=n_process)

        y_pred = clf.fit_predict(X=distance_matrix)
        n_errors = (y_pred != y_true).sum()
#         print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

    #     X_scores = clf.negative_outlier_factor_

        precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, y_pred, average='binary')

#         print("F1 score is ", round(fbeta_score,4), " with  Contamination = ", try_c)
        contamination_F1.append([try_c, round(fbeta_score,4)])

    contamination_F1_df = pd.DataFrame(contamination_F1, columns = ["Contamination percentage", "F1 Score"])
    fig = px.line(contamination_F1_df, 
                  x="Contamination percentage", 
                  y="F1 Score", 
                  title='F1 Score vs Contamination (LOF)')
    fig.show()
    
def LOOP_contamination_plotter(ged_dis_mat, 
                               graph_stream_labels, 
                               n_process= -1,
                               n_neighbors=4):
    min_c = 0.000001
    max_c = 0.5

    contamination_F1 = []
    y_true = graph_stream_labels

    for try_c in  np.linspace(min_c, max_c, 100):
        d, idx = my_ged.k_dist_idx(ged_dis_mat,n_neighbors)
        m = loop.LocalOutlierProbability(distance_matrix=d, 
                                         neighbor_matrix=idx, 
                                         extent=2, 
                                         n_neighbors=n_neighbors, 
                                         use_numba=True, 
                                         progress_bar=True).fit()
        scores = m.local_outlier_probabilities
        y_pred = [1 if(my_score<(1-try_c)) else -1 for my_score in scores]


        n_errors = (y_pred != y_true).sum()
    #         print('\naccuracy =' , round(1 - (n_errors / len(y_true)),4))

        precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_true, 
                                                                                 y_pred, 
                                                                                 average='binary')

    #         print("F1 score is ", round(fbeta_score,4), " with  Contamination = ", try_c)
        contamination_F1.append([try_c, round(fbeta_score,4)])

    contamination_F1_df = pd.DataFrame(contamination_F1, columns = ["Contamination percentage", 
                                                                    "F1 Score"])
    fig = px.line(contamination_F1_df, 
                  x="Contamination percentage", 
                  y="F1 Score", 
                  title='F1 Score vs Contamination (LoOP)')
    fig.show()
    
def zero_one_label_converter(graph_stream_labels):
    y_zero_one = graph_stream_labels.copy() # new labels (0=1 and 1=-1) 
    y_zero_one[y_zero_one==1] = 0
    y_zero_one[y_zero_one==-1] = 1
    return y_zero_one