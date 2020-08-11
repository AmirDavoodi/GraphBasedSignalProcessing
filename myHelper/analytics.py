from myHelper import analytics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve
from PyNomaly import loop
from myHelper import my_ged
import plotly.graph_objects as go

def my_pre_recall(ged_dis_mat,
                  graph_stream_labels,
                  y_zero_one,
                  n_neighbors,
                  contamination = 0.03,
                  n_process = -1):
    d, idx = my_ged.k_dist_idx(ged_dis_mat,n_neighbors)
    m = loop.LocalOutlierProbability(distance_matrix=d, 
                                     neighbor_matrix=idx, 
                                     extent=2, 
                                     n_neighbors=n_neighbors, 
                                     use_numba=True, 
                                     progress_bar=True).fit()
    scores = m.local_outlier_probabilities

    # average Precision-recall score
    LOOP_average_precision = average_precision_score(graph_stream_labels, scores)
    # calculate model precision-recall curve
    LOOP_precision, LOOP_recall, _ = precision_recall_curve(y_zero_one, scores)


    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=n_neighbors,
                             contamination=contamination,
                             novelty=False,
                             n_jobs=n_process)
    y_pred = lof.fit_predict(X=ged_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    local_out_factors = -lof.negative_outlier_factor_
    lof_threshold = -lof.offset_
    normalized = (local_out_factors-min(local_out_factors))/(max(local_out_factors)- \
                                                             min(local_out_factors))
    normalized_thre = (lof_threshold-min(local_out_factors))/(max(local_out_factors)- \
                                                              min(local_out_factors))
    # Precision-recall score
    LOF_average_precision = average_precision_score(graph_stream_labels, normalized)
    # calculate model precision-recall curve
    LOF_precision, LOF_recall, _ = precision_recall_curve(y_zero_one, normalized)




    # # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_zero_one[y_zero_one==1]) / len(y_zero_one)
    
    fig = go.Figure()
    

    # plot the LOOP model precision-recall curve
    fig.add_trace(go.Scatter(x=LOOP_recall, y=LOOP_precision,
                        mode='lines+markers',
                        name='LoOP Avg {0:0.4f}'.format(LOOP_average_precision)))
    
    # plot the LOF model precision-recall curve
    fig.add_trace(go.Scatter(x=LOF_recall, y=LOF_precision,
                        mode='lines+markers',
                        name='LOF Avg {0:0.4f}'.format(LOF_average_precision)))
    # plot the no skill precision-recall curve
    fig.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill],
                        name='No Skill', line = dict(color='green', width=4, dash='dash')))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title="Implementations",
    )

    fig.show()
    
    
def my_roc_curve(ged_dis_mat,
                 graph_stream_labels,
                 y_zero_one,
                 n_neighbors,
                 contamination = 0.03,
                 n_process = -1):
    d, idx = my_ged.k_dist_idx(ged_dis_mat,n_neighbors)
    m = loop.LocalOutlierProbability(distance_matrix=d, 
                                     neighbor_matrix=idx, 
                                     extent=2, 
                                     n_neighbors=n_neighbors, 
                                     use_numba=True, 
                                     progress_bar=True).fit()
    scores = m.local_outlier_probabilities

    # calculate roc auc score
    LOOP_roc_auc = roc_auc_score(y_zero_one, scores)
    # calculate roc curve for model
    LOOP_fpr, LOOP_tpr, _ = roc_curve(y_zero_one, scores)
    
    lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=n_neighbors,
                             contamination=contamination,
                             novelty=False,
                             n_jobs=n_process)
    y_pred = lof.fit_predict(X=ged_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    local_out_factors = -lof.negative_outlier_factor_
    lof_threshold = -lof.offset_
    normalized = (local_out_factors-min(local_out_factors))/(max(local_out_factors)- \
                                                             min(local_out_factors))
    normalized_thre = (lof_threshold-min(local_out_factors))/(max(local_out_factors)- \
                                                              min(local_out_factors))

    # calculate roc auc
    LOF_roc_auc = roc_auc_score(y_zero_one, normalized)
    # calculate roc curve for model
    LOF_fpr, LOF_tpr, _ = roc_curve(y_zero_one, normalized)

    
    fig = go.Figure()
    

    # plot roc curve for LoOP model 
    fig.add_trace(go.Scatter(x=LOOP_fpr, y=LOOP_tpr,
                        mode='lines+markers',
                        name='LoOP Avg ROC-AUC score {0:0.3f}'.format(LOOP_roc_auc)))
    
    # plot roc curve for LOF model 
    fig.add_trace(go.Scatter(x=LOF_fpr, y=LOF_tpr,
                        mode='lines+markers',
                        name='LOF Avg ROC-AUC score {0:0.3f}'.format(LOF_roc_auc)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        name='No Skill', line = dict(color='green', width=4, dash='dash')))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Implementations",
    )

    fig.show()