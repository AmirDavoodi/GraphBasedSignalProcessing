from myHelper import analytics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve
from PyNomaly import loop
from myHelper import my_ged
import plotly.graph_objects as go
import os

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

def combine_pre_recall(vec_dis_mat,
                       ged_dis_mat,
                       hb_labels,
                       y_zero_one,
                       vec_n_neighbors=200,
                       vec_c =0.2,
                       ged_n_neighbors=4,
                       ged_c=0.03,
                       n_process = -1):
    #compute combine
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
    
    combine_score = vec_y_scores+ged_y_scores
    combine_threshold = vec_threshold + ged_threshold
    combine_y_pred = [-1 if(score<combine_threshold) else 1 for score in combine_score]
    
    com_local_out_factors= -combine_score
    com_lof_threshold = -combine_threshold
    com_normalized = (com_local_out_factors-min(com_local_out_factors))/(max(com_local_out_factors)- \
                                                             min(com_local_out_factors))
    # Precision-recall score
    com_LOF_average_precision = average_precision_score(hb_labels, com_normalized)
    # calculate model precision-recall curve
    com_LOF_precision, com_LOF_recall, _ = precision_recall_curve(y_zero_one, com_normalized)


    vec_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=vec_n_neighbors,
                             contamination=vec_c,
                             novelty=False,
                             n_jobs=n_process)
    vec_y_pred = vec_lof.fit_predict(X=vec_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    vec_local_out_factors = -vec_lof.negative_outlier_factor_
    vec_lof_threshold = -vec_lof.offset_
    vec_normalized = (vec_local_out_factors-min(vec_local_out_factors))/(max(vec_local_out_factors)- \
                                                             min(vec_local_out_factors))
    # Precision-recall score
    vec_LOF_average_precision = average_precision_score(hb_labels, vec_normalized)
    # calculate model precision-recall curve
    vec_LOF_precision, vec_LOF_recall, _ = precision_recall_curve(y_zero_one, vec_normalized)

    # # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_zero_one[y_zero_one==1]) / len(y_zero_one)
    
    fig = go.Figure()
    

    # plot the LOOP model precision-recall curve
    fig.add_trace(go.Scatter(x=com_LOF_recall, y=com_LOF_precision,
                        mode='lines+markers',
                        name='Combine Avg {0:0.4f}'.format(com_LOF_average_precision)))
    
    # plot the LOF model precision-recall curve
    fig.add_trace(go.Scatter(x=vec_LOF_recall, y=vec_LOF_precision,
                        mode='lines+markers',
                        name='Vec LOF Avg {0:0.4f}'.format(vec_LOF_average_precision)))
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
    
def sig_pre_recall(vec_dis_mat,
                   sig_dis_mat,
                   hb_labels,
                   y_zero_one,
                   p_id,
                   vec_n_neighbors=200,
                   vec_c =0.2,
                   sig_n_neighbors=4,
                   sig_c=0.03,
                   n_process = -1):
    
    sig_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=sig_n_neighbors,
                             contamination=sig_c,
                             novelty=False,
                             n_jobs=n_process)
    sig_y_pred = sig_lof.fit_predict(X=sig_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    sig_local_out_factors = -sig_lof.negative_outlier_factor_
    sig_lof_threshold = -sig_lof.offset_
    sig_normalized = (sig_local_out_factors-min(sig_local_out_factors))/(max(sig_local_out_factors)- \
                                                             min(sig_local_out_factors))
    # Precision-recall score
    sig_LOF_average_precision = average_precision_score(hb_labels, sig_normalized)
    # calculate model precision-recall curve
    sig_LOF_precision, sig_LOF_recall, _ = precision_recall_curve(y_zero_one, sig_normalized)


    vec_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=vec_n_neighbors,
                             contamination=vec_c,
                             novelty=False,
                             n_jobs=n_process)
    vec_y_pred = vec_lof.fit_predict(X=vec_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    vec_local_out_factors = -vec_lof.negative_outlier_factor_
    vec_lof_threshold = -vec_lof.offset_
    vec_normalized = (vec_local_out_factors-min(vec_local_out_factors))/(max(vec_local_out_factors)- \
                                                             min(vec_local_out_factors))
    # Precision-recall score
    vec_LOF_average_precision = average_precision_score(hb_labels, vec_normalized)
    # calculate model precision-recall curve
    vec_LOF_precision, vec_LOF_recall, _ = precision_recall_curve(y_zero_one, vec_normalized)

    # # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_zero_one[y_zero_one==1]) / len(y_zero_one)
    
    fig = go.Figure()
    

    # plot the LOOP model precision-recall curve
    fig.add_trace(go.Scatter(x=sig_LOF_recall, y=sig_LOF_precision,
                        mode='lines+markers',
                        name='Graph-Based Avg Precision {0:0.4f}'.format(sig_LOF_average_precision)))
    
    # plot the LOF model precision-recall curve
    fig.add_trace(go.Scatter(x=vec_LOF_recall, y=vec_LOF_precision,
                        mode='lines+markers',
                        name='Event-Based Avg Precision {0:0.4f}'.format(vec_LOF_average_precision)))
    # plot the no skill precision-recall curve
    fig.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill],
                        name='No Skill', line = dict(color='green', width=4, dash='dash')))
    
    fig.update_layout(
        title="Graph-Based vs Event-Based Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_title="Implementations",
    )
    
    #fig.show()
    if not os.path.exists("PR_images"):
        os.mkdir("PR_images")
    fig.to_image(format="png", width=300, height=300, scale=2)
    fig.write_image("PR_images/PR_curve_"+str(p_id)+".png")
    
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
    
def combine_roc_curve(vec_dis_mat,
                      ged_dis_mat,
                      hb_labels,
                      y_zero_one,
                      vec_n_neighbors=200,
                      vec_c =0.2,
                      ged_n_neighbors=4,
                      ged_c=0.03,
                      n_process = -1):
    #compute combine
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
    
    combine_score = vec_y_scores+ged_y_scores
    combine_threshold = vec_threshold + ged_threshold
    combine_y_pred = [-1 if(score<combine_threshold) else 1 for score in combine_score]
    
    com_local_out_factors= -combine_score
    com_lof_threshold = -combine_threshold
    com_normalized = (com_local_out_factors-min(com_local_out_factors))/(max(com_local_out_factors)- \
                                                             min(com_local_out_factors))
    
    
    # calculate roc auc
    com_LOF_roc_auc = roc_auc_score(y_zero_one, com_normalized)
    # calculate roc curve for model
    com_LOF_fpr, com_LOF_tpr, _ = roc_curve(y_zero_one, com_normalized)
    
    vec_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=vec_n_neighbors,
                             contamination=vec_c,
                             novelty=False,
                             n_jobs=n_process)
    vec_y_pred = vec_lof.fit_predict(X=vec_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    vec_local_out_factors = -vec_lof.negative_outlier_factor_
    vec_lof_threshold = -vec_lof.offset_
    vec_normalized = (vec_local_out_factors-min(vec_local_out_factors))/(max(vec_local_out_factors)- \
                                                             min(vec_local_out_factors))

    # calculate roc auc
    vec_LOF_roc_auc = roc_auc_score(y_zero_one, vec_normalized)
    # calculate roc curve for model
    vec_LOF_fpr, vec_LOF_tpr, _ = roc_curve(y_zero_one, vec_normalized)

    
    fig = go.Figure()
    

    # plot roc curve for LoOP model 
    fig.add_trace(go.Scatter(x=com_LOF_fpr, y=com_LOF_tpr,
                        mode='lines+markers',
                        name='Combine Avg ROC-AUC score {0:0.3f}'.format(com_LOF_roc_auc)))
    
    # plot roc curve for LOF model 
    fig.add_trace(go.Scatter(x=vec_LOF_fpr, y=vec_LOF_tpr,
                        mode='lines+markers',
                        name='LOF Avg ROC-AUC score {0:0.3f}'.format(vec_LOF_roc_auc)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        name='No Skill', line = dict(color='green', width=4, dash='dash')))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Implementations",
    )

    fig.show()
    
def sig_roc_curve(vec_dis_mat,
                  sig_dis_mat,
                  hb_labels,
                  y_zero_one,
                  p_id,
                  vec_n_neighbors=200,
                  vec_c =0.2,
                  sig_n_neighbors=4,
                  sig_c=0.03,
                  n_process = -1):
    sig_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=sig_n_neighbors,
                             contamination=sig_c,
                             novelty=False,
                             n_jobs=n_process)
    sig_y_pred = sig_lof.fit_predict(X=sig_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    sig_local_out_factors = -sig_lof.negative_outlier_factor_
    sig_lof_threshold = -sig_lof.offset_
    sig_normalized = (sig_local_out_factors-min(sig_local_out_factors))/(max(sig_local_out_factors)- \
                                                             min(sig_local_out_factors))

   
    try:
        # calculate roc auc
        sig_LOF_roc_auc = roc_auc_score(y_zero_one, sig_normalized)
    except ValueError:
        sig_LOF_roc_auc = 0.001
    
    # calculate roc curve for model
    sig_LOF_fpr, sig_LOF_tpr, _ = roc_curve(y_zero_one, sig_normalized)
    
    
    vec_lof = LocalOutlierFactor(algorithm="brute",
                             metric="precomputed",
                             n_neighbors=vec_n_neighbors,
                             contamination=vec_c,
                             novelty=False,
                             n_jobs=n_process)
    vec_y_pred = vec_lof.fit_predict(X=vec_dis_mat)
    # lof.score_samples(X=ged_dis_mat)
    vec_local_out_factors = -vec_lof.negative_outlier_factor_
    vec_lof_threshold = -vec_lof.offset_
    vec_normalized = (vec_local_out_factors-min(vec_local_out_factors))/(max(vec_local_out_factors)- \
                                                             min(vec_local_out_factors))
    
    try:
        # calculate roc auc
        vec_LOF_roc_auc = roc_auc_score(y_zero_one, vec_normalized)
    except ValueError:
        vec_LOF_roc_auc = 0.001
    
    # calculate roc curve for model
    vec_LOF_fpr, vec_LOF_tpr, _ = roc_curve(y_zero_one, vec_normalized)

    
    fig = go.Figure()
    

    # plot roc curve for LoOP model 
    fig.add_trace(go.Scatter(x=sig_LOF_fpr, y=sig_LOF_tpr,
                        mode='lines+markers',
                        name='Graph-based ROC-AUC score {0:0.3f}'.format(sig_LOF_roc_auc)))
    
    # plot roc curve for LOF model 
    fig.add_trace(go.Scatter(x=vec_LOF_fpr, y=vec_LOF_tpr,
                        mode='lines+markers',
                        name='Event-based ROC-AUC score {0:0.3f}'.format(vec_LOF_roc_auc)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        name='No Skill', line = dict(color='green', width=4, dash='dash')))
    
    fig.update_layout(
        title="ROC-Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Implementations",
    )

    # fig.show()
    if not os.path.exists("ROC_images"):
        os.mkdir("ROC_images")
    fig.to_image(format="png", width=300, height=300, scale=2)
    fig.write_image("ROC_images/ROC_curve_"+str(p_id)+".png")
