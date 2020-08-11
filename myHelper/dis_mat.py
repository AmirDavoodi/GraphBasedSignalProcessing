from sklearn.metrics.pairwise import euclidean_distances
from myHelper import timing
# Gmatch4py use networkx graph 
import networkx as nx 
# import the GED using the munkres algorithm
import gmatch4py as gm

@timing.time_it
def cal_vec_dist_mat(heart_beats, metric='euclidean'):
    '''
    Compute the distance matrix between vector of heartbeats
    ------------------------------------------------------------------
    :param heart_beats: list of heartbeat dataframe. Notice each dataframe containing
                        two columns one for sample and other for the value of the 
                        signal.
    :param metric: the euclidean metric is the only measurement criteria for now.
    
    :return vec_dis_mat: a metrix with the shape of (N*N) which N = Heartbeat numbers
    '''
    
    hb_vecs = [hb[hb.columns[1]] for hb in heart_beats]
    
    # euclidean distance between vector of heart_beats.
    vec_dis_mat = euclidean_distances(hb_vecs, hb_vecs)
    
    return vec_dis_mat

@timing.time_it
def cal_ged_dist_mat(graph_stream):
    # convert from dictionary of graphs to list of graphs
    graph_stream_list = [v for k, v in graph_stream.items()] 
    ged=gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
    non_symetric_GED_matrix=ged.compare(graph_stream_list,None)
    symetric_ged_matrix = (non_symetric_GED_matrix.transpose()+non_symetric_GED_matrix)/2
    return symetric_ged_matrix
