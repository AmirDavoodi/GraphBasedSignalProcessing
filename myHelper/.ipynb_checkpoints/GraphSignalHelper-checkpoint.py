import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import seaborn as sns
import math
import networkx as nx
from ecgdetectors import Detectors

# old_settings = np.seterr('raise')
# import warnings
# warnings.filterwarnings('ignore')

def movingaverage(original_ecg, hrw=0.75, fs = 360):
    '''
    Compute simple moving average of the signal using the window size of
    0.75s in both directions
    --------------------------------------------------------------------------

    :param original_ecg: recorded ECG signal
    :param hrw: One-sided window size, as proportion of the sampling frequency
                hrw=0.75 means window of 0.75 second in both directions
    :param fs: frequency in which the data is recorded
               for our MIT-BIH the frequency is 360 Hz

    :return mov_avg: calculated moving average for the signal
    '''

    #Calculate moving average using the formula
    mov_avg = original_ecg.rolling(int(hrw*fs)).mean()

    #Impute where moving average function returns NaN, which
    # is the beginning of the signal where x hrw
    avg_hr = (np.mean(original_ecg))
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    return mov_avg

def detect_peaks(original_ecg, mov_avg, ma_perc):
    '''
    Using the computed moving average and detect R-peaks
    --------------------------------------------------------------------------

    :param original_ecg: recorded ECG signal
    :param mov_avg: moving average of the signal
    :param ma_perc: moving average percentage to shift for better detection

    :return peaklist: Notate positions of the point where R-peak detected on
                      the X-axis
    :return ybeat: y-value of all peaks for plotting purposes
    '''
    #Raise the average by ma_perc% to prevent the secondary
    # heart contraction from interfering
    mov_avg = [(x+((x/100)*ma_perc)) for x in mov_avg]

    #Mark regions of interest
    window = []
    peaklist = []

    #We use a counter to move over the different data columns
    listpos = 0
    for datapoint in original_ecg:
        rollingmean = mov_avg[listpos] #Get local mean

        #If no detectable R-complex activity -> do nothing
        if (datapoint <= rollingmean) and (len(window) < 1):
            listpos += 1
        #If signal comes above local mean, mark ROI
        elif (datapoint > rollingmean):
            window.append(datapoint)
            listpos += 1
        #If signal drops below local mean -> determine highest point
        else:
            maximum = max(window)
            #Notate the position of the point on the X-axis
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    #Get the y-value of all peaks for plotting purposes
    ybeat = [original_ecg[x] for x in peaklist]

    return peaklist, ybeat

def measure_rr(peaklist, fs = 360):
    '''
    Measure the R-R distance which is the distance between each two R-Peak
    for each two adjacent heartbeat.
    --------------------------------------------------------------------------

    :param peaklist: detected position of R-peaks in the signal.
    :param fs: frequency in which the data is recorded

    :return RR_list: list containing the of R-R distance for all heartbeats
    '''
    RR_list = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        #Calculate distance between beats in # of samples
        RR_interval = (peaklist[cnt+1] - peaklist[cnt])
        #Convert sample distances to ms distances
        ms_dist = ((RR_interval / fs) * 1000.0)
        RR_list.append(ms_dist) #Append to list
        cnt += 1
    return RR_list

def measure_rrsd(RR_list):
    '''
    Compute the standard deviation of the R-R distances (RRSD)
    --------------------------------------------------------------------------

    :param RR_list: list containing the of R-R distance for all heartbeats

    :return std: standard deviation
    '''
    # list is empty for warning
    if not RR_list:
        return 0
    else:
        return np.std(RR_list)

def measure_bpm(RR_list):
    '''
    Compute the Beat Per Minute (BPM) values from the list of (RR_list).
    --------------------------------------------------------------------------

    :param RR_list: list containing the of R-R distance for all heartbeats

    :return bpm_list: Beat Per Minute (BPM) for each heartbeat as list for the
                      whole signal
    '''

    # bpm for each beat.
    bpm_list = [60000/RR_ms_dist for RR_ms_dist in RR_list]
    return bpm_list

def measure_bpm_signal(mov_avg, bpm, peaklist):
    '''
    For plotting purpose, I create a bpm_signal list which is as same size as
    the original signal.
    --------------------------------------------------------------------------

    :param mov_avg: moving average for the signal
    :param bpm: Beat Per Minute (BPM) for each heartbeat as list for the
                whole signal
    :param peaklist: positions of the point where R-peak detected on
                     the X-axis of the signal.

    :return bpm_signal: bpm_signal which is a list as same size as the
                        original signal.
    :return avg_bpm: the average BPM for the whole signal
    '''

    avg_bpm = np.mean(bpm)

    bpm_signal = np.repeat(avg_bpm, len(mov_avg))

    i = 0
    last_set_bpm = avg_bpm
    for index, datapoint in enumerate(bpm_signal):
        if (index == peaklist[i]):
            last_set_bpm = bpm[i]
            bpm_signal[index] = last_set_bpm
            if(i < len(bpm)-1):
                i+= 1
        else:
            bpm_signal[index] = last_set_bpm
    return bpm_signal, avg_bpm

def find_best_ma_perc_shift(original_ecg, mov_avg, fs):
    '''
    Test some possible moving average percentage shift (ma_perc_shift)
    for selecting the best from by looking at RRSD and BPM
    --------------------------------------------------------------------------

    :param original_ecg: recorded ECG signal
    :param mov_avg: moving average of the signal
    :param fs: frequency in which the data is recorded

    :return ma_perc_best: best percentatge shift for the moving average
    '''
    #List with moving average raise percentages, make
    # as detailed as you like but keep an eye on speed
    ma_perc_list = [5, 10, 15, 20, 25, 30]
#     ma_perc_list = [*range(1, 30, 1)] 
    rrsd_list = []
    valid_ma = []

    #Detect peaks with all percentages, append results to list 'rrsd'
    for ma_perc in ma_perc_list:
        (peaklist, ybeat) = detect_peaks(original_ecg, mov_avg, ma_perc)
        peak_bpm = ((len(peaklist)/(len(original_ecg)/fs))*60)
        RR_list = measure_rr(peaklist, fs)
        rrsd = measure_rrsd(RR_list)
        rrsd_list.append([rrsd, peak_bpm, ma_perc])

    #Test rrsd_list entries and select valid measures
    # valid moving average percentage shift is the one which give us lowest non-zero RRSD and
    # belivable BPM here I took the belivable BPM (30 < bpm < 130)
    for rrsd, peak_bpm, ma_perc in rrsd_list:
        if ((rrsd > 1) and ((peak_bpm > 30) and (peak_bpm < 130))):
            valid_ma.append([rrsd, ma_perc])

    #Save the ma_perc for plotting purposes later on (not needed)
    ma_perc_best = min(valid_ma, key = lambda t: t[0])[1]
    rrsd_best = min(valid_ma, key = lambda t: t[0])[0]
    return ma_perc_best

def fixed_hb_split(original_ecg_sample_n, peaklist, annotations_df):
    '''
    Split the ECG signal [original_ecg] into its fixed heart beats using
    the index of the R-peaks [peaklist]
    --------------------------------------------------------------------------

    :param original_ecg: recorded ECG signal
    :param peaklist: positions of the point where R-peak detected on
                     the X-axis of the signal
    :param annotations_df: annotations for the ECG data

    :return heart_beats: the origianl signal for each heartbeat
    :ruturn hbs_annotations: annotations for each heartbeat
    '''
    fix = np.mean([(j-i) for i, j in zip(peaklist[:-1], peaklist[1:])]).astype(int)
    fix_half = int(fix/2)
    heart_beats = [original_ecg_sample_n[i-fix_half:i+fix_half] for i in peaklist]
    heart_beats = heart_beats[1:-1]
    
    hbs_annotations = []
    for hb in heart_beats:
        # get the maximum and minimum of the sample numbers of each heart beat
        max_sample_number = hb[hb.columns[0]].max()
        min_sample_number = hb[hb.columns[0]].min()

        # sort to get only annotations for the sample numbers in range
        sorted_annotations_df = annotations_df.loc[(annotations_df['Sample'] <= max_sample_number)
                                                   & (annotations_df['Sample'] >= min_sample_number)]
        hbs_annotations.append(sorted_annotations_df)
    return heart_beats, hbs_annotations

def hb_split(original_ecg_sample_n, peaklist, annotations_df):
    '''
    Split the ECG signal [original_ecg] into its heart beats using
    the index of the R-peaks [peaklist]
    --------------------------------------------------------------------------

    :param original_ecg: recorded ECG signal
    :param peaklist: positions of the point where R-peak detected on
                     the X-axis of the signal
    :param annotations_df: annotations for the ECG data

    :return heart_beats: the origianl signal for each heartbeat
    :ruturn hbs_annotations: annotations for each heartbeat
    '''
    btw_R_peaks = [int((i+j)/2) for i, j in zip(peaklist[:-1], peaklist[1:])]
    heart_beats = [original_ecg_sample_n[i : j] for i, j in zip([0] + btw_R_peaks, btw_R_peaks + [None])]
    heart_beats = heart_beats[1:-1]
    
    hbs_annotations = []
    for hb in heart_beats:
        # get the maximum and minimum of the sample numbers of each heart beat
        max_sample_number = hb[hb.columns[0]].max()
        min_sample_number = hb[hb.columns[0]].min()

        # sort to get only annotations for the sample numbers in range
        sorted_annotations_df = annotations_df.loc[(annotations_df['Sample'] <= max_sample_number)
                                                   & (annotations_df['Sample'] >= min_sample_number)]
        hbs_annotations.append(sorted_annotations_df)
    return heart_beats, hbs_annotations

def nvg(series, timeLine):
    '''
    produce the natural visibility graph given a time series
    --------------------------------------------------------------------------

    :param series: a time series
    :param timeLine: is the vector containing the time stamps

    :return all_visible: the output which is formated like the following:
                         [[[1, [2]],[2, [3, 4, 5]],...]
                         This means [node 1 have edge to node 2] and [node 2
                         have edge to nodes 3, 4 and 5] and ...
    '''
    L = len(series)
    # timeLine is the vector containing the time stamps
    #if timeLine == None: timeLine = range(L)

    # initialise output
    all_visible = []


    for i in range(L-1):
        node_visible = []
        ya = float(series[i])
        ta = timeLine[i]

        for j in range(i+1,L):
            yb = float(series[j])
            tb = timeLine[j]

            yc = series[i+1:j]
            tc = timeLine[i+1:j]

            if all( yc[k] < (ya + (yb - ya)*(tc[k] - ta)/(tb-ta)) for k in range(len(yc)) ):
                node_visible.append(tb)

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible

def hvg(series, timeLine):
    '''
    Produce Horizontal visibility graph given a time series
    --------------------------------------------------------------------------

    :param series: a time series
    :param timeLine: is the vector containing the time stamps

    :return all_visible: the output which is formated like the following:
                         [[[1, [2]],[2, [3, 4, 5]],...]
                         This means [node 1 have edge to node 2] and [node 2
                         have edge to nodes 3, 4 and 5] and ...
    '''
    # series is the data vector to be transformed
    #if timeLine == None: timeLine = range(len(series))
    # Get length of input series
    L = len(series)
    # initialise output
    all_visible = []

    for i in range(L-1):
        node_visible = []
        ya = series[i]
        ta = timeLine[i]
        for j in range(i+1,L):

            yb = series[j]
            tb = timeLine[j]

            yc = series[i+1:j]
            tc = timeLine[i+1:j]

            if all( yc[k] < min(ya,yb) for k in range(len(yc)) ):
                node_visible.append(tb)
            elif any( yc[k] >= max(ya,yb) for k in range(len(yc)) ):
                break

        if len(node_visible)>0 : all_visible.append([ta, node_visible])

    return all_visible

def downsample(data_df, annotations_df, downstep=2):
    '''
    Downsample ECG dataframe(data_df) and its annotations(annotations_df)
    --------------------------------------------------------------------------
    :param data_df: ECG(EKG) data from the dataset
    :param annotations_df: annotations for the ECG data
    
    :return downsp_data_df: downsampled dataframe for recorded ECG
    :return downsp_annotations_df: downsampled annotations for the ECG data
    '''
    # downsampling the ecg signal (drop one in a row)
    downsp_data_df = data_df.iloc[::downstep]
    downsp_annotations_df = annotations_df.copy()
    
    
    # find the annotations which their samples have been deleted in downsampling process
    deleted_samples = ~downsp_annotations_df['Sample'].isin(downsp_data_df.iloc[:,0])
    
    # replace the deleted sample values of annotation with the previous existing sample value
    downsp_annotations_df.loc[deleted_samples, 'Sample'] = downsp_annotations_df.loc[deleted_samples, 'Sample'] - \
                                      (downsp_annotations_df.loc[deleted_samples, 'Sample'] % downstep)
    
    #reset the indexing after droping some rows of the dataframe
    downsp_data_df.index = range(len(downsp_data_df.index))
    
    return downsp_data_df, downsp_annotations_df

def anomaly_labels(hbs_annotations):
    '''
    Assign the anomaly labels based on the annotations of heach HB in our database
    1 = Normal heartbeat ('Type' column is ['N'] or [two 'N'])
    -1 = Anomalous heartbeat ('Type' != 'N')
    --------------------------------------------------------------------------

    :param hbs_annotations: a list of dataframes which can contains annotations for Heartbeats.
    
    :return true_labels: list containing the true labels of (1) or (-1)
    '''
    true_labels = []
    for hb_index, hb_annotation in enumerate(hbs_annotations):
        if(len(hb_annotation) == 1):
            if(hb_annotation == "N"):
                true_labels.append(1)
            else:
                true_labels.append(-1)
        elif(len(hb_annotation) > 1):
            if('N' in hb_annotation):
                true_labels.append(1)
            else:
                true_labels.append(-1)
        elif (len(hb_annotation) == 0):
            true_labels.append(-1)
    return np.array(true_labels)

def hb_concated_anno(hbs_annotations):
    '''
    Contatinate all the annotations for each heartbeat
    --------------------------------------------------------------------------

    :param hbs_annotations: a list of dataframes which can contains annotations for Heartbeats.
    
    :return annotations: list containing the contatinated annotations for heartbeats
    '''
    annotations = []
    for graph_index, graph_annotation in enumerate(hbs_annotations):
        annotations.append(graph_annotation['Type'].str.cat())
    return np.array(annotations)

def generate_graph_stream(heart_beats, hb_labels):
    '''
    Function to generate the graph stream and label for each graph.
    --------------------------------------------------------------------------
    :param heart_beats: the origianl signal for each heartbeat
    :param hb_labels: annotations for each heartbeat

    :return graph_stream: array of graphs
    :return graph_stream_labels: array of labels coresponding to each
                                 graph in the stream.
    '''
    
    hbs_vg = {}
    # hbs_hvg = {}
    for hb_num, hb in enumerate(heart_beats):
        series = hb[hb.columns[1]].tolist()
        time_samples = range(len(series))
        # time_samples = hb.index.values.tolist()
        nvg_array = nvg(series, time_samples)
        # hvg_array = hvg(series, time_samples)

        hbs_vg[hb_num] = nx.Graph()
        for node_connections in nvg_array:
            edges = [(node_connections[0], item) for item in node_connections[1]]
            hbs_vg[hb_num].add_edges_from(edges)

        # hbs_hvg[hb_num] = nx.Graph()
        # for node_connections in hvg_array:
        #     edges = [(node_connections[0], item) for item in node_connections[1]]
        #     hbs_hvg[hb_num].add_edges_from(edges)

    graph_stream = hbs_vg
    graph_stream_labels = hb_labels
    return graph_stream, graph_stream_labels

def my_single_ma_R_detector(original_ecg, fs):
    '''
    Simple R peak detector by computing moving average
    --------------------------------------------------------------------------
    :param original_ecg: recorded ECG signal
    :param hrw: One-sided window size, as proportion of the sampling frequency
                hrw=0.75 means window of 0.75 second in both directions
    :param fs: frequency in which the data is recorded
               for our MIT-BIH the frequency is 360 Hz

    :return peaklist: Notate positions of the point where R-peak detected on
                      the X-axis
    :return ybeat: y-value of all peaks for plotting purposes if needed
    '''
    mov_avg = movingaverage(original_ecg, fs=fs)

    # test some possible ma_perc_shift for selecting the
    # best from by looking at RRSD and BPM
    ma_perc_best = find_best_ma_perc_shift(original_ecg, mov_avg, fs)

#     print("ma_perc_best = ", ma_perc_best)
#     ma_perc_best = 5

    #Detect peaks with 'ma_perc_best'
    (peaklist, ybeat) = detect_peaks(original_ecg,
                                     mov_avg,
                                     ma_perc = ma_perc_best)
    return peaklist, ybeat

def heartbeat_spliting(data_df, annotations_df, R_detector='my', fs = 360):
    '''
    Split the signal and its labels into heartbeats
    --------------------------------------------------------------------------
    :param data_df: ECG(EKG) data from the dataset
    :param annotations_df: annotations for the ECG data
    :param fs: frequency in which the data is recorded
               for our MIT-BIH the frequency is 360 Hz
    
    :return heart_beats: the origianl signal for each heartbeat
    :ruturn hb_labels: annotations for each heartbeat
    '''
    original_ecg = data_df[data_df.columns[1]]
    original_ecg_sample_n = data_df.iloc[:, 0:2]
    
    if(R_detector == 'my'):
        (peaklist, ybeat) = my_single_ma_R_detector(original_ecg, fs=fs)
    elif(R_detector == 'tompkins'):
        detectors = Detectors(fs)
        peaklist = detectors.swt_detector(original_ecg)
    elif(R_detector == 'wavelet'):
        detectors = Detectors(fs)
        peaklist = detectors.pan_tompkins_detector(original_ecg)
    elif(R_detector == 'two_aver'):
        detectors = Detectors(fs)
        peaklist = detectors.two_average_detector(original_ecg)
        

    (heart_beats, hbs_annotations) = hb_split(original_ecg_sample_n,
                                              peaklist,
                                              annotations_df)
    hb_con_annot = hb_concated_anno(hbs_annotations)
    return heart_beats, hb_con_annot

def fixed_heartbeat_spliting(data_df, annotations_df, R_detector='my', fs = 360):
    '''
    Split the signal and its labels into fixed size heartbeats
    --------------------------------------------------------------------------
    :param data_df: ECG(EKG) data from the dataset
    :param annotations_df: annotations for the ECG data
    :param fs: frequency in which the data is recorded
               for our MIT-BIH the frequency is 360 Hz
    
    :return heart_beats: the origianl signal for each heartbeat
    :ruturn hb_labels: annotations for each heartbeat
    '''
    original_ecg = data_df[data_df.columns[1]]
    original_ecg_sample_n = data_df.iloc[:, 0:2]
    
    if(R_detector == 'my'):
        (peaklist, ybeat) = my_single_ma_R_detector(original_ecg, fs=fs)
    elif(R_detector == 'tompkins'):
        detectors = Detectors(fs)
        peaklist = detectors.swt_detector(original_ecg)
    elif(R_detector == 'wavelet'):
        detectors = Detectors(fs)
        peaklist = detectors.pan_tompkins_detector(original_ecg)
    elif(R_detector == 'two_aver'):
        detectors = Detectors(fs)
        peaklist = detectors.two_average_detector(original_ecg)
        

#     (heart_beats, hbs_annotations) = hb_split(original_ecg_sample_n,
#                                               peaklist,
#                                               annotations_df)
    (heart_beats, hbs_annotations) = fixed_hb_split(original_ecg_sample_n,
                                                    peaklist,
                                                    annotations_df)

    hb_con_annot = hb_concated_anno(hbs_annotations)
    return heart_beats, hb_con_annot