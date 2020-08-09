import glob
import pandas as pd
from myHelper import GraphSignalHelper
import plotly.graph_objects as go

def get_file_name(path, path_directory):
    '''
    A simple function to get the name of the file by passing the directory name
    --------------------------------------------------------------------------
    :param text: the full path string
    :param prefix: the prefix string directory path string to be removed
    
    :return name: the name of the file without prefix and appendix
    '''
    name = path[path.startswith(path_directory) and len(path_directory):-4]
    return name

def get_patients_id(DB_dir):
    '''
    get the patients number from the Database directory
    --------------------------------------------------------------------------
    :param DB_dir: path to the database directory
    
    :return patients_num: a list containing the patients numbers
    '''
    path = DB_dir+'*'
    
    patients_num = []
    for file in glob.glob(path):
        file_name = get_file_name(file, DB_dir)
        if(len(file_name)==3):
            patients_num.append(file_name)
    patients_num.sort()
    return patients_num

def read_heartbeat_database(DB_dir, p_id, nRowsRead=None):
    '''
    function to read the csv file by passing its name from the database
    --------------------------------------------------------------------------
    :param name: name of the csv file
    :param nRowsRead: spcify the number of rows to read from the csv file
                      nRowsRead = None # if want to read whole file
    :return dataframe
    '''
    data_df = pd.read_csv(DB_dir+p_id+'.csv', delimiter=',', nrows = nRowsRead)
    data_df.dataframeName = p_id +'.csv'
    return data_df

def read_annotation_database(DB_dir, p_id):
    '''
    function to read the annotation from the txt file in the database
    --------------------------------------------------------------------------
    :param name: name of the txt file containing the annotations for
                 the csv file

    :return dataframe
    '''
    annotations = pd.read_csv(DB_dir+p_id+'annotations.txt', quotechar=None, quoting=3,  delim_whitespace=True )
    annotations.dataframeName = p_id +'annotations.txt'
    annotations.drop(['Aux'], axis=1, inplace=True)
    annotations.columns = ["Time", "Sample", "Type", "Sub", "Chan", "Num", "Aux"]
    return annotations

def db_observer(DB_dir):
    original_db_df = manual_db_df()
    
    patients_id = get_patients_id(DB_dir)
    patients_id.sort()
    R_detectors = ['my', 'tompkins', 'wavelet', 'two_aver']
    for r_detect in R_detectors:
        # a list for storing the number of anomalous and nomal heartbeat for our data
        DB_anomaly_observation = []

        for p_id in patients_id:
            data_df = read_heartbeat_database(DB_dir, p_id, nRowsRead=None)
            annotations_df = read_annotation_database(DB_dir, p_id)

            # downsampling the signal
            downstep = 8
            (downsp_data_df, downsp_annotations_df) = GraphSignalHelper.downsample(data_df,
                                                                           annotations_df,
                                                                           downstep)
            # split the the signal into heartbeat using the R-peak detection
            heart_beats, hbs_annotations = GraphSignalHelper.heartbeat_spliting(downsp_data_df, 
                                                                downsp_annotations_df,
                                                                R_detector=r_detect,
                                                                fs = (360 / downstep))
            # labeling heartbeats by looking at their annotations into 
            # (-1) or (1) for Anomalous or Normal
            hb_labels = GraphSignalHelper.anomaly_labels(hbs_annotations)

            num_anomalous_hbs = sum(hb_labels == -1)
            num_normal_hbs = sum(hb_labels == 1)
            total_hb = len(heart_beats)
            total_labels = num_anomalous_hbs+num_normal_hbs

            DB_anomaly_observation.append([num_normal_hbs, 
                                           num_anomalous_hbs,
                                           total_hb, 
                                           (total_labels==total_hb)])

        DB_anomaly_observation_df = pd.DataFrame(DB_anomaly_observation, 
                                                 columns = ['Normal ('+r_detect+')',
                                                            'Anomalous ('+r_detect+')',
                                                            'Total ('+r_detect+')',
                                                            'All HB labeled  ('+r_detect+')'])
        original_db_df = pd.concat([original_db_df, 
                                  DB_anomaly_observation_df], 
                                  axis=1, join='inner', sort=False)
        print(r_detect, "r_peak detection done.")
    original_db_df = original_db_df.reindex(sorted(original_db_df.columns), axis=1)
    return original_db_df

def db_observer_plotter(original_db_df):
    # computing the sum over relevant columns of the dataframe
    original_db_df.loc['Total',:]= original_db_df.drop([original_db_df.columns[0],
                                          original_db_df.columns[1],
                                          original_db_df.columns[2],
                                          original_db_df.columns[3],
                                          original_db_df.columns[4],
                                          original_db_df.columns[-1]],
                                         axis=1).sum()
    # get the row of the dataframe which containing the aggregated numbers
    total_row = original_db_df.iloc[-1]

    # calculating the difference of each method
    diff = []
    temporary = 9
    for j in [temporary,temporary+5,temporary+5+5]:    
        for i in range(4,0,-1):
            diff.append(abs(total_row[j]-total_row[j-i]))
        diff.append(total_row[j])
        
    values = [['Normal Heartbeats', 'Anomalous Heartbeats', '<b>Total Heartbeats</b>'], #1st col
              [str(int(total_row[10]))+' ('+str(int(diff[5]))+')',
               str(int(total_row[5]))+' ('+str(int(diff[0]))+')', 
               str(int(total_row[15]))+' ('+str(int(diff[10]))+')'], #2nd col
              [str(int(total_row[10+1]))+' ('+str(int(diff[5+1]))+')',
               str(int(total_row[5+1]))+' ('+str(int(diff[0+1]))+')', 
               str(int(total_row[15+1]))+' ('+str(int(diff[10+1]))+')'],#3rd col
              [str(int(total_row[10+2]))+' ('+str(int(diff[5+2]))+')', 
               str(int(total_row[5+2]))+' ('+str(int(diff[0+2]))+')', 
               str(int(total_row[15+2]))+' ('+str(int(diff[10+2]))+')'],#4th col
              [str(int(total_row[10+3]))+' ('+str(int(diff[5+3]))+')', 
               str(int(total_row[5+3]))+' ('+str(int(diff[0+3]))+')',  
               str(int(total_row[15+3]))+' ('+str(int(diff[10+3]))+')'],#5th col
              [str(int(total_row[10+4])), 
               str(int(total_row[5+4])), 
               str(int(total_row[15+4]))]]#6th col


    fig = go.Figure(data=[go.Table(
      columnorder = [1,2,3,4,5,6],
      columnwidth = [120,115,100,100,100,80],
      header = dict(
        values = ['<b>R-Peak Method</b><br>Anomalies',
                   '<b>Simple Moving Average</b>',
                   '<b>Pan and Tompkins</b>',
                   '<b>Two Moving Average</b>',
                   '<b>Stationary Wavelet Transform</b>',
                   '<b>Ground truth (Database)</b>'],
        line_color='darkslategray',
        fill_color='royalblue',
        align=['left','center','center','center','center','center'],
        font=dict(color='white', size=12),
        height=40
      ),
      cells=dict(
        values=values,
        line_color='darkslategray',
        fill=dict(color=['paleturquoise', 'white']),
        align=['left', 'center','center','center','center','center'],
        font_size=12,
        height=25)
        )
    ])
    fig.show()

def db_patient_plotter(original_db_df, p_id):
    db_det = [original_db_df.loc[original_db_df['patient #'] == p_id]['Normal HB'],
              original_db_df.loc[original_db_df['patient #'] == p_id]['Anomalous HB'],
              original_db_df.loc[original_db_df['patient #'] == p_id]['Total HB']]
    my_det = [original_db_df.loc[original_db_df['patient #'] == p_id]['Normal (my)'],
              original_db_df.loc[original_db_df['patient #'] == p_id]['Anomalous (my)'],
              original_db_df.loc[original_db_df['patient #'] == p_id]['Total (my)']]
    tomp_det = [original_db_df.loc[original_db_df['patient #'] == p_id]['Normal (tompkins)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Anomalous (tompkins)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Total (tompkins)']]
    two_m_det = [original_db_df.loc[original_db_df['patient #'] == p_id]['Normal (two_aver)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Anomalous (two_aver)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Total (two_aver)']]
    wavelet_det = [original_db_df.loc[original_db_df['patient #'] == p_id]['Normal (wavelet)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Anomalous (wavelet)'],
                original_db_df.loc[original_db_df['patient #'] == p_id]['Total (wavelet)']]
    # convert elements to string
    db_det = [str(int(i)) for i in db_det]
    my_det = [str(int(i)) for i in my_det]
    tomp_det = [str(int(i)) for i in tomp_det]
    two_m_det = [str(int(i)) for i in two_m_det]
    wavelet_det = [str(int(i)) for i in wavelet_det]

    values = [['Normal Heartbeats', 'Anomalous Heartbeats', '<b>Total Heartbeats</b>'], #1st col
              my_det, #2nd col
              tomp_det,#3rd col
              two_m_det,#4th col
              wavelet_det,#5th col
              db_det]#6th col


    fig = go.Figure(data=[go.Table(
      columnorder = [1,2,3,4,5,6],
      columnwidth = [120,115,100,100,100,80],
      header = dict(
        values = ['<b>R-Peak Method</b><br>Anomalies',
                   '<b>Simple Moving Average</b>',
                   '<b>Pan and Tompkins</b>',
                   '<b>Two Moving Average</b>',
                   '<b>Stationary Wavelet Transform</b>',
                   '<b>Ground truth (Database)</b>'],
        line_color='darkslategray',
        fill_color='royalblue',
        align=['left','center','center','center','center','center'],
        font=dict(color='white', size=12),
        height=40
      ),
      cells=dict(
        values=values,
        line_color='darkslategray',
        fill=dict(color=['paleturquoise', 'white']),
        align=['left', 'center','center','center','center','center'],
        font_size=12,
        height=25)
        )
    ])
    fig.show()
    
def manual_db_df():
    '''
    Manually adding number of normal and anomalous and total heartbeat form
    
    the following documentation page of MIT-BIH Database
    https://www.physionet.org/physiobank/database/html/mitdbdir/records.htm
    -----------------------------------------------------------------------
    :return DB_documentation: a dataframe containing the information for each patient
    '''
    DB_documentation = []

    p_id=100
    num_normal_hbs = 2239
    num_anomalous_hbs = 33+1
    total_hb = 2273
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=101
    num_normal_hbs = 1860
    num_anomalous_hbs = 3+2
    total_hb = 1865
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=102
    num_normal_hbs = 99
    num_anomalous_hbs = 4+2028+56
    total_hb = 2187
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=103
    num_normal_hbs = 2082
    num_anomalous_hbs = 2
    total_hb = 2084
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=104
    num_normal_hbs = 163
    num_anomalous_hbs = 2+1380+666+18
    total_hb = 2229
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=105
    num_normal_hbs = 2526
    num_anomalous_hbs = 41+5
    total_hb = 2572
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=106
    num_normal_hbs = 1507
    num_anomalous_hbs = 520
    total_hb = 2027
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=107
    num_normal_hbs = 0
    num_anomalous_hbs = 2078+59
    total_hb = 2137
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=108
    num_normal_hbs = 1740
    num_anomalous_hbs = 4+17+2+11
    total_hb = 1774
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=109
    num_normal_hbs = 0
    num_anomalous_hbs = 2492+38+2
    total_hb = 2532
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=111
    num_normal_hbs = 0
    num_anomalous_hbs = 2123+1
    total_hb = 2124
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=112
    num_normal_hbs = 2537
    num_anomalous_hbs = 2
    total_hb = 2539
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=113
    num_normal_hbs = 1789
    num_anomalous_hbs = 6
    total_hb = 1795
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=114
    num_normal_hbs = 1820
    num_anomalous_hbs = 10+2+43+4
    total_hb = 1879
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=115
    num_normal_hbs = 1953
    num_anomalous_hbs = 0
    total_hb = 1953
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=116
    num_normal_hbs = 2302
    num_anomalous_hbs = 109+1
    total_hb = 2412
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=117
    num_normal_hbs = 1534
    num_anomalous_hbs = 1
    total_hb = 1535
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=118
    num_normal_hbs = 0
    num_anomalous_hbs = 2166+96+16+10
    total_hb = 2288
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=119
    num_normal_hbs = 1543
    num_anomalous_hbs = 444
    total_hb = 1987
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=121
    num_normal_hbs = 1861
    num_anomalous_hbs = 1+1
    total_hb = 1863
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=122
    num_normal_hbs = 2476
    num_anomalous_hbs = 0
    total_hb = 2476
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=123
    num_normal_hbs = 1515
    num_anomalous_hbs = 3
    total_hb = 1518
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=124
    num_normal_hbs = 0
    num_anomalous_hbs = 1531+2+29+47+5+5
    total_hb = 1619
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=200
    num_normal_hbs = 1743
    num_anomalous_hbs = 30+826+2
    total_hb = 2601
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=201
    num_normal_hbs = 1625
    num_anomalous_hbs = 30+97+1+198+2+10+37
    total_hb = 2000
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=202
    num_normal_hbs = 2061
    num_anomalous_hbs = 36+19+19+1
    total_hb = 2136
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=203
    num_normal_hbs = 2529
    num_anomalous_hbs = 2+444+1+4
    total_hb = 2980
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=205
    num_normal_hbs = 2571
    num_anomalous_hbs = 3+71+11
    total_hb = 2656
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=207
    num_normal_hbs = 0
    num_anomalous_hbs = 1457+86+107+105+472+105
    total_hb = 2332
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=208
    num_normal_hbs = 1586
    num_anomalous_hbs = 2+992+373+2
    total_hb = 2955
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=209
    num_normal_hbs = 2621
    num_anomalous_hbs = 383+1
    total_hb = 3005
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=210
    num_normal_hbs = 2423
    num_anomalous_hbs = 22+194+10+1
    total_hb = 2650
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=212
    num_normal_hbs = 923
    num_anomalous_hbs = 1825
    total_hb = 2748
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=213
    num_normal_hbs = 2641
    num_anomalous_hbs = 25+3+220+362
    total_hb = 3251
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=214
    num_normal_hbs = 0
    num_anomalous_hbs = 2003+256+1+2
    total_hb = 2262
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=215
    num_normal_hbs = 3195
    num_anomalous_hbs = 3+164+1
    total_hb = 3363
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=217
    num_normal_hbs = 244
    num_anomalous_hbs = 162+1542+260
    total_hb = 2208
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=219
    num_normal_hbs = 2082
    num_anomalous_hbs = 7+64+1+133
    total_hb = 2287
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=220
    num_normal_hbs = 1954
    num_anomalous_hbs = 94
    total_hb = 2048
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=221
    num_normal_hbs = 2031
    num_anomalous_hbs = 396
    total_hb = 2427
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=222
    num_normal_hbs = 2062
    num_anomalous_hbs = 208+1+212
    total_hb = 2483
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=223
    num_normal_hbs = 2029
    num_anomalous_hbs = 72+1+473+14+16
    total_hb = 2605
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=228
    num_normal_hbs = 1688
    num_anomalous_hbs = 3+362
    total_hb = 2053
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=230
    num_normal_hbs = 2255
    num_anomalous_hbs = 1
    total_hb = 2256
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=231
    num_normal_hbs = 314
    num_anomalous_hbs = 1254+1+2+2
    total_hb = 1573
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=232
    num_normal_hbs = 0
    num_anomalous_hbs = 397+1382+1
    total_hb = 1780
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=233
    num_normal_hbs = 2230
    num_anomalous_hbs = 7+831+11
    total_hb = 3079
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    p_id=234
    num_normal_hbs = 2700
    num_anomalous_hbs = 50+3
    total_hb = 2753
    total_labels = num_anomalous_hbs+num_normal_hbs
    DB_documentation.append([p_id, 
                             num_normal_hbs, 
                             num_anomalous_hbs,
                             total_hb, 
                             (total_labels==total_hb)])
    DB_documentation = pd.DataFrame(DB_documentation, columns = ['patient #', 
                                                                 'Normal HB',
                                                                 'Anomalous HB',
                                                                 'Total HB',
                                                                 'All HB labeled'])
    return DB_documentation