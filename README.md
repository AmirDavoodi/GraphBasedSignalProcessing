# Graph-Based Signal Processing on ECG(EKG) signal

This reportory is the implementation for my master thesis "Goal-directed graph generation for anomaly and change detection" to specifically test my model (Pipeline) on ECG signal to validate the performance of our model on reallife signal.

In the model, we do the following steps from a pure signal to a labeled signal which indicates anomalies and changes in the signal.

1. Apply some preprocessing actions on the signal.
    - Apply a filter to clean the signal from any possible noise.
    - Apply a series of processes to split the signal into multiple sections (Here in ECG signal, we splited out signal into different heartbeats.)
        * We first extract some statistical continuous value from the signal. (such as moving average)
        * Using the computed statistical values, we find the R-peaks in the ECG signal (R-peaks are the maximum points)
        * Compute some other measurements like Beat Per Minutes (BPM) for reliability check of the found R-peaks.
        * Shitf the moving average line for a better R-peak detection.
2. Generate the appropriate graph for each splited part of the signal (signal for each heartbeat) using the proper algorithm.
    - We are generating this graph with a specific goal of anomaly detection, as the name of my thesis states "Goal-directed graph generation". At this repository, we are testing the effect of different algorithms on the generated graph and in the end the detection of the anomalies in the signal. These algorithms are the followings:
        * Visibility algorithm
        * Horizontal visibility algorithm
3. Anomaly detection on graph representation of the signal: using the generated graph for each heartbeat, we want to find the heartbeat which is showing anomalies compare to others. (We are using a anotated database, so we first train based on the labels corespondance to each heartbeat and then we predict the anomalies on the future data based on the trained model.). For this aim, I am experimenting and implementing the results of the following machine learning algorithms.
    - KNN : k nearest neighbor
    - LSTM: Long short-term memory
4. Benchmarking: comparing the result of the model to other non-graph representation approaches and conclude the effect of our goal-directed graph generation approach on the performance of the anomaly detection model.


The database that I used is downloaded from the following kaggle link which is just the csv version of the original physionet dataset.
my downloaded dataset: https://www.kaggle.com/mondejar/mitbih-database
Original physionet dataset: https://www.physionet.org/content/mitdb/1.0.0/
