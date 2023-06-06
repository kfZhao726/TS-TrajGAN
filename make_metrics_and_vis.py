## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.discriminative_metrics_no_minmaxscalar import discriminative_score_metrics2
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from data_loading import read_txt_traj_data_loading
import time
import pathlib
import json
import csv
np.set_printoptions(suppress=True)


def do_metrics_and_vis(ori_path, syn_path, max_seq_len, metric_iteration, out_data_path):

    def padding_end_point(dataset, max_traj_len):
        padding_d = list()
        for data in dataset:
            if len(data) <= max_traj_len:
                padding_element = list()
                # padding_element.append((data[-1][0], data[-1][1], 0.0))
                padding_element.append((0.0, data[-1][1], data[-1][-1]))
                for _ in range(len(data), max_traj_len):
                    data = np.insert(data, len(data), padding_element, axis=0)
            padding_d.append(data)
        return padding_d


    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    ori_data = np.array(read_txt_traj_data_loading(ori_path, max_seq_len=max_seq_len, for_test=True)[-1])
    syn_data = np.array(read_txt_traj_data_loading(syn_path, max_seq_len=max_seq_len, for_test=True)[-1])

    ori_data = np.array(padding_end_point(ori_data, max_seq_len))
    syn_data = np.array(padding_end_point(syn_data, max_seq_len))
    print("*" * 100)
    print(ori_data[0])
    print("*" * 100)
    print(syn_data[0])
    print("*"*100)
    print("ori_data's shape = " + str(ori_data.shape))
    print("syn_data's shape = " + str(syn_data.shape))

    if len(syn_data) < len(ori_data):
        metrics_data = ori_data[:len(syn_data)]
        syn_data_test = syn_data
    if len(syn_data) >= len(ori_data):
        syn_data_test = syn_data[:len(ori_data)]
        metrics_data = ori_data

    syn_data_test = np.array(syn_data_test)
    metrics_data = np.array(metrics_data)
    print('trajectory evaluation dataset is ready.')
    print('Begin to make metrics and visualize syn trajectories...')
    # '--metric_iteration'
    metric_results2 = dict()
    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics(metrics_data, syn_data_test)
        discriminative_score.append(temp_disc)

    metric_results2['discriminative'] = np.mean(discriminative_score)

    # 1. Discriminative Score no MinMaxScalar
    discriminative_score2 = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics2(metrics_data, syn_data_test)
        discriminative_score2.append(temp_disc)
    metric_results2['discriminative2'] = np.mean(discriminative_score2)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(metric_iteration):
        temp_pred = predictive_score_metrics(metrics_data, syn_data_test)
        predictive_score.append(temp_pred)

    metric_results2['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(metrics_data, syn_data_test, out_data_path, 'pca', pic_num=1)
    visualization(metrics_data, syn_data_test, out_data_path, 'tsne', pic_num=1)

    ## Print discriminative and predictive scores
    print(metric_results2)

    metric_results_path = out_data_path + "metrics_score.txt"
    with open(metric_results_path, 'w') as f:
        f.write("the metric_result of first model syn_data: /r")
        f.write("/r")
        f.write("the metric_result of second model pred_data: /r")
        f.write(json.dumps(metric_results2))

    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))


if __name__ == '__main__':

    # ori_traj_txt_path = 'porto10/0_porto_grid10_b5_2w_.txt'
    # syn_traj_txt_path = 'porto10/timegan.txt'
    #
    # ori_traj_txt_path = './data/porto20_b5/0_porto_grid20_b5_2w_.txt'
    ori_traj_txt_path = './data/porto_grid20_b4_2w.txt'
    syn_traj_txt_path = './data/porto20_b4/ttsgan/tts_b4_porto20_time_4.txt'
    # syn_traj_txt_path = './data/porto20_b5/0_porto_grid20_b5_2w_.txt'

    do_metrics_and_vis(ori_traj_txt_path, syn_traj_txt_path,
                       max_seq_len=30,  # 30, 50
                       metric_iteration=10,
                       out_data_path='./metrics_result/porto20_b4/3_ttsgan/'  # output path
                       # 1_seqgan
                       # 2_timegan
                       # 3_tts_gan
                       # 4_lbs
                       # 5_ours
                       # 6_ours_no_pred
                       )
