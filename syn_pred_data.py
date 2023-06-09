## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 1. TimeGAN model
# from syngan_model import syngan
from one_stage_syngan_model import syngan
from two_stage_predgan_model import predgan

# 2. Data loading
from data_loading import read_txt_traj_data_loading
from utils import padding, getmaxlen, formalize_data
import time
import pathlib


def main(args):
    """Main function for experiments.

    Args:
      - max_seq_len: max trajectory length
      - cut_seq_len + 1: one stage begin trajectory length (default 9 + 1)
      - pred_seq_len: two stage predicted trajectory length (default 1)
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation

    Returns:
      - None
    """
    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    # learning_rate = 0.001(default)
    parameters = dict()
    parameters['dataset_path'] = args.dataset_path
    pathlib.Path(args.out_data_path).mkdir(parents=True, exist_ok=True)
    parameters['out_data_path'] = args.out_data_path
    parameters['cell_num'] = args.cell_num
    parameters['boundary'] = args.boundary
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['max_seq_len'] = args.max_seq_len
    parameters['cut_seq_len'] = args.cut_seq_len
    parameters['pred_seq_len'] = args.pred_seq_len
    data_output = args.out_data_path + 'syn_data_output/'
    pathlib.Path(data_output).mkdir(parents=True, exist_ok=True)

    print("Start data loading...")
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("----------------------------------------------")
    ## Data loading
    # one_stage_train_data: 带length (?, cut_seq_len + 1[begin point] + 1[length], 3)
    # two_stage_train_data: (?, cut_seq_len, 3)
    one_stage_train_data, two_stage_train_data, _ = \
        read_txt_traj_data_loading(args.dataset_path, begin_seq_len=args.cut_seq_len + 1,
                                   pred_seq_len=args.pred_seq_len,
                                   max_seq_len=args.max_seq_len, boundary=args.boundary, with_end_sign=False, for_test=False)

    print("1.1 Begin one stage to synthese initial trajectory segments ")
    # to_end_generated_data: (m, ?[len<=5], 3)
    # to_two_stage_generated_data: (n, cut_seq_len + 1[begin point], 3)
    # generated_data_length: (?, )
    to_end_generated_data, to_two_stage_generated_data, generated_data_length \
        = syngan(one_stage_train_data, parameters)
    # print("generated_data's shape = " + str(np.array(generated_data).shape))
    print("generated_data_length's shape = " + str(np.array(generated_data_length).shape))

    # co_name = ['lat', 'lng', 'time_float']
    # write_data2csv(np.expand_dims(np.expand_dims(generated_data_length, axis=1), axis=1), 1, co_name,
    #                data_output + "pred_begin_trajectory_length.csv")
    print("One Stage initial trajectory segments Syned Successfully")
    print("----------------------------------------------")
    print("2.1 Begin to predict trajectory ")

    predicted_data_m = predgan(two_stage_train_data, to_two_stage_generated_data, generated_data_length, parameters)

    print("Trajectory Predicted Successfully")

    result_traj = []
    result_traj_with_time = []

    # One Stage
    for single_traj in to_end_generated_data:
        traj = []
        traj_with_time = []
        for single_point in single_traj:
            traj.append(single_point[:-1])
            traj_with_time.append(single_point)
        result_traj.append(traj)
        result_traj_with_time.append(traj_with_time)

    # Two Stage
    for single_traj in predicted_data_m:
        traj = []
        traj_with_time = []
        for single_point in single_traj:
            traj.append(single_point[:-1])
            traj_with_time.append(single_point)
        result_traj.append(traj)
        result_traj_with_time.append(traj_with_time)

    predicted_data_to_txt_pic = np.array(result_traj)
    predicted_data_with_time_to_txt_pic = np.array(result_traj_with_time)
    max_len, min_len = getmaxlen(predicted_data_to_txt_pic)

    # save trajectory result without time info
    out_path_txt = data_output + 'formalized_txt_syn' + str(args.cut_seq_len) + '_pred' + str(args.pred_seq_len) + \
                   '_min' + str(min_len) + '_max' + str(max_len) + '.txt'
    formalize_data(predicted_data_to_txt_pic, out_path_txt, is_timeinterval=True, without_time=True)

    # save trajectory result with time info
    out_path_txt_with_time = data_output + 'formalized_txt_syn' + str(args.cut_seq_len) + '_pred' + str(
        args.pred_seq_len) + \
                             '_min' + str(min_len) + '_max' + str(max_len) + '_with_time.txt'
    formalize_data(predicted_data_with_time_to_txt_pic, out_path_txt_with_time, is_timeinterval=True,
                   without_time=False)

    # write_data2csv(predicted_data_with_time_to_txt_pic, args.max_seq_len, co_name,
    #                data_output + "two_stage_syn_condition_len" + str(args.cut_seq_len) +
    #                "_pred_len" + str(args.pred_seq_len) +
    #                "_trajectory_data.csv")
    print('Finish Synthetic and Predicted Trajectory')
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        # default="./data/porto_grid20_b5_2w.txt",
        default="./data/porto_grid20_b4_2w.txt",
        type=str)
    parser.add_argument(
        '--out_data_path',
        # default='./4_22_geo_grid6_final/syn_5_pred_4/',
        default='./5_26_boundary4_porto20/syn_9_pred_1/',
        type=str)
    parser.add_argument(
        '--cell_num',
        # default=6,
        default=20,
        type=int)
    parser.add_argument(
        '--boundary',
        # default=(30.822000, 31.999900, 120.203300, 121.990000),  # shanghai
        # default=(39.827000, 39.989000, 116.282000, 116.493000),  # t-drive
        # default=(39.788000, 40.093000, 116.148000, 116.612000),  # geo-life
        # default=(41.06421, 41.20999, -8.661858, -8.525016),  # porto_b5
        default=(41.10421, 41.24999, -8.665258, -8.528333),  # porto_b4
        type=tuple)
    parser.add_argument(
        '--max_seq_len',
        help='predict sequence length',
        default=30,
        type=int)
    parser.add_argument(
        '--cut_seq_len',
        help='initial conditional sequence length',
        default=9,
        type=int)
    parser.add_argument(
        '--pred_seq_len',
        help='initial predicted sequence length',
        default=1,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=64,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=1,
        type=int)

    args = parser.parse_args()
    # Calls main function
    main(args)
