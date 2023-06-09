## Necessary Packages
import math
import time
import numpy as np
import pandas as pd


def get_float_time(timestamp, begin_float_time):
    if len(timestamp) == 19:
        return time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')) - begin_float_time
    else:
        return time.mktime(time.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')) - begin_float_time


def read_txt_traj_data_loading(txt_path, begin_seq_len=5, pred_seq_len=1, max_seq_len=50,
                               boundary=(39.788000, 40.093000, 116.148000, 116.612000), with_end_sign=False, for_test=False):

    end_sign = list([boundary[0] - (boundary[1] - boundary[0]), boundary[2] - (boundary[-1] - boundary[-2]), -1])
    with open(txt_path, 'r') as f:
        con = f.readlines()[:]
    one_stage_read_begin_traj = list()
    two_stage_read_cut_traj = list()
    metrics_traj = list()
    # time_interval_list = list()

    for i in range(int(len(con) / 2)):
        traj = con[2 * i + 1].strip('>0:').strip('\n').split(';')[:-1]
        traj_begin_str_time = traj[0][-19:-9] + ' 00:00:00'
        traj_begin_float_time = time.mktime(time.strptime(traj_begin_str_time, '%Y-%m-%d %H:%M:%S'))
        each_traj = list()
        for i in range(len(traj)):
            traj[i] = traj[i].split(',')
            if i == 0:
                time_info = get_float_time(traj[i][2], traj_begin_float_time)
            else:
                time_info_next = get_float_time(traj[i][2], traj_begin_float_time)
                time_info_before = get_float_time(traj[i - 1][2], traj_begin_float_time)
                time_info = time_info_next - time_info_before
                # time_interval_list.append(time_info)
            if for_test:
                coor = (float(time_info), float(traj[i][1]), float(traj[i][0]))
            else:
                coor = (float(traj[i][0]), float(traj[i][1]), float(time_info))
            each_traj.append(coor)

        metrics_traj.append(each_traj[:max_seq_len])
        # the train set of one stage
        if len(each_traj) < begin_seq_len:
            each_length = math.nan
            temp_list = each_traj
        else:
            each_length = len(each_traj)
            temp_list = each_traj[:begin_seq_len]
        temp_list.append((each_length, each_length, each_length))
        one_stage_read_begin_traj.append(temp_list)

        # the train set of two stage
        cut_seq_len = begin_seq_len - 1 + pred_seq_len
        each_for_cut = each_traj[1:]
        if len(each_for_cut) < cut_seq_len:
            pass
        else:
            # each_for_cut = np.array(each_for_cut)
            # for j in range(len(each_for_cut) - cut_seq_len + 1):
            #     cut_list = each_for_cut[j: j + cut_seq_len, :]
            #     two_stage_read_cut_traj.append(cut_list)
            if with_end_sign:
                each_for_cut = np.array(each_for_cut)
                new_cut = np.insert(each_for_cut, len(each_for_cut), end_sign, axis=0)
            else:
                new_cut = np.array(each_for_cut)
            for j in range(len(new_cut) - cut_seq_len + 1):
                cut_list = new_cut[j: j + cut_seq_len, :]
                two_stage_read_cut_traj.append(cut_list)

    one_stage_read_begin_traj = one_stage_read_begin_traj[::-1]
    idx = np.random.permutation(len(one_stage_read_begin_traj))
    one_stage_data = []
    for i in range(len(one_stage_read_begin_traj)):
        one_stage_data.append(one_stage_read_begin_traj[idx[i]])

    print("the first GAN 's train dataset's shape is " + str(np.array(one_stage_data).shape[0]))

    # shuffle data
    two_stage_read_cut_traj = two_stage_read_cut_traj[::-1]
    idx = np.random.permutation(len(two_stage_read_cut_traj))
    two_stage_data = []
    for i in range(len(two_stage_read_cut_traj)):
        two_stage_data.append(two_stage_read_cut_traj[idx[i]])

    print("the second CGAN 's train dataset's shape is" + str(np.array(two_stage_data).shape))

    metrics_traj = metrics_traj[::-1]
    print("the metrics data's shape is " + str(np.array(metrics_traj).shape[0]))

    return one_stage_data, two_stage_data, metrics_traj


