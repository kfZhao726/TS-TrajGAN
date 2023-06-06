## Necessary Packages
import csv
import pathlib
import time
import warnings

from geopy.distance import geodesic
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
import os
from PIL import Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow.compat.v1 as tf


def padding(dataset, max_seq_len):
    # 用0填充
    padding_d = list()
    padding_element = list()
    for n in range(len(dataset[0][0])):
        padding_element.append(0.0)
    for data in dataset:
        if len(data) <= max_seq_len:
            for _ in range(len(data), max_seq_len):
                data = np.insert(data, len(data), padding_element, axis=0)
        padding_d.append(data)
    return padding_d


def padding_end_point(dataset, max_seq_len):
    # 用终点填充
    padding_d = list()
    for data in dataset:
        if len(data) <= max_seq_len:
            padding_element = list()
            padding_element.append((data[-1][0], data[-1][1], 0.0))
            for _ in range(len(data), max_seq_len):
                data = np.insert(data, len(data), padding_element, axis=0)
        padding_d.append(data)
    return padding_d


def MinMaxScaler(data):
    """Min-Max Normalizer.

Args:
  - data: raw data

Returns:
  - norm_data: normalized data
  - min_val: minimum values (for renormalization)
  - max_val: maximum values (for renormalization)
"""
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)  # 原来是e-7

    return norm_data, min_val, max_val
# f(z) = 1 / (1+e**-z)

def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        # 查询每一条seq的长度
        max_seq_len = max(max_seq_len, len(data[i][:]))
        time.append(len(data[i][:]))
        # time:list返回一个seq长度序列 T = {T1,T2……Tm}
    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.

    Args:
      - module_name: gru, lstm, or lstmLN

    Returns:
      - rnn_cell: RNN Cell
    """
    assert module_name in ['gru', 'lstm', 'lstmLN']

    # GRU , return new_h, new_h 只输出最后一个时间步的隐藏层
    if (module_name == 'gru'):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    # LSTM , return new_h, new_state 输出(最后一个时间步的隐藏层,最后一个时间步的输出状态)
    elif (module_name == 'lstm'):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    # LSTM Layer Normalization
    elif (module_name == 'lstmLN'):
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        # 对每个元素添加一个随机的小量
        # temp_Z += np.random.normal(loc=0.0, scale=0.01, size=[T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb

# def random_generator(batch_size, z_dim, T_mb, max_seq_len):
#     """Random vector generation.
#     Args:
#         - batch_size: size of the random vector
#         - z_dim: dimension of random vector
#         - T_mb: time information for the random vector
#         - max_seq_len: maximum sequence length
#     Returns:
#         - Z_mb: generated random vector
#     """
#     Z_mb = list()
#     for i in range(batch_size):
#         temp = np.zeros([max_seq_len, z_dim])
#         temp_Z = np.random.normal(loc=0.0, scale=1.0, size=[T_mb[i], z_dim])
#         # 对每个元素添加一个随机的小量
#         # temp_Z += np.random.normal(loc=0.0, scale=0.001, size=[T_mb[i], z_dim])
#         temp[:T_mb[i], :] = temp_Z
#         Z_mb.append(temp)
#     return Z_mb

# def random_generator(batch_size, z_dim, T_mb, max_seq_len):
#     """Random vector generation.
#
#     Args:
#       - batch_size: size of the random vector
#       - z_dim: dimension of random vector
#       - T_mb: time information for the random vector
#       - max_seq_len: maximum sequence length
#
#     Returns:
#       - Z_mb: generated random vector
#     """
#     Z_mb = list()
#     for i in range(batch_size):
#         temp = np.zeros([max_seq_len, z_dim])
#         Z_loc = np.random.uniform(low=0., high=1, size=[T_mb[i], 2])
#         Z_time = np.random.rand(T_mb[i], 1)
#         temp_Z = np.concatenate((Z_loc, Z_time),axis=1)
#         temp[:T_mb[i], :] = temp_Z
#         Z_mb.append(temp_Z)
#     return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


# def comp_speed(coor_a, coor_b):
#     distance = geodesic(coor_a[0:2], coor_b[0:2]).m
#     # time_interval = round((coor_b[2] - coor_a[2]), 6)
#     time_interval = coor_b[2]
#     # 1m/s = 3.6 km/h
#     speed = (distance / time_interval) * 3.6
#     return speed


# def check_speed(traj_piece, limited_speed):
#     for i in traj_piece:
#         speed = comp_speed(i)


def write_data2csv(syn_dataset, max_seq_len, column_name, output_path):
    final_predicted_data = list()

    if len(column_name) > 3:
        new_dataset = list()
        for a in syn_dataset:
            data_list = np.zeros((a.shape[0], len(column_name)))
            data_list[0] = np.insert(a[0], len(a[0]), (0, 0, 0), axis=0)
            for i in range(1, len(a)):
                distance = round(geodesic(a[i - 1][0:2], a[i][0:2]).m, 6)
                time_interval = round((a[i][2] - a[i - 1][2]), 6)
                # 1m/s = 3.6 km/h
                speed = (distance / time_interval) * 3.6
                data_list[i] = np.insert(a[i], len(a[i]), (distance, time_interval, speed), axis=0)
            new_dataset.append(data_list)

        new_dataset = padding(new_dataset, max_seq_len)

        for i in range(len(new_dataset)):
            for j in range(len(new_dataset[i])):
                final_predicted_data.append(list(new_dataset[i][j]))

    if len(column_name) == 3:
        syn_dataset = padding(syn_dataset, max_seq_len)
        for i in range(len(syn_dataset)):
            for j in range(len(syn_dataset[i])):
                final_predicted_data.append(list(syn_dataset[i][j]))

    final_predicted_data = np.round(final_predicted_data, 8)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_name)
        for row in final_predicted_data:
            writer.writerow(row)


# def check_distance(traj, syn_seq_len, pred_seq_len):
#     temp = 0
#     for i in range(syn_seq_len + 1, len(traj) - pred_seq_len):
#         temp_point = traj[i]
#         distance_list = list()
#         for j in range(1, pred_seq_len + 1):
#             distance = geodesic(temp_point, traj[i + j]).km
#             distance_list.append(distance)
#             #         print(distance_list)
#         if np.all(np.array(distance_list) < 0.01):
#             return i + pred_seq_len + 1
#         else:
#             temp = i
#     return temp + pred_seq_len + 1


def getCellIndex(boundary, cell_num, each_coor):  # coor是个1*2数组，boundary是个1*4数组

    lat_cell_count = cell_num
    lng_cell_count = cell_num
    # boundary = getboundary(coor)
    height_cell = (boundary[1] - boundary[0]) / lat_cell_count
    width_cell = (boundary[3] - boundary[2]) / lng_cell_count

    cloumnindex = int((float(each_coor[1]) - boundary[2]) / width_cell)
    rowindex = int((boundary[1] - float(each_coor[0])) / height_cell)

    if rowindex >= lat_cell_count:
        rowindex -= 1
    if cloumnindex >= lng_cell_count:
        cloumnindex -= 1

    cellindex = rowindex * lng_cell_count + cloumnindex
    if cellindex >= lat_cell_count * lng_cell_count:
        print("something is wrong..")

    return cellindex


def merge_traj_data(boundary, cell_num, traj_dataset):
    merge_traj_list = list()
    for each_traj in traj_dataset:
        temp_index = 0
        merge_traj = list()
        time_info = 0
        for each_coor in each_traj:
            each_coor = np.array(each_coor)
            cell_index = getCellIndex(boundary, cell_num, each_coor[:2])
            if cell_index != temp_index:
                merge_traj.append(list((each_coor[0], each_coor[1], round(each_coor[2] + time_info, 6))))
                # merge_traj.append(list((each_coor[0], each_coor[1])))
                temp_index = cell_index
                time_info = 0
            else:
                time_info += each_coor[-1]
        merge_traj_list.append(merge_traj)
    return merge_traj_list


def getmaxlen(data):
    seq_len_list = list()
    for i in data:
        seq_len_list.append(len(i))

    seq_len_list = np.array(seq_len_list)
    res_max = np.max(seq_len_list)
    res_min = np.min(seq_len_list)
    return res_max, res_min


def draw_loss_pic(losses_list, save_path):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(range(len(losses_list)), losses_list)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(save_path, format='png')
    plt.close()


def formalize_data(data_list, out_path, is_timeinterval, without_time=True):
    float_begin_time = time.mktime(time.strptime('2008-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
    with open(out_path, 'w') as f:
        for i in range(len(data_list)):
            f.write("#" + str(i) + ":" + "\r")
            f.write(">0:")
            if without_time:
                for j in data_list[i]:
                    f.write(str(round(j[0], 6)) + "," + str(round(j[1], 6)) + ";")
            else:
                time_info_sum = 0
                for j in range(len(data_list[i])):
                    lng = str(round(data_list[i][j][0], 6))
                    lat = str(round(data_list[i][j][1], 6))
                    single_time_info = float(data_list[i][j][2])
                    time_info_sum += single_time_info
                    if is_timeinterval:
                        time_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_info_sum + float_begin_time))
                    else:
                        time_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(single_time_info + float_begin_time))
                    f.write(lng + "," + lat + "," + time_info + ";")
            f.write("\r")
    print('write sucess!')


def getCellIndex_list(coor, boundary, goal_size):
    index_list = []
    lat_cell_count = goal_size
    lng_cell_count = goal_size
    height_cell = (boundary[1] - boundary[0]) / lat_cell_count
    width_cell = (boundary[3] - boundary[2]) / lng_cell_count
    for each_coor in coor:
        cloumnindex = int((float(each_coor[1]) - boundary[2]) / width_cell)
        rowindex = int((boundary[1] - float(each_coor[0])) / height_cell)
        if rowindex >= lat_cell_count:
            rowindex -= 1
        if cloumnindex >= lng_cell_count:
            cloumnindex -= 1
        cellindex = rowindex * lng_cell_count + cloumnindex
        if cellindex >= lat_cell_count * lng_cell_count:
            print("something is wrong..")
        index_list.append((rowindex, cloumnindex))
    return list(set(index_list))


def draw_piexl_pic(traj, boundary, goal_size, out_path, filename):
    out_demo_img = out_path + filename + '.png'
    imgWhite = np.zeros((goal_size, goal_size), dtype=np.uint8) + 255
    index_list = np.array(getCellIndex_list(traj, boundary, goal_size))
    for index in index_list:
        imgWhite[index[0], index[1]] = 0
    result_img = Image.fromarray(imgWhite)
    result_img.save(out_demo_img)


def draw_pic(traj_list, boundary, goal_size, outpath):
    count = 0
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    for traj in traj_list:
        filename = 'traj_' + str(count)
        draw_piexl_pic(traj, boundary, goal_size, outpath, filename)
        count += 1
    print('syn pic sucess')
