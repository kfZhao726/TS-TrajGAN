# Necessary Packages
# import warnings
# warnings.filterwarnings("ignore")
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import os
import pathlib

import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt

tf.disable_eager_execution()
import numpy as np

from utils import extract_time, rnn_cell, padding, draw_loss_pic, random_generator
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, MinMaxScaler


def predgan(ori_data, pre_syn_data_before, pre_syn_data_before_length, parameters):
    """predgan function.
  
      Use original trajectory cut as training set to predict trajectory data

      Args:
        - ori_data: original trajectory cut data
        - pre_syn_data_before: previous syn trajectory piece data
        - pre_syn_data_before_length: previous syn piece traj data's length
        - parameters: predgan network parameters

      Returns:
        - predicted_data: predicted trajectory data
      """
    # Initialization on the Graph
    tf.reset_default_graph()

    # utils
    def extract_time(data, pred_seq_len):
        time = list()
        for i in range(len(data)):
            time.append(pred_seq_len)
        return time

    pred_seq_len = parameters['pred_seq_len']
    ori_time = extract_time(ori_data, pred_seq_len)

    def rnn_cell(module_name, hidden_dim, name):
        """Basic RNN Cell.

        Args:
          - module_name: gru, lstm, or lstmLN

        Returns:
          - rnn_cell: RNN Cell
        """
        assert module_name in ['gru', 'lstm', 'lstmLN']

        # GRU , return new_h, new_h 只输出最后一个时间步的隐藏层
        if (module_name == 'gru'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name=name)
        # LSTM , return new_h, new_state 输出(最后一个时间步的隐藏层,最后一个时间步的输出状态)
        elif (module_name == 'lstm'):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        # LSTM Layer Normalization
        elif (module_name == 'lstmLN'):
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        return rnn_cell

    def batch_generator(data, time, batch_size):
        """Mini-batch generator.

        Args:
          - data: trajectory cut data
          - time: time information
          - batch_size: the number of samples in each batch

        Returns:
          - X_mb: trajectory piece data in each batch
          - C_mb: condition data in each batch
          - T_mb: time information in each batch
        """
        no = len(data)
        idx = np.random.permutation(no)
        train_idx = idx[:batch_size]

        X_mb = list(data[i][-pred_seq_len:] for i in train_idx)
        C_mb = list(data[i][0: -pred_seq_len] for i in train_idx)
        T_mb = list(time[i] for i in train_idx)
        return X_mb, C_mb, T_mb

    ## Build a RNN networks
    # Network Parameters
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    cut_seq_len = parameters['cut_seq_len']
    condi_len = cut_seq_len
    hidden_dim = parameters['hidden_dim']
    # hidden_dim = 100
    max_traj_len = parameters['max_seq_len']
    out_data_path = parameters['out_data_path']
    boundary = parameters['boundary']
    cell_num = parameters['cell_num']

    # Basic Parameters(trajectory)
    no, seq_len, dim = np.asarray(ori_data).shape
    syn_no, syn_seq_len, syn_dim = np.asarray(pre_syn_data_before).shape
    # hidden_dim = condi_len * dim

    # 这里要去掉每条轨迹的第一个点（上一个模型生成的 --> 所以第一阶段要生成condition + 1 长度的轨迹）
    first_point_list = list()
    pre_syn_data = list()
    for i in range(len(pre_syn_data_before)):
        first_point_list.append(pre_syn_data_before[i][0])
        pre_syn_data.append(pre_syn_data_before[i][1:])

    ori_data = np.array(ori_data)
    ori_data_nolabel = np.array(ori_data)
    _, seq_len_ori, dim_ori = np.asarray(ori_data_nolabel).shape
    ori_data_nolabel = np.reshape(ori_data_nolabel, (-1, dim_ori))
    no_ori = ori_data_nolabel.shape[0]

    pre_syn_data = np.array(pre_syn_data)
    no_pre, seq_len_pre, dim_pre = np.asarray(pre_syn_data).shape
    pre_syn_data = np.reshape(pre_syn_data, (-1, dim_pre))

    norm_data = np.concatenate((ori_data_nolabel, pre_syn_data), axis=0)
    stand_scaler = MinMaxScaler()
    stand_scaler.fit_transform(norm_data)
    norm_data = stand_scaler.transform(norm_data)

    # data for train
    ori_norm_data = norm_data[:no_ori].reshape(-1, seq_len_ori, dim_ori)
    # data for test(predict)
    pre_norm_data = norm_data[no_ori:].reshape(-1, seq_len_pre, dim_pre)

    for i in range(len(ori_data)):
        ori_data[i, :, :] = ori_norm_data[i, :, :]


    z_dim = dim
    gamma = 1
    beta = 1

    # Input place holders
    X = tf.placeholder(tf.float32, [None, pred_seq_len, dim], name="myinput_x")  # 每个切片的最后一个点，shape = (?, 1, 3)
    C = tf.placeholder(tf.float32, [None, condi_len, dim], name="my_input_c")
    Z = tf.placeholder(tf.float32, [None, pred_seq_len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")

    def Multi_denes(input, type):
        input = tf.layers.dense(input, int(hidden_dim * pred_seq_len), activation=tf.nn.relu, name=type + '_dense4')
        return input

    def embedder(X, C, T):
        """Encoding network between original feature space to latent space.

        Args:
          - X: input trajectory piece features
          - C: input condition features
          - T: input time information

        Returns:
          - H: encodings
        """
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            X = Multi_denes(X, 'x')
            C_d = Multi_denes(tf.layers.flatten(C), 'c')
            C_d = tf.reshape(C_d, (-1, pred_seq_len, hidden_dim))
            E_Input = tf.concat([X, C_d], axis=2)
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim, name='e_cell') for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, E_Input, dtype=tf.float32, sequence_length=T)
            H = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='e_out_dense')
        return H

    def recovery(H, C, T):
        """recovery network from latent space to original space.

        Args:
          - H: latent representation
          - C: input condition features
          - T: input time information

        Returns:
          - X_tilde: recovered data
        """
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            r_cell = tf.keras.layers.StackedRNNCells(
                [rnn_cell(module_name, hidden_dim, name='r_cell') for _ in range(num_layers)])  # module_name = 'gru'
            C_d = Multi_denes(tf.layers.flatten(C), 'c')
            C_d = tf.reshape(C_d, (-1, pred_seq_len, hidden_dim))
            H = Multi_denes(H, 'h')
            input_H = tf.concat([H, C_d], axis=2)
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, input_H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.layers.dense(r_outputs, dim, activation=tf.nn.sigmoid, name='out_dense')
        return X_tilde

    def generator(Z, C, T):
        """Generator function: Generate trajectory data in latent space.

        Args:
          - Z: random variables
          - C: input condition features
          - T: input time information

        Returns:
          - E: generated encoding
        """
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            Z = Multi_denes(Z, 'z')
            C_d = Multi_denes(tf.layers.flatten(C), 'c')
            C_d = tf.reshape(C_d, (-1, pred_seq_len, hidden_dim))
            E_Input = tf.concat([Z, C_d], axis=2)
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim, name='e_cell') for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, E_Input, dtype=tf.float32, sequence_length=T)
            E = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='out_dense')
        return E

    def supervisor(H, T):
        """Generate next sequence using the previous sequence.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - S: generated sequence based on the latent representations generated by the generator
        """
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            s_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim, name='s_cell') for _ in range(num_layers - 1)])
            H = Multi_denes(H, 'h')
            e_outputs, e_last_states = tf.nn.dynamic_rnn(s_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='out_dense')
        return S

    def discriminator(H, C, T):
        """Discriminate the original and synthetic trajectory data.

        Args:
          - H: latent representation
          - C: input condition features
          - T: input time information

        Returns:
          - Y_hat: classification results between original and synthetic trajectory
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim, name='d_cell') for _ in range(num_layers)])
            C_d = Multi_denes(tf.layers.flatten(C), 'c')
            C_d = tf.reshape(C_d, (-1, pred_seq_len, hidden_dim))
            H = Multi_denes(H, 'h')
            input_Out = tf.concat([H, C_d], axis=2)
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, input_Out, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.layers.dense(d_outputs, 1, activation=None, name='out_dense4')
        return Y_hat

    # embedder & recovery
    H = embedder(X, C, T)  # (?, pred_len, hidden_dim)
    X_tilde = recovery(H, C, T)  # (?, pred_len, ori_dim)

    # Generator
    E_hat = generator(Z, C, T)  # （?, pred_len, hidden_dim）
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, C, T)

    # Discriminator
    Y_real = discriminator(H, C, T)
    Y_fake_e = discriminator(E_hat, C, T)
    Y_fake = discriminator(H_hat, C, T)

    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    G_loss_S = tf.losses.mean_squared_error(H[:, :, :], H_hat_supervise[:, :, :])

    # 2. G_loss: 计算目标点潜码和预测点的MSE距离
    G_loss_V = tf.losses.mean_squared_error(X, X_hat)

    # 3. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * tf.sqrt(G_loss_V)

    # embedder network loss
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    # 单独训练
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
    # 联合训练
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    # learning_rate=0.0005
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)

    ## predgan training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()  ##:如果你指定的设备不存在,允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True  ##动态分配内存
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # print("*"*50)

    saver_en_de = tf.train.Saver(var_list=e_vars + r_vars)
    saver_sup = tf.train.Saver(var_list=g_vars + s_vars)
    saver_joint = tf.train.Saver(var_list=e_vars + r_vars + d_vars + g_vars + s_vars)

    save_model_para_path = out_data_path + "/Two_Stage_Model/"
    save_en_de_para_path = save_model_para_path + "Em_recovery_Model/"
    save_sup_para_path = save_model_para_path + "Sup_Model/"
    save_gan_joint_para_path = save_model_para_path + "GAN_joint_Model/"
    pathlib.Path(save_model_para_path).mkdir(parents=True, exist_ok=True)

    losses_only = {'embedder_loss': [], 'sup_loss': []}
    losses_joint = {'embedder_loss': [], 'G_loss': [], 'D_loss': []}
    save_path = out_data_path + 'loss_pic/two_stage/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    print('Traing Parameters:')
    print('batch_size = ' + str(batch_size))
    print('hidden_dim = ' + str(hidden_dim))
    print('num_layers = ' + str(num_layers))
    print('module_name = ' + str(module_name))

    print_iter = 1000
    num_epochs = int(iterations / print_iter)

    if os.path.exists(save_en_de_para_path + 'checkpoint'):
        print("加载已有二阶段编解码器模型参数")
        saver_en_de.restore(sess, save_en_de_para_path + 'en_de_model.ckpt')
    else:
        pathlib.Path(save_en_de_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练二阶段编解码器模型...")
        print('Start Encoding Network Training')
        # 1. Embedding network training(embedder & recovery)
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                X_mb, C_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
                # Train embedder
                _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, C: C_mb, T: T_mb})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(e_loss=np.round(np.sqrt(step_e_loss), 4))
                losses_only['embedder_loss'].append(np.round(np.sqrt(step_e_loss), 4))
        draw_loss_pic(losses_only['embedder_loss'], save_path + 'two_stage_' + 'embedder_only_loss.png')
        save_path1 = os.path.join(save_en_de_para_path, "en_de_model.ckpt")
        os.chmod(save_en_de_para_path, 0o755)  # 修改文件夹权限
        saver_en_de.save(sess, save_path1)
        print('Finish embedder & recovery Network Training')

    if os.path.exists(save_sup_para_path + 'checkpoint'):
        print("加载已有二阶段监督器模型参数")
        saver_sup.restore(sess, save_sup_para_path + 'sup_model.ckpt')
    else:
        pathlib.Path(save_sup_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练二阶段监督器模型...")
        # 2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Set mini-batch
                X_mb, C_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, pred_seq_len)
                # Train generator
                _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, C: C_mb, X: X_mb, T: T_mb})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(s_loss=np.round(np.sqrt(step_g_loss_s), 4))
                losses_only['sup_loss'].append(np.round(np.sqrt(step_g_loss_s), 4))
        draw_loss_pic(losses_only['sup_loss'], save_path + 'two_stage_' + 'sup_only_loss.png')
        save_path2 = os.path.join(save_sup_para_path, "sup_model.ckpt")
        os.chmod(save_sup_para_path, 0o755)  # 修改文件夹权限
        saver_sup.save(sess, save_path2)
        print('Finish Training with Supervised Loss Only')

    if os.path.exists(save_gan_joint_para_path + 'checkpoint'):
        print("加载已有二阶段联合训练模型参数")
        saver_joint.restore(sess, save_gan_joint_para_path + 'joint_model.ckpt')
    else:
        pathlib.Path(save_gan_joint_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练二阶段联合训练模型...")
        # 3. Joint Training(5models)
        print('Start Generator and discriminator Training')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Generator training
                # Set mini-batch
                for _ in range(2):
                    X_mb, C_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
                    # Random vector generation
                    Z_mb = random_generator(batch_size, z_dim, T_mb, pred_seq_len)
                    # Train generator
                    _, step_g_loss_u, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_V],
                                                               feed_dict={Z: Z_mb, C: C_mb, X: X_mb, T: T_mb})
                    # Train embedder
                    _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0],
                                                 feed_dict={Z: Z_mb, X: X_mb, C: C_mb, T: T_mb})

                # Discriminator training
                # Set mini-batch
                X_mb, C_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, pred_seq_len)
                # Train discriminator
                step_d_loss = sess.run(D_loss, feed_dict={X: X_mb, C: C_mb, T: T_mb, Z: Z_mb})
                if step_d_loss > 0.4:
                    _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, C: C_mb, T: T_mb, Z: Z_mb})
                # Print multiple checkpoints
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(d_loss=np.round(step_d_loss, 4),
                                 g_loss=np.round(step_g_loss_u + step_g_loss_v, 4),
                                 e_loss_t=np.round(np.sqrt(step_e_loss_t0), 4))
                losses_joint['G_loss'].append(np.round(step_g_loss_u + step_g_loss_v, 4))
                losses_joint['D_loss'].append(np.round(step_d_loss, 4))
                losses_joint['embedder_loss'].append(np.round(np.sqrt(step_e_loss_t0), 4))
        draw_loss_pic(losses_joint['embedder_loss'], save_path + 'two_stage_' + 'embedder_joint_loss.png')
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(range(len(losses_joint['G_loss'])), losses_joint['G_loss'], color='r', label='Generator')
        plt.plot(range(len(losses_joint['D_loss'])), losses_joint['D_loss'], color='b', label='Discriminator')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss')
        plt.legend()
        plt.savefig(save_path + 'two_stage_' + 'G&D_joint_loss.png', format='png')
        plt.close()
        print('Finish Generator and discriminator Training')
        save_path3 = os.path.join(save_gan_joint_para_path, "joint_model.ckpt")
        os.chmod(save_gan_joint_para_path, 0o755)  # 修改文件夹权限
        saver_joint.save(sess, save_path3)
    print('Start Trajectory Predicting...')

    def checkCellIndex(each_coor):  # coor是个1*2数组，boundary是个1*4数组
        lat_cell_count = cell_num
        lng_cell_count = cell_num
        if boundary[0] < float(each_coor[0]) < boundary[1] and boundary[2] < float(each_coor[1]) < boundary[3]:
            height_cell = (boundary[1] - boundary[0]) / lat_cell_count
            width_cell = (boundary[3] - boundary[2]) / lng_cell_count
            cloumnindex = int((float(each_coor[1]) - boundary[2]) / width_cell)
            rowindex = int((boundary[1] - float(each_coor[0])) / height_cell)

            return (rowindex, cloumnindex)
        else:
            return False

    def check_traj_cell_length(traj, traj_length):
        count = 0
        temp_index = (0, 0)
        index = 0
        for i in range(len(traj)):
            coor_cell_index = checkCellIndex(traj[i])
            if coor_cell_index:
                if coor_cell_index != temp_index:
                    count += 1
                    temp_index = coor_cell_index
                    if count >= traj_length:
                        return index + 1
                    else:
                        pass
                else:
                    pass
            else:
                pass
            index += 1
        return index

    def predict(input_traj):
        c = list()
        c.append(input_traj[-condi_len:, :])
        while True:
            if len(input_traj) < max_traj_len - 1:
                noise = np.array(random_generator(1, z_dim, np.array([pred_seq_len]), pred_seq_len))
                predict_data_curr = sess.run(X_hat, feed_dict={Z: noise, C: c, T: np.array([pred_seq_len])})
                input_traj = np.append(input_traj, predict_data_curr[0, :, :], axis=0)
                c = np.append(c, predict_data_curr, axis=1)
                c = c[:, -condi_len:, :]
            else:
                break
        return input_traj

    generated_data = list()

    total = len(pre_norm_data)
    desc = 'Predicting Trajectory'
    position = 0
    for i, each_traj in tqdm(enumerate(pre_norm_data), total=total, desc=desc, position=position, miniters=1):
        traj_length = math.floor(pre_syn_data_before_length[i]) + 1
        if traj_length <= max_traj_len:
            pass
        else:
            traj_length = max_traj_len
        new_each_traj = stand_scaler.inverse_transform(predict(each_traj))
        new_each_traj = np.insert(new_each_traj, 0, first_point_list[i], axis=0)
        index = check_traj_cell_length(new_each_traj, traj_length)
        cut_traj = new_each_traj[:index]
        generated_data.append(cut_traj)

    print("prediction done")
    print("已成功生成预测了" + str(len(generated_data)) + "条轨迹")

    return generated_data
