# Necessary Packages
# import warnings
# warnings.filterwarnings("ignore")
import math
import os
import pathlib

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np
from utils import extract_time, rnn_cell, random_generator, padding, draw_loss_pic
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, MinMaxScaler


def syngan(ori_data, parameters):
    """syngan function.

        Use original begin trajectory data as training set to generate begin trajectory data

        Args:
            - ori_data: original begin trajectory data
            - parameters: syngan network parameters

        Returns:
            - generated_data: generated begin trajectory data
        """
    # Initialization on the Graph
    tf.reset_default_graph()

    # utils
    def batch_generator(data, time, batch_size):
        """Mini-batch generator.

        Args:
          - data: begin trajectory data
          - time: time information
          - batch_size: the number of samples in each batch

        Returns:
          - X_mb: begin trajectory data in each batch
          - C_mb: condition data in each batch
          - T_mb: time information in each batch
        """
        no = len(data)
        idx = np.random.permutation(no)
        train_idx = idx[:batch_size]

        X_mb = list(data[i][:-1] for i in train_idx)
        Length_mb = list(data[i][-1][0] for i in train_idx)
        T_mb = list(time[i] for i in train_idx)
        Length_mb = np.expand_dims(Length_mb, axis=1)

        return X_mb, Length_mb, T_mb

    ## Build a RNN networks
    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    out_data_path = parameters['out_data_path']

    # Maximum sequence length and each sequence length
    max_seq_len = parameters['cut_seq_len'] + 1
    pred_traj_num = parameters['pred_traj_num']
    dim = len(ori_data[0][0])
    z_dim = dim

    ori_time_list = np.array(extract_time(ori_data)[0]) - 1

    # 这个地方可能要改一下
    syn_time_list = ori_time_list[:]

    data_for_norm = list()
    for each_traj in ori_data:
        for each_coor in each_traj[:-1]:
            data_for_norm.append(np.array(each_coor))
    stand_scaler = MinMaxScaler()
    stand_scaler.fit_transform(data_for_norm)
    data_norm = stand_scaler.transform(data_for_norm)

    norm_data_index = 0
    for i in range(len(ori_time_list)):
        ori_data[i][:-1] = data_norm[norm_data_index: norm_data_index + int(ori_time_list[i])]
        norm_data_index += int(ori_time_list[i])

    length_data_for_norm = list()
    for each_traj in ori_data:
        length_data_for_norm.append(np.array(each_traj[-1]))
    stand_scaler_length = MinMaxScaler()
    stand_scaler_length.fit_transform(length_data_for_norm)
    length_data_norm = stand_scaler_length.transform(length_data_for_norm)
    for i in range(len(ori_data)):
        ori_data[i][-1] = length_data_norm[i]

    # data input for Predictor
    train_data_for_predictor = list()
    for each_traj in ori_data:
        if np.isnan(each_traj[-1][0]):
            pass
        else:
            train_data_for_predictor.append(each_traj)
    print(train_data_for_predictor[0])

    # Input placeholders
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Length = tf.placeholder(tf.float32, [None, 1], name="goal_length")
    Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")

    def embedder(X, T):
        """Embedding network between original feature space to latent space.

        Args:
          - X: input begin trajectory features
          - T: input time information

        Returns:
          - H: embeddings
        """
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
            H = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense2')
        return H

    def recovery(H, T):
        """recovery network from latent space to original space.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - X_tilde: recovered data
        """
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            de_cell = tf.keras.layers.StackedRNNCells(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])  # module_name = 'gru'
            de_outputs, de_last_states = tf.nn.dynamic_rnn(de_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.layers.dense(de_outputs, dim, activation=tf.nn.sigmoid, name='dense2')
        return X_tilde

    def predictor(H, T):
        """Predict the begin trajectory's length from latent space.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Pred_length: Predicted length results
        """
        with tf.variable_scope("predictor", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            pooled = tf.reduce_mean(e_outputs, axis=1)
            Pred_length = tf.layers.dense(pooled, 1, activation=tf.nn.sigmoid, name='out_dense')
        return Pred_length

    def generator(Z, T):
        """Generator function: Generate begin trajectory data in latent space.

        Args:
          - Z: random variables
          - T: input time information

        Returns:
          - E: generated embedding
        """
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
            E = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense2')
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
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense2')
        return S

    def discriminator(H, T):
        """Discriminate the original and synthetic begin trajectory data.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Y_hat: classification results between original and synthetic begin trajectory
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            # H = Multi_denes(H, 'h')
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
            # output = tf.layers.dense(d_outputs, int(hidden_dim / 4), activation=tf.nn.leaky_relu, name='out_dense1')
            Y_hat = tf.layers.dense(d_outputs, 1, activation=None, name='dis_dense')
        return Y_hat

    # embedder & recovery
    H = embedder(X, T)  # (?, seq_len, hidden_dim)
    X_tilde = recovery(H, T)  # (?, seq_len, ori_dim)

    # Predictor
    pred_length = predictor(H, T)

    # Generator
    E_hat = generator(Z, T)  # （?, seq_len, hidden_dim）
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, T)

    # Discriminator
    Y_real = discriminator(H, T)
    Y_fake = discriminator(H_hat, T)
    Y_fake_e = discriminator(E_hat, T)

    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    pred_vars = [v for v in tf.trainable_variables() if v.name.startswith('predictor')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    gamma = 1

    # Predictor loss
    Pred_loss = tf.losses.mean_squared_error(pred_length, Length)

    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss_adv = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # D_loss_pred = tf.losses.mean_squared_error(pred_length, Length)

    D_loss = D_loss_adv

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    # 2. Supervised loss
    G_loss_S = tf.losses.mean_squared_error(H[:, :, :], H_hat_supervise[:, :, :])

    # 2. Two Moments
    G_loss_V1 = tf.reduce_mean(
        tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2

    # 3. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # embedder network loss
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    # 单独训练
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)
    Pred_solver = tf.train.AdamOptimizer().minimize(Pred_loss, var_list=pred_vars)
    # 联合训练
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)

    ## syngan training
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这里是gpu的序号，指定使用的gpu对象
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()  ##:如果你指定的设备不存在,允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True  ##动态分配内存
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver_en_de = tf.train.Saver(var_list=e_vars + r_vars)
    saver_sup = tf.train.Saver(var_list=g_vars + s_vars + pred_vars)
    saver_joint = tf.train.Saver(var_list=e_vars + r_vars + d_vars + g_vars + s_vars + pred_vars)

    save_model_para_path = out_data_path + "/One_Stage_Model/"
    save_en_de_para_path = save_model_para_path + "Em_recovery_Model/"
    save_sup_pred_para_path = save_model_para_path + "Sup_Model/"
    save_gan_joint_para_path = save_model_para_path + "GAN_joint_Model/"
    pathlib.Path(save_model_para_path).mkdir(parents=True, exist_ok=True)

    losses_only = {'embedder_loss': [], 'sup_loss': [], 'pred_loss': []}
    losses_joint = {'embedder_loss': [], 'g_loss': [], 'd_loss': [], 'pred_loss': []}
    save_path = out_data_path + 'loss_pic/one_stage/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    print('Traing Parameters:')
    print('batch_size = ' + str(batch_size))
    print('hidden_dim = ' + str(hidden_dim))
    print('num_layers = ' + str(num_layers))
    print('module_name = ' + str(module_name))
    print('Start embedder & recovery Network Training')
    print_iter = 1000
    num_epochs = int(iterations / print_iter)

    if os.path.exists(save_en_de_para_path + 'checkpoint'):
        print("加载已有一阶段编解码器模型参数")
        saver_en_de.restore(sess, save_en_de_para_path + 'en_de_model.ckpt')
    else:
        pathlib.Path(save_en_de_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练一阶段编解码器模型...")
        print('Start Encoding Network Training')
        # 1. Embedding network training(embedder & recovery)
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                # Train embedder
                _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: padding(X_mb, max_seq_len), T: T_mb})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(e_loss=np.round(np.sqrt(step_e_loss), 4))
                losses_only['embedder_loss'].append(np.round(np.sqrt(step_e_loss), 4))
        draw_loss_pic(losses_only['embedder_loss'], save_path + 'one_stage_' + 'embedder_only_loss.png')
        save_path1 = os.path.join(save_en_de_para_path, "en_de_model.ckpt")
        os.chmod(save_en_de_para_path, 0o755)  # 修改文件夹权限
        saver_en_de.save(sess, save_path1)
        print('Finish embedder & recovery Network Training')

    if os.path.exists(save_sup_pred_para_path + 'checkpoint'):
        print("加载已有一阶段监督器和预测器模型参数")
        saver_sup.restore(sess, save_sup_pred_para_path + 'sup_model.ckpt')
    else:
        pathlib.Path(save_sup_pred_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练一阶段监督器和预测器模型...")
        # 2. Training with supervised loss and prediction loss
        print('Start Training Supervisor and Predictor Network Only')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Set mini-batch
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)

                X_mb_for_pre, Length_mb_for_pre, T_mb_for_pre = \
                    batch_generator(train_data_for_predictor, ori_time_list, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train generator
                _, step_g_loss_s = sess.run([GS_solver, G_loss_S],
                                            feed_dict={Z: padding(Z_mb, max_seq_len), X: padding(X_mb, max_seq_len),
                                                       T: T_mb})
                _, step_pred_loss = sess.run([Pred_solver, Pred_loss],
                                             feed_dict={X: X_mb_for_pre, Length: Length_mb_for_pre, T: T_mb_for_pre})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(s_loss=np.round(np.sqrt(step_g_loss_s), 4),
                                 pred_loss=np.round(np.sqrt(step_pred_loss), 4))
                losses_only['sup_loss'].append(np.round(np.sqrt(step_g_loss_s), 4))
                losses_only['pred_loss'].append(np.round(np.sqrt(step_pred_loss), 4))
        draw_loss_pic(losses_only['sup_loss'], save_path + 'one_stage_' + 'sup_only_loss.png')
        draw_loss_pic(losses_only['pred_loss'], save_path + 'one_stage_' + 'pred_only_loss.png')
        save_path2 = os.path.join(save_sup_pred_para_path, "sup_model.ckpt")
        os.chmod(save_sup_pred_para_path, 0o755)  # 修改文件夹权限
        saver_sup.save(sess, save_path2)
        print('Finish Training Supervisor and Predictor Network Only')

    if os.path.exists(save_gan_joint_para_path + 'checkpoint'):
        print("加载已有一阶段联合训练模型参数")
        saver_joint.restore(sess, save_gan_joint_para_path + 'joint_model.ckpt')
    else:
        pathlib.Path(save_gan_joint_para_path).mkdir(parents=True, exist_ok=True)
        print("开始训练一阶段联合训练模型...")
        # 3. Joint Training(6models)
        print('Start Joint Training')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Set mini-batch
                for _ in range(2):
                    X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                    # Random vector generation
                    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                    # Train generator
                    _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V],
                                                                              feed_dict={Z: padding(Z_mb, max_seq_len),
                                                                                         X: padding(X_mb, max_seq_len),
                                                                                         T: T_mb})
                    # Train embedder
                    _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: padding(Z_mb, max_seq_len),
                                                                                   X: padding(X_mb, max_seq_len),
                                                                                   T: T_mb})
                # Discriminator training
                # Set mini-batch
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                X_mb_for_pre, Length_mb_for_pre, T_mb_for_pre = \
                    batch_generator(train_data_for_predictor, ori_time_list, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train discriminator
                _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: padding(X_mb, max_seq_len), T: T_mb,
                                                                         Z: padding(Z_mb, max_seq_len)})
                _, step_pred_loss = sess.run([Pred_solver, Pred_loss],
                                             feed_dict={X: X_mb_for_pre, Length: Length_mb_for_pre, T: T_mb_for_pre})

                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(d_loss=np.round(step_d_loss, 4),
                                 g_loss=np.round(step_g_loss_u + step_g_loss_v, 4),
                                 g_loss_s=np.round(np.sqrt(step_g_loss_s), 4),
                                 e_loss_t=np.round(np.sqrt(step_e_loss_t0), 4),
                                 pred_loss=np.round(np.sqrt(step_pred_loss), 4))
                losses_joint['g_loss'].append(np.round(step_g_loss_u + step_g_loss_v, 4))
                losses_joint['d_loss'].append(np.round(step_d_loss, 4))
                losses_joint['embedder_loss'].append(np.round(np.sqrt(step_e_loss_t0), 4))
                losses_joint['pred_loss'].append(np.round(np.sqrt(step_pred_loss), 4))
        draw_loss_pic(losses_joint['embedder_loss'], save_path + 'one_stage_' + 'embedder_joint_loss.png')
        draw_loss_pic(losses_joint['pred_loss'], save_path + 'one_stage_' + 'pred_joint_loss.png')
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(range(len(losses_joint['g_loss'])), losses_joint['g_loss'], color='r', label='Generator')
        plt.plot(range(len(losses_joint['d_loss'])), losses_joint['d_loss'], color='b', label='Discriminator')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss')
        plt.legend()
        plt.savefig(save_path + 'one_stage_' + 'G&D_joint_loss.png', format='png')
        plt.close()
        print('Finish Generator and discriminator Training')
        save_path3 = os.path.join(save_gan_joint_para_path, "joint_model.ckpt")
        os.chmod(save_gan_joint_para_path, 0o755)  # 修改文件夹权限
        saver_joint.save(sess, save_path3)
    print("Start Data Generating")
    ## Synthetic data generation

    pred_traj_num = len(syn_time_list)
    Z_mb = random_generator(pred_traj_num, z_dim, syn_time_list, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: padding(Z_mb, max_seq_len), T: syn_time_list})

    # 分为两种数据：一种是到达终点的轨迹，直接送出模型
    to_end_traj_data = list()
    # 第二种是生成初始段但没到终点了，送入第二个模型
    to_two_stage_traj_data = list()
    to_two_stage_traj_length = list()

    for i in tqdm(range(pred_traj_num)):
        each_gen = generated_data_curr[i, :syn_time_list[i], :]
        each_gen_inverse = stand_scaler.inverse_transform(each_gen)
        if len(each_gen) < max_seq_len:
            to_end_traj_data.append(each_gen_inverse)
        else:
            pred_length_curr = sess.run(pred_length, feed_dict={X: np.array([each_gen]), T: np.array([max_seq_len])})
            pred_length_res = \
            np.squeeze(stand_scaler_length.inverse_transform([np.stack([np.squeeze(pred_length_curr)] * 3, axis=0)]))[0]
            if math.floor(pred_length_res) < max_seq_len + 1:
                to_end_traj_data.append(each_gen_inverse)
            else:
                to_two_stage_traj_data.append(each_gen_inverse)
                to_two_stage_traj_length.append(pred_length_res)

    return to_end_traj_data, to_two_stage_traj_data, to_two_stage_traj_length
