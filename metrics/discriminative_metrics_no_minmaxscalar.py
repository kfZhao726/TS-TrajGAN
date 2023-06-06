# Necessary Packages
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator, MinMaxScaler
# from sklearn.preprocessing import MinMaxScaler


def discriminative_score_metrics2(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Initialization on the Graph
    tf.reset_default_graph()
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    # no = np.asarray(ori_data).shape[0]
    # # seq_len is not sure
    # dim = 3

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    ## Builde a post-hoc RNN discriminator network
    # Network parameters
    # hidden_dim = int(dim/2)
    hidden_dim = int(dim / 2) + 1 # 2
    # hidden_dim = 12
    iterations = 2000
    batch_size = 128

    # Input place holders
    # Feature
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x_hat")

    T = tf.placeholder(tf.int32, [None], name="myinput_t")
    T_hat = tf.placeholder(tf.int32, [None], name="myinput_t_hat")

    # discriminator function
    def discriminator(x, t):
        """Simple discriminator function.

        Args:
          - x: time-series data
          - t: time information

        Returns:
          - y_hat_logit: logits of the discriminator output
          - y_hat: discriminator output
          - d_vars: discriminator variables
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as vs:
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='d_cell')
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
            y_hat_logit = tf.layers.dense(d_last_states, 1, activation=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

        return y_hat_logit, y_hat, d_vars

    y_logit_real, y_pred_real, d_vars = discriminator(X, T)
    y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)

    # Loss for the discriminator
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_real,
                                                                         labels=tf.ones_like(y_logit_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_fake,
                                                                         labels=tf.zeros_like(y_logit_fake)))
    d_loss = d_loss_real + d_loss_fake

    # optimizer
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)

    ## Train the discriminator
    # Start session and initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()  ##:如果你指定的设备不存在,允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True  ##动态分配内存
    sess = tf.Session(config=config)
    # print(device_lib.list_local_devices())
    # print("*"*50)
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training step
    for _ in tqdm(range(iterations), miniters=1):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        # Train discriminator
        _, step_d_loss = sess.run([d_solver, d_loss],
                                  feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})

    ## Test the performance on the testing set
    y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake],
                                                  feed_dict={X: test_x, T: test_t, X_hat: test_x_hat,
                                                             T_hat: test_t_hat})

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score