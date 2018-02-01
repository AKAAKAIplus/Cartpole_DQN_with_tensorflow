import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class double_network_DQN:
    def __init__(self, num_of_actions, num_of_features, learning_rate=0.01, alpha=0.9,batch_size=5, learning_buffer_size=200,replace_network_number=30):

        self.num_of_actions = num_of_actions
        self.num_of_features = num_of_features
        self.learning_rate = learning_rate
        self.alpha = alpha
        

        self.learning_counter = 0
        self.buffer_counter = 0

        self.learning_buffer_size = learning_buffer_size
        self.learning_buffer = np.zeros((self.learning_buffer_size, num_of_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.build_network()
        self.sess = tf.Session()
        self.sess.run(self.init)


        self.replace_network_number = replace_network_number        
        self.batch_size = batch_size
        
        

    def build_network(self):
        self.evaluate_network_input = tf.placeholder(tf.float32, [None, self.num_of_features])
        self.q_value = tf.placeholder(tf.float32, [None, self.num_of_actions], name='Q_target')
            
        self.target_network_input = tf.placeholder(tf.float32, [None, self.num_of_features])    # input
        n_l1 = 10

        with tf.variable_scope('eval_net'):
            w_initializer_eval, b_initializer_eval = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            evaluate_network_parameter= ['evaluate_network_parameter', tf.GraphKeys.GLOBAL_VARIABLES]

            w1 = tf.get_variable('w1', [self.num_of_features, n_l1], initializer=w_initializer_eval, collections=evaluate_network_parameter)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer_eval, collections=evaluate_network_parameter)
            l1 = tf.nn.relu(tf.matmul(self.evaluate_network_input, w1) + b1)

            w2 = tf.get_variable('w2', [n_l1, self.num_of_actions], initializer=w_initializer_eval, collections=evaluate_network_parameter)
            b2 = tf.get_variable('b2', [1, self.num_of_actions], initializer=b_initializer_eval, collections=evaluate_network_parameter)
            self.q_value_evaluate_network = tf.matmul(l1, w2) + b2

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_value, self.q_value_evaluate_network))
        self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('target_net'):
            w_initializer_target, b_initializer_eval = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            target_network_parameter = ['target_network_parameter', tf.GraphKeys.GLOBAL_VARIABLES]

            w1 = tf.get_variable('w1', [self.num_of_features, n_l1], initializer=w_initializer_target, collections=target_network_parameter)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer_eval, collections=target_network_parameter)
            l1 = tf.nn.relu(tf.matmul(self.target_network_input, w1) + b1)

            w2 = tf.get_variable('w2', [n_l1, self.num_of_actions], initializer=w_initializer_target, collections=target_network_parameter)
            b2 = tf.get_variable('b2', [1, self.num_of_actions], initializer=b_initializer_eval, collections=target_network_parameter)
            self.q_next = tf.matmul(l1, w2) + b2
        
        self.init = tf.global_variables_initializer()

    def learning_buffers(self, s, a, r, s_):
        # transition = np.hstack((s, [a, r], s_))
        # # example: [ 0.01787666  0.36940159 -0.10496763 -0.68529127 , 0. 0.12284906 , 0.02526469  0.17588144 -0.11867345 -0.42741259]
        # index = self.buffer_counter % self.learning_buffer_size
        # self.learning_buffer[index, :] = transition
        # self.buffer_counter += 1

    def action(self, observation):
        # observation = observation[np.newaxis, :]
        # actions_value = self.sess.run(self.q_value_evaluate_network, feed_dict={self.evaluate_network_input: observation})
        # action = np.argmax(actions_value)
        # return action

    def learn(self):
        

