import numpy as np
import pandas as pd
import tensorflow as tf


# Deep Q Network off-policy
class nature_DQN:
    def __init__(self,num_of_actions,num_of_features,learning_rate=0.1,buffer_size=1000,batch_size=500):
        self.num_of_actions = num_of_actions
        self.num_of_features = num_of_features
        self.learning_rate = learning_rate
                
        # build network based on Tensorflow
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.build_network()
        self.sess.run(self.init)

        self.buffer_size = buffer_size
        self.learning_buffer = np.zeros((self.buffer_size, num_of_features * 2 + 2))
        self.buffer_counter = 0
        self.batch_size = batch_size

        self.apha = 0.85

    def build_network(self):
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, [None, self.num_of_features], name='input')
            self.target_action = tf.placeholder(tf.float32, [None, self.num_of_actions], name='target_action')
            
            n_l1, weight_init, bias_init = 5000, tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            w1 = tf.get_variable('w_1', [self.num_of_features, n_l1], initializer=weight_init)
            b1 = tf.get_variable('b_1', [1, n_l1], initializer=bias_init)
            l1 = tf.nn.relu(tf.matmul(self.input, w1) + b1)

            w2 = tf.get_variable('w_2', [n_l1, self.num_of_actions], initializer=weight_init)
            b2 = tf.get_variable('b_2', [1, self.num_of_actions], initializer=bias_init)
            self.q_eval = tf.matmul(l1, w2) + b2

            self.loss = tf.reduce_mean(tf.squared_difference(self.target_action, self.q_eval))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            self.init = tf.global_variables_initializer()

    def action(self, observation):
        observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.input: observation})
        action = np.argmax(actions_value)
        
        return action

    def learning_buffers(self, observation_pre, action, reward, observation_after):
        if self.buffer_counter >= int(self.buffer_size):
            self.buffer_counter = 0

        each_record = np.hstack((observation_pre, [action, reward], observation_after))
        # example: [ 0.01787666  0.36940159 -0.10496763 -0.68529127 , 0. 0.12284906 , 0.02526469  0.17588144 -0.11867345 -0.42741259]
        self.learning_buffer[self.buffer_counter, :] = each_record
        self.buffer_counter += 1


    def learn(self):
        sample_index = np.random.choice(self.buffer_counter, size=self.batch_size)
        batch_memory = self.learning_buffer[sample_index, :]

        q_target_action = self.sess.run([self.q_eval], feed_dict={self.input: batch_memory[:, :self.num_of_features]})

        reward = batch_memory[:,self.num_of_features+1:self.num_of_features+2]
        
        q_target_action = q_target_action[0]

        j=0
        for i in q_target_action:
            index = np.argmax(i)
            i[index] = reward[j][0] * 1.15 +self.apha * i[index]
            j += 1

        optizize, self.cost = self.sess.run([self.optimizer, self.loss],feed_dict={self.input: batch_memory[:, :self.num_of_features],self.target_action: q_target_action})

        self.learning_buffer = np.zeros((self.buffer_size, self.num_of_features * 2 + 2))


