import  numpy as np
import tensorflow as tf

class DQNclass():
    def __init__(self,
                 features,
                 actions,
                 e_gready_max = 0.9,
                 learning_rate = 0.01,
                 reward_greedy = 0.9,
                 memory_size = 20,
                 e_greedy_increment = None,
                 batch_size = 5):
        self.e_greedy_max = e_gready_max
        self.e_greedy_increment = e_greedy_increment
        self.e_greedy = 0 if e_greedy_increment else self.e_gready_max
        self.features = features
        self.actions = actions
        self.memory_counter = 0
        self.memory_size = memory_size
        self.lr = learning_rate
        self.memory = np.zeros(self.memory_size,self.features * 2 + 2)
        self.batch_size = batch_size

    def __build__network__(self):
        self.cnames = ["eval_network",tf.GraphKeys.GLOBAL_VARIABLES]
        h_layers = 10
        w_initailizer = tf.random_normal_initializer(0,0.3)
        b_initializer = tf.constant_initializer(0.1)
        s = tf.placeholder(tf.float32,[None,len(self.features)])
        eval_action = tf.placeholder(tf.float32,[None,len(self.actions)])
        with tf.variable_scope("eval_network"):
            with tf.variable_scope("eval_input_layer"):
                eval_layer1_w = tf.get_variable([len(self.features),h_layers],initializer=w_initailizer,collections=cnames)
                eval_layer1_b = tf.get_variable([h_layers],initializer=b_initializer,collections=cnames)
                self.eval_layer1_output = tf.nn.relu(tf.matmul(s,eval_layer1_b) + eval_layer1_b)
            with tf.variable_scope("eval_layer2"):
                eval_layer2_weights = tf.get_variable([h_layers,len(self.actions)],initializer=w_initailizer,collections=cnames)
                eval_layer2_b = tf.get_variable([len(self.actions)],initializer=b_initializer,collections = cnames)
                self.eval_layer2_output = tf.matmul(self.eval_layer1_output,eval_layer2_weights) + eval_layer2_b

        s_ = tf.placeholder(tf.float32,[None,len(self.features)])
        self.cnames_target = ["target_params",tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope("target_network"):
            with tf.variable_scope("target_net_layer1"):
                target_layer1_weights = tf.get_variable([len(self.features),h_layers],initializer=w_initailizer,collections=cnames_target)
                target_layer1_b = tf.get_variable([h_layers],initializer=b_initializer,collections=cnames_target)
                self.target_layer1_output = tf.nn.relu(tf.matmul(s_,target_layer1_weights) + target_layer1_b)

            with tf.variable_scope("target_layer2"):
                target_layer2_weights = tf.get_variable([h_layers,len(self.actions)],initializer=w_initailizer,collections=cnames_target)
                target_layer2_b = tf.get_variable([len(self.actions)],initializer=b_initializer,collections=cnames_target)
                self.target_layer2_output = tf.matmul(s_,target_layer2_weights) + target_layer2_b


        self.sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    def store_trancision(self,s,a,r,s_):
        if not hasattr(self,"memory_counter"):
            self.memory_counter = 0
        transition = tf.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter % 10 == 0:
            batch_index = np.random.choice(self.memory_size,size = self.batch_size)
            batch_sample_target = self.sess.run(self.target_layer2_output,feed_dict={s_:self.memory[batch_index,-self.features:]})
            batch_sample_target_max = batch_sample_target.max()
            best_eval_batch = batch_sample_eval_max + self.lr(self.memory[batch_index,len(self.features) + 1]  +  self.e_greedy * batch_sample_target_max - self.memory[batch_index,len(self.features)] )
            batch_eval_value = self.sess.run(self.eval_layer2_output,feed_dict={s:self.memory[batch_index,:self.features]})
            batch_eval_value = best_eval_batch[:,self.memory[batch_index,self.features]]
            with tf.variable_scope("loss"):
                cost =tf.reduce_mean( tf.squared_difference(batch_eval_value,best_eval_batch))

            with tf.variable_scope("train"):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
            self.sess.run(train_step)
        if self.memory_counter % 20 == 0:
            self.cnames_target = self.cnames

        if self.e_greedy_increment:
            if (self.e_greedy  + 0.2)< self.e_greedy_max:
                self.e_greedy += 0.2


    def choose_action(self,s):
        if np.random.uniform() < self.e_greedy:
            action = self.sess.run(self.eval_layer2_output,feed_dict={s:s}).argmax()

        else:
            action = np.random.randint(0,self.actions)
        return action


