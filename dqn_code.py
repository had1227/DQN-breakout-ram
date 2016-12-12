'''
A DQN agent solving atari breakout game
using tensorflow and openai gym environment 'Breakout-RAM-v0'

https://github.com/had1227/DQN-breakout-ram/blob/master/dqn_code.py

Seoul national university. 2017.
timothy ha & inyoung bang
'''

import gym
import tensorflow as tf
import random
import numpy as np
import time
from argparse import ArgumentParser

random.seed(1)

try:
  import scipy.misc
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave
except:
  import cv2
  imresize = cv2.resize
  imwrite = cv2.imwrite


OUT_DIR = 'breakout-ram-experiment'  # default saving directory
MAX_SCORE_QUEUE_SIZE = 100  # number of episode scores to calculate average performance
GAME = 'Breakout-ram-v0'  # name of game
TIMESTEP_LIMIT = 10000  # Time step limit of each episode
USER_OBSV_DIM = [4,84,84]  # observation dimension after preprocessing. [history_size,height,width]
actionset = [0,0,0,0,0,0]

test_num = 0

'''
The DQN model itself.
Remain unchanged when applied to different problems.
'''

device_type = "/gpu:2"
with tf.device(device_type):
    class auto_options():
        def __init__(self, env):
    
            self.render = False
            self.env_type = env.env_type
            self.load_model=False
    
            self.MAX_EPISODE = 5000000
    
            self.OBSERVATION_DIM = env.obsv_dim  # for Atari, a list of length 3, (N)CHW
    
            self.history_length = self.OBSERVATION_DIM[0]
    
            self.ACTION_DIM = env.act_dim
    
            self.BATCH_SIZE = 512
            self.MAX_EXPERIENCE = 5000000  # size of experience memory
            self.target_net_update_freq = 10000
    
            self.GAMMA = 0.95  # discount factor
    
            self.action_repeat = 1
            self.update_freq = 4
    
            # learning parameter
            self.LR = 0.00001  # learning rate
            self.GM = 0.95  # gradient momentum
            self.SGM = 0.95  # squared gradient momentum
            self.Optimizer = tf.train.AdamOptimizer(self.LR)
    
            # epsilon greedy
            self.LINEAR_EPS_DECAY = True
            self.INIT_EPS = 1.0
            self.FINAL_EPS = 0.2
            self.EPS_DECAY_RATE = 0.1
            self.EPS_ANNEAL_STEPS = 10000000
    
            self.replay_start_size = 100000
    
            self.no_op_max = 30
    
            self.test_freq = 100
    
            # For ClassicControl DQN
            self.layer_size = 5
            
            self.H1_SIZE = 4096
            self.H2_SIZE = 4096
            self.H3_SIZE = 4096
            self.H4_SIZE = 4096
            self.H5_SIZE = 4096
    
            # For Atari DQN
            self.padding = 'SAME'
    
            self.c1_size = [5,5,4,32]
            self.stride1_size = [1,1,1,1]
    
            self.c2_size = [5,5,32,32]
            self.stride2_size = [1,1,1,1]
    
            self.c3_size = [4,4,32,64]
            self.stride3_size = [1,1,1,1]
    
            self.c4_size = [3,3,64,64]
            self.stride4_size = [1,1,1,1]
    
            self.f1_size = [84*84*64,512]
            self.f2_size = [512, self.ACTION_DIM]
    
    
    class gym_ClassicControl_Environment():
        def __init__(self, env_name):
            self.env = gym.make(env_name)
            self.env_type = 'ClassicControl'
            self.monitor = self.env.monitor
    
            self.obsv_dim = list(self.env.observation_space.shape)
    
            self.action_space = self.env.action_space
            self.act_dim = self.env.action_space.n
    
            self.env.spec.timestep_limit = TIMESTEP_LIMIT
    
            print("{} has {} action").format(env_name, self.act_dim)
            print("env_type={}, obsv_dim={}, act_dim={}".format(self.env_type, self.obsv_dim, self.act_dim))
            print ""
    
        def step(self, action):
            return self.env.step(action)
        def render(self, close_var=False):
            self.env.render(close=close_var)
        def reset(self):
            return self.env.reset()
        def no_op_reset(self, no_op_max):
            return self.reset()
        def step_repeat(self, action, act_repeat=4):
            return self.step(action)
        
    class gym_AtariRAM_Environment():
        def __init__(self, env_name, fast_restart=True, display=False):
            self.env = gym.make(env_name)
            self.env_type = 'AtariRAM'
            self.monitor = self.env.monitor
    
            self.obsv_dim = list(self.env.observation_space.shape)
    
            self.action_space = self.env.action_space
            self.act_dim = self.env.action_space.n
            
            self.display = display # gym environment render option
            self.fast_restart = fast_restart
    
            print("{} has {} action").format(env_name, self.act_dim)
            print("env_type={}, obsv_dim={}, act_dim={}".format(self.env_type, self.obsv_dim, self.act_dim))
            print ""
    
        def step(self, action):
            reward = 0.0
    
            if (self.env_type != 'AtariRAM'):
                observation, reward, done, _ = self.env.step(action)
                return observation, reward, done, {}  # for classic control environment
    
            observation, reward, done, _ = self.env.step(action)
    
            if(self.fast_restart):
                done = (self.lives > self.env.ale.lives()) # terminate when it died first time
                if done:
                    reward = -1
            else:
                done = self.env.ale.game_over() # terminate when it's remainning lives == 0

                if not done and (self.lives > self.env.ale.lives()):
                    self.env.step(1) # start new life
                    self.lives -= 1
                    reward = -1
                
                if done:
                    reward = -1
    
            return observation, reward, done, {}
    
        def render(self, close_var=False):
            self.env.render(close=close_var)
        def reset(self, fast_restart_setting=None):
    
            if fast_restart_setting is None:
                pass
            else:
                self.fast_restart = fast_restart_setting
            obs = self.env.reset()
            self.lives=self.env.ale.lives()
    
            return obs
        def no_op_reset(self, no_op_max, fast_restart_setting=None):
            obsv = self.reset(fast_restart_setting)
            done = False
    
            obsv, reward, done, _ = self.step(1) # start game
    
            if(done):
                print "It failed to start. Please re-start or check error"
                return None
            else:
                return obsv
        def step_repeat(self, action, act_repeat=4):
            return self.step(action)
        
    class gym_Atari_Environment():
        def __init__(self, env_name, fast_restart=False, display=False):
            self.env = gym.make(env_name)
            self.env_type = 'Atari'
            self.monitor = self.env.monitor
    
            self.user_obsv_dim = USER_OBSV_DIM  # [4,84,84] is recommended
    
            self.hidden_obsv_dim = list(self.env.observation_space.shape)
    
            if self.user_obsv_dim is None:  # user_obsv_dim is a list type variable. User defiend observation dimension
                self.user_obsv_dim = None
                self.obsv_dim = list(self.env.observation_space.shape)
                self.history_size = 1
            else:
                self.user_obsv_dim = USER_OBSV_DIM  # user defined dimension
                self.obsv_dim = USER_OBSV_DIM  # if env_type is 'Atari', user_obsv_dim must be length 3 listr_obsv_dim
                self.history_size = USER_OBSV_DIM[0]  # (N)CHW form is used
    
            self.action_space = self.env.action_space
            self.act_dim = self.env.action_space.n
    
            self.display = display # gym environment render option
            self.fast_restart = fast_restart
            # if fast_restart is True, training will start new game when agent lose just 1 life.
    
            if hasattr(self.env, 'get_action_meanings'):
                meanings = self.env.get_action_meanings()
                print("{} has {} action : {}").format(env_name,self.act_dim,meanings)
    
            print("obsv_dim={}, act_dim={}, fast_restart={}".format(self.obsv_dim, self.act_dim, self.fast_restart))
            print("")
    
        def preprocess(self, observation, pre_observation=None):
            # Use max value btwn pre_obsv, obsv to remove flickering
    
            if pre_observation is None:
                obsv = observation
            else:
                obsv = observation
                #obsv = np.fmax(pre_observation, observation) # use max value for each element
    
            # Use Y channel (weight values from BT.601(for SDTV))
            obsv_y = 0.299 * obsv[:,:,0] + 0.587 * obsv[:,:,1] + 0.114 * obsv[:,:,2]
            obsv_y = obsv_y.astype(np.uint8)
    
    
            if self.user_obsv_dim is None: # if there is no user-defined obsv_dim
                pass
            else:
                obsv_y = imresize(obsv_y, self.user_obsv_dim[1:3]) # image resize
    
            return obsv_y
    
        def step(self, action):
            reward = 0.0
    
            if (self.env_type != 'Atari'):
                observation, reward, done, _ = self.env.step(action)
                return observation, reward, done, {}  # for classic control environment
    
            observation, reward, done, _ = self.env.step(action)
            obsv_y = self.preprocess(observation)
    
            if(self.fast_restart):
                done = (self.lives > self.env.ale.lives()) # terminate when it died first time
            else:
                done = self.env.ale.game_over() # terminate when it's remainning lives == 0

                if not done and (self.lives > self.env.ale.lives()):
                    self.env.step(1) # start new life
                    self.lives -= 1
    
            return obsv_y, reward, done, {}
    
        def step_by_one(self, action):
    
            reward = self.env.ale.act(action) # return value is a reward
            ob = self.env._get_obs() # return value is an observation
            obsv_y = self.preprocess(ob)
            done = False
    
            if(self.fast_restart):
                done = (self.lives > self.env.ale.lives()) # terminate when it died first time
            else:
                done = self.env.ale.game_over() # terminate when it's remainning lives == 0

                if not done and (self.lives > self.env.ale.lives()):
                    self.env.step(1) # start new life
                    self.lives -= 1
    
            return obsv_y, reward, done, {}
    
        def step_repeat(self, action, act_repeat=4):
    
            result_obsv = []
            reward = 0.0
            pre_ob = self.env._get_obs()
            done = False
    
            for i in xrange(act_repeat):  # (act_repeat) becomes history size
                reward += self.env.ale.act(action)
                ob = self.env._get_obs()
                obsv_y = self.preprocess(ob, pre_ob)
    
                if(self.fast_restart):
                    done = (self.lives > self.env.ale.lives()) # terminate wheuser_obsv_dimn it died first time
                else:
                    done = self.env.ale.game_over() # terminate when it's remainning lives == 0
                    if (self.lives > self.env.ale.lives()):
                        self.env.step(1) # start new life
                        self.lives -= 1
    
                if done:
                    obsv_y = np.zeros_like(obsv_y)
                result_obsv.append(obsv_y)
    
            result_obsv = np.asarray(result_obsv)
    
            return result_obsv, reward, done, {}
    
    
        def step_repeat_for_test(self, action, epi_step, act_repeat=4):
    
            global test_num
    
            result_obsv = []
            reward = 0.0
            pre_ob = self.env._get_obs()
            done = False
    
            for i in xrange(act_repeat):  # (act_repeat) becomes history size
                reward += self.env.ale.act(action)
                ob = self.env._get_obs()
                obsv_y = self.preprocess(ob, pre_ob)
    
                if(self.fast_restart):
                    done = (self.lives > self.env.ale.lives()) # terminate wheuser_obsv_dimn it died first time
                else:
                    done = self.env.ale.game_over() # terminate when it's remainning lives == 0
                    if (self.lives > self.env.ale.lives()):
                        self.env.step(1) # start new life
                        self.lives -= 1
    
                if done:
                    obsv_y = np.zeros_like(obsv_y)
                result_obsv.append(obsv_y)
    
                dir_name = 'test' + str(test_num) + 'obsv' + str(epi_step) + 'num' + str(i) + '.png'
                scipy.misc.toimage(obsv_y, cmin=0.0).save(dir_name)
    
            result_obsv = np.asarray(result_obsv)
    
            return result_obsv, reward, done, {}
    
        def reset(self, fast_restart_setting=None):
    
            if fast_restart_setting is None:
                pass
            else:
                self.fast_restart = fast_restart_setting
            obs = self.env.reset()
            self.lives=self.env.ale.lives()
            obsv_y = self.preprocess(obs)
    
            return obsv_y
    
        def no_op_reset(self, no_op_max, fast_restart_setting=None):
            obsv = self.reset(fast_restart_setting)
            done = False
    
            for i in xrange(no_op_max/4):
                obsv, reward, done, _ = self.step_repeat(0) # 'not doing' action
    
            obsv, reward, done, _ = self.step_repeat(1) # start game
    
            if(done):
                print "It failed to start. Please re-start or check error"
                return None
            else:
                return obsv
    
        def no_op_reset_for_test(self, no_op_max, fast_restart_setting=None):
            obsv = self.reset(fast_restart_setting)
            done = False
    
            for i in xrange(no_op_max/4):
                obsv, reward, done, _ = self.step_repeat(0) # 'not doing' action
    
            obsv, reward, done, _ = self.step_repeat_for_test(1,0,4) # start game
    
            if(done):
                print "It failed to start. Please re-start or check error"
                return None
            else:
                return obsv
    
        def render(self, close_=False):
            self.env.render(close=close_)
    
    
    class Atari_QAgent:
        def __init__(self, options):
    
            self.options = options
    
            # filter_size : [filter_height, filter_width, in_channels, out_channels]
    
            # Convolutional layers
            self.c1_filter = self.weight_variable(options.c1_size)
            self.c2_filter = self.weight_variable(options.c2_size)
            self.c3_filter = self.weight_variable(options.c3_size)
            self.c4_filter = self.weight_variable(options.c4_size)
    
            self.c1_bias = self.bias_variable([options.c1_size[3]])
            self.c2_bias = self.bias_variable([options.c2_size[3]])
            self.c3_bias = self.bias_variable([options.c3_size[3]])
            self.c4_bias = self.bias_variable([options.c4_size[3]])
    
            # Fully-connected layers
            self.f1_w = self.weight_variable(options.f1_size)
            self.f1_b = self.bias_variable([options.f1_size[1]])
            self.f2_w = self.weight_variable(options.f2_size)
            self.f2_b = self.bias_variable([options.f2_size[1]])
    
        def conv_layer(self, _input, _filter, _stride, _bias, activation_func=tf.nn.relu):
            _input_dim = np.shape(_input)  # NCHW form is used
    
            # Convolution
            _conv1 = tf.nn.conv2d(_input, _filter, _stride, self.options.padding, data_format="NCHW")
            # Add Bias
            _conv2 = tf.nn.bias_add(_conv1,_bias, data_format="NCHW")
            # Pass Activation Function
            _conv3 = activation_func(_conv2)
    
            return _conv3
    
        def max_pooling(self, _input, k_size, stride_size=1):
    
            ksize = [1, 1, k_size, k_size]
            stride = [1, 1, stride_size, stride_size]
            # Max-pooling
            _pool = tf.nn.max_pool(_input, ksize, stride, self.options.padding, data_format="NCHW")
    
            return _pool
    
        # Add options to graph
        def add_value_net(self, options):
    
            observation = tf.placeholder(tf.float32, [None] + options.OBSERVATION_DIM)
    
            # convolutional layers
            l1 = self.conv_layer(observation, self.c1_filter, options.stride1_size, self.c1_bias)
            l1_ = self.max_pooling(l1, 2)
            l2 = self.conv_layer(l1_, self.c2_filter, options.stride2_size, self.c2_bias)
            l2_ = self.max_pooling(l2, 2)
            l3 = self.conv_layer(l2_, self.c3_filter, options.stride3_size, self.c3_bias)
            l3_ = self.max_pooling(l3, 2)
            l4 = self.conv_layer(l3_, self.c4_filter, options.stride4_size, self.c4_bias)
    
            print l1
            print l2
            print l3
            print l4
    
            # vectorize
            _dense = tf.reshape(l4, [-1, self.f1_w.get_shape().as_list()[0]])
    
            print _dense
            # fully-connected layers
            l5  = tf.add(tf.matmul(_dense, self.f1_w), self.f1_b)
            Q = tf.add(tf.matmul(l5, self.f2_w), self.f2_b)
    
            print l5
            print Q
            print ""
    
            return observation, Q
    
        def create_additional_net(self, options):
    
            # filter_size : [filter_height, filter_width, in_channels, out_channels]
    
            # Convolutional layers
            self.new_c1_filter = self.weight_variable(options.c1_size)
            self.new_c2_filter = self.weight_variable(options.c2_size)
            self.new_c3_filter = self.weight_variable(options.c3_size)
            self.new_c4_filter = self.weight_variable(options.c4_size)
    
            self.new_c1_bias = self.bias_variable([options.c1_size[3]])
            self.new_c2_bias = self.bias_variable([options.c2_size[3]])
            self.new_c3_bias = self.bias_variable([options.c3_size[3]])
            self.new_c4_bias = self.bias_variable([options.c4_size[3]])
    
            # Fully-connected layers
            self.new_f1_w = self.weight_variable(options.f1_size)
            self.new_f1_b = self.bias_variable([options.f1_size[1]])
            self.new_f2_w = self.weight_variable(options.f2_size)
            self.new_f2_b = self.bias_variable([options.f2_size[1]])
    
            # Construct additional value network
            observation = tf.placeholder(tf.float32, [None] + options.OBSERVATION_DIM)
    
            # convolutional layers
            l1 = self.conv_layer(observation, self.new_c1_filter, options.stride1_size, self.new_c1_bias)
            l1_ = self.max_pooling(l1, 2)
            l2 = self.conv_layer(l1_, self.new_c2_filter, options.stride2_size, self.new_c2_bias)
            l2_ = self.max_pooling(l2, 2)
            l3 = self.conv_layer(l2_, self.new_c3_filter, options.stride3_size, self.new_c3_bias)
            l3_ = self.max_pooling(l3, 2)
            l4 = self.conv_layer(l3_, self.new_c4_filter, options.stride4_size, self.new_c4_bias)
    
            # vectorize
            _dense = tf.reshape(l4, [-1, self.new_f1_w.get_shape().as_list()[0]])
    
            # fully-connected layers
            l5  = tf.add(tf.matmul(_dense, self.new_f1_w), self.new_f1_b)
            Q = tf.add(tf.matmul(l5, self.new_f2_w), self.new_f2_b)
    
            return observation, Q
    
    
        def run_copy_op(self, sess, var, new_var):
    
            copy_op = new_var.assign(var)
            sess.run(copy_op)
    
    
        def copy_network(self, sess):
    
            self.run_copy_op(sess, self.c1_filter, self.new_c1_filter)
            self.run_copy_op(sess, self.c2_filter, self.new_c2_filter)
            self.run_copy_op(sess, self.c3_filter, self.new_c3_filter)
            self.run_copy_op(sess, self.c4_filter, self.new_c4_filter)
    
            self.run_copy_op(sess, self.c1_bias, self.new_c1_bias)
            self.run_copy_op(sess, self.c2_bias, self.new_c2_bias)
            self.run_copy_op(sess, self.c3_bias, self.new_c3_bias)
            self.run_copy_op(sess, self.c4_bias, self.new_c4_bias)
    
            self.run_copy_op(sess, self.f1_w, self.new_f1_w)
            self.run_copy_op(sess, self.f1_b, self.new_f1_b)
            self.run_copy_op(sess, self.f2_w, self.new_f2_w)
            self.run_copy_op(sess, self.f2_b, self.new_f2_b)
    
    
        # Sample action with random rate eps
        def sample_action(self, Q, feed, eps, options):
            if random.random() <= eps:
                action_index = env.action_space.sample()
            else:
                global actionset
                act_values = Q.eval(feed_dict=feed)
                action_index = np.argmax(act_values)
                actionset[action_index] += 1
            action = np.zeros(options.ACTION_DIM)
            action[action_index] = 1
    
            return action
    
        # Weights initializer
        def xavier_initializer(self, shape):
            dim_sum = np.sum(shape)
            if len(shape) == 1:
                dim_sum += 1
            bound = np.sqrt(6.0 / dim_sum)
            return tf.random_uniform(shape, minval=-bound, maxval=bound)
    
        # Tool function to create weight variables
        def weight_variable(self, shape):
            return tf.Variable(self.xavier_initializer(shape))
    
        # Tool function to create bias variables
        def bias_variable(self, shape):
            return tf.Variable(self.xavier_initializer(shape))
    
    class QAgent:
        # A naive neural network with 3 hidden layers and relu as non-linear function.
        def __init__(self, options):
            if options.layer_size == 4:                
                self.W1 = self.weight_variable(options.OBSERVATION_DIM + [options.H1_SIZE])
                self.b1 = self.bias_variable([options.H1_SIZE])
                self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
                self.b2 = self.bias_variable([options.H2_SIZE])
                self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
                self.b3 = self.bias_variable([options.H3_SIZE])
                self.W4 = self.weight_variable([options.H3_SIZE, options.H4_SIZE])
                self.b4 = self.bias_variable([options.H4_SIZE])
                self.W5 = self.weight_variable([options.H4_SIZE, options.ACTION_DIM])
                self.b5 = self.bias_variable([options.ACTION_DIM])
            else:
                self.W1 = self.weight_variable(options.OBSERVATION_DIM + [options.H1_SIZE])
                self.b1 = self.bias_variable([options.H1_SIZE])
                self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
                self.b2 = self.bias_variable([options.H2_SIZE])
                self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
                self.b3 = self.bias_variable([options.H3_SIZE])
                self.W4 = self.weight_variable([options.H3_SIZE, options.H4_SIZE])
                self.b4 = self.bias_variable([options.H4_SIZE])
                self.W5 = self.weight_variable([options.H4_SIZE, options.H5_SIZE])
                self.b5 = self.bias_variable([options.H5_SIZE])
                self.W6 = self.weight_variable([options.H5_SIZE, options.ACTION_DIM])
                self.b6 = self.bias_variable([options.ACTION_DIM])
                
        # Weights initializer
        def xavier_initializer(self, shape):
            dim_sum = np.sum(shape)
            if len(shape) == 1:
                dim_sum += 1
            bound = np.sqrt(6.0 / dim_sum)
            return tf.random_uniform(shape, minval=-bound, maxval=bound)
    
        # Tool function to create weight variables
        def weight_variable(self, shape):
            return tf.Variable(self.xavier_initializer(shape))
    
        # Tool function to create bias variables
        def bias_variable(self, shape):
            return tf.Variable(self.xavier_initializer(shape))
    
        # Add options to graph
        def add_value_net(self, options):
            if options.layer_size == 4:
                observation = tf.placeholder(tf.float32, [None] + options.OBSERVATION_DIM)
                h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
                h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
                h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
                h4 = tf.nn.relu(tf.matmul(h3, self.W4) + self.b4)
                Q = tf.squeeze(tf.matmul(h4, self.W5) + self.b5)
            else:
                observation = tf.placeholder(tf.float32, [None] + options.OBSERVATION_DIM)
                h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
                h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
                h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
                h4 = tf.nn.relu(tf.matmul(h3, self.W4) + self.b4)
                h5 = tf.nn.relu(tf.matmul(h4, self.W5) + self.b5)
                Q = tf.squeeze(tf.matmul(h5, self.W6) + self.b6)
            
            return observation, Q
        
        def create_additional_net(self, options):
            if options.layer_size == 4:                
                self.new_W1 = self.weight_variable(options.OBSERVATION_DIM + [options.H1_SIZE])
                self.new_b1 = self.bias_variable([options.H1_SIZE])
                self.new_W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
                self.new_b2 = self.bias_variable([options.H2_SIZE])
                self.new_W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
                self.new_b3 = self.bias_variable([options.H3_SIZE])
                self.new_W4 = self.weight_variable([options.H3_SIZE, options.H4_SIZE])
                self.new_b4 = self.bias_variable([options.H4_SIZE])
                self.new_W5 = self.weight_variable([options.H4_SIZE, options.ACTION_DIM])
                self.new_b5 = self.bias_variable([options.ACTION_DIM])
            else:
                self.new_W1 = self.weight_variable(options.OBSERVATION_DIM + [options.H1_SIZE])
                self.new_b1 = self.bias_variable([options.H1_SIZE])
                self.new_W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
                self.new_b2 = self.bias_variable([options.H2_SIZE])
                self.new_W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
                self.new_b3 = self.bias_variable([options.H3_SIZE])
                self.new_W4 = self.weight_variable([options.H3_SIZE, options.H4_SIZE])
                self.new_b4 = self.bias_variable([options.H4_SIZE])
                self.new_W5 = self.weight_variable([options.H4_SIZE, options.H5_SIZE])
                self.new_b5 = self.bias_variable([options.H5_SIZE])
                self.new_W6 = self.weight_variable([options.H5_SIZE, options.ACTION_DIM])
                self.new_b6 = self.bias_variable([options.ACTION_DIM])
                
            observation = tf.placeholder(tf.float32, [None] + options.OBSERVATION_DIM)
            h1 = tf.nn.relu(tf.matmul(observation, self.new_W1) + self.new_b1)
            h2 = tf.nn.relu(tf.matmul(h1, self.new_W2) + self.new_b2)
            h3 = tf.nn.relu(tf.matmul(h2, self.new_W3) + self.new_b3)
            h4 = tf.nn.relu(tf.matmul(h3, self.new_W4) + self.new_b4)
            Q = tf.squeeze(tf.matmul(h4, self.new_W5) + self.new_b5)
    
            return observation, Q
    
        def run_copy_op(self, sess, var, new_var):
    
            copy_op = new_var.assign(var)
            sess.run(copy_op)
    
        def copy_network(self, sess):
    
            self.run_copy_op(sess, self.W1, self.new_W1)
            self.run_copy_op(sess, self.W2, self.new_W2)
            self.run_copy_op(sess, self.W3, self.new_W3)
            self.run_copy_op(sess, self.W4, self.new_W4)
            self.run_copy_op(sess, self.W5, self.new_W5)
            
            self.run_copy_op(sess, self.b1, self.new_b1)
            self.run_copy_op(sess, self.b2, self.new_b2)
            self.run_copy_op(sess, self.b3, self.new_b3)
            self.run_copy_op(sess, self.b4, self.new_b4)
            self.run_copy_op(sess, self.b5, self.new_b5)
            
        # Sample action with random rate eps
        def sample_action(self, Q, feed, eps, options):
            if random.random() <= eps:
                action_index = env.action_space.sample()
            else:
                act_values = Q.eval(feed_dict=feed)
                action_index = np.argmax(act_values)
            action = np.zeros(options.ACTION_DIM)
            action[action_index] = 1
            return action
        
    
    
    def test(env, agent, Q1, obs, episode_num):
        options = auto_options(env)
    
        global test_num
        global actionset
    
        #observation = env.no_op_reset_for_test(options.no_op_max)          # with options saving observation image
        observation = env.no_op_reset(options.no_op_max,fast_restart_setting=True)
        done = False
        score = 0
        epi_step = 0
    
        while not done:
            epi_step += 1
    
            if options.render:
                env.render()
    
            if env.env_type == 'Atari':
                action = agent.sample_action(Q1, {obs: np.reshape(observation, tuple([-1] + options.OBSERVATION_DIM))}, 0.01, options)
            else:
                action = agent.sample_action(Q1, {obs: np.reshape(observation, (1, -1))}, 0.01, options)
    
            #observation, reward, done, _ = env.step_repeat_for_test(np.argmax(action), epi_step, options.action_repeat)
            observation, reward, done, _ = env.step_repeat(np.argmax(action), options.action_repeat)
            
            if reward >0:
                score += reward
    
            #print action
        
            if epi_step > TIMESTEP_LIMIT:
                done = True
        
        env.fast_restart = True
            
        return score, epi_step
    
    def new_report_file(options):
        filename = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        filename = "report-" + GAME + "-" + filename + ".txt"
    
        with open(filename, "w") as f:
            f.write(options.env_type + " agent report file: " + filename + "\n\n")
            f.write("gym env:" + GAME + "\n")
            f.write("observation dimension: " + str(options.OBSERVATION_DIM) + "\n")
            f.write("batch size: " + str(options.BATCH_SIZE) + "  ")
            f.write("MAX_EXPERIENCE: " + str(options.MAX_EXPERIENCE) + "  ")
            f.write("Net_update_freq: " + str(options.target_net_update_freq) + "\n")
            f.write("Gamma: " + str(options.GAMMA) + "  ")
            f.write("LR: " + str(options.LR) + "  ")
            f.write("Eps anneal_steps: " + str(options.EPS_ANNEAL_STEPS) + "  ")
            f.write("Optimizer: AdamOptimizer" + "\n\n")
                
            f.write("Network size: \n")
            
            if (options.env_type == 'Atari'):
                f.write("conv1: " + str(options.c1_size) + "  ")
                f.write("stride1: " + str(options.stride1_size) + "\n")
                f.write("conv2: " + str(options.c2_size) + "  ")
                f.write("stride2: " + str(options.stride2_size) + "\n")
                f.write("conv3: " + str(options.c3_size) + "  ")
                f.write("stride3: " + str(options.stride3_size) + "\n")
                f.write("conv4: " + str(options.c4_size) + "  ")
                f.write("stride4: " + str(options.stride4_size) + "\n")
                f.write("fc1: " + str(options.f1_size) + "\n")
                f.write("fc2: " + str(options.f2_size) + "\n\n")
            else:
                if options.layer_size == 4:
                    f.write("h1: " + str(options.H1_SIZE) + "\n")
                    f.write("h2: " + str(options.H2_SIZE) + "\n")
                    f.write("h3: " + str(options.H3_SIZE) + "\n")
                    f.write("h4: " + str(options.H4_SIZE) + "\n\n")
                else:
                    f.write("h1: " + str(options.H1_SIZE) + "\n")
                    f.write("h2: " + str(options.H2_SIZE) + "\n")
                    f.write("h3: " + str(options.H3_SIZE) + "\n")
                    f.write("h4: " + str(options.H4_SIZE) + "\n")
                    f.write("h5: " + str(options.H5_SIZE) + "\n\n")
                
        return filename
    
    def train(env):
        # Define placeholders to catch inputs and add options
        options = auto_options(env)
        
        filename = new_report_file(options)
        
        if options.env_type == 'ClassicControl':
            agent = QAgent(options)
        elif options.env_type == 'Atari':
            agent = Atari_QAgent(options)
        else:
            agent = QAgent(options)
    
        sess = tf.InteractiveSession()
        
        # placeholders
        obs, Q1 = agent.add_value_net(options)
        act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
        rwd = tf.placeholder(tf.float32, [None, ])
        is_done = tf.placeholder(tf.float32, [None,])
    
        if options.env_type == 'ClassicControl':
            next_obs, Q2 = agent.add_value_net(options)
        elif options.env_type == 'Atari':
            next_obs, Q2 = agent.create_additional_net(options)  # network of target action-value function Q-
        else:
            next_obs, Q2 = agent.add_value_net(options)
            #next_obs, Q2 = agent.create_additional_net(options)  # network of target action-value function Q-
    
        # loss function and optimizer
        values1 = tf.reduce_sum(tf.mul(Q1, act), reduction_indices=1)            # predicted Q_value
        values2 = rwd + (1. - is_done) * options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)  # sampled Q_value
    
        loss = tf.reduce_mean(tf.square(values1 - values2))
    
        train_step = options.Optimizer.minimize(loss)
        
        sess.run(tf.initialize_all_variables())        
        
        if options.env_type == 'Atari':
            agent.copy_network(sess)
        #elif options.env_type == 'AtariRAM':
            #agent.copy_network(sess)
    
        # Some initial local variables
        feed = {}
        eps = options.INIT_EPS
        global_step = 0
        exp_pointer = 0
        exp_mem_size = 0
        learning_finished = False
    
        # The replay memory
        obs_queue = np.empty([options.MAX_EXPERIENCE] + options.OBSERVATION_DIM)
        act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
        rwd_queue = np.empty([options.MAX_EXPERIENCE])
        is_done_queue = np.empty([options.MAX_EXPERIENCE])
        next_obs_queue = np.empty([options.MAX_EXPERIENCE] + options.OBSERVATION_DIM)
    
        # Score cache
        score_queue = []
    
        for i_episode in xrange(options.MAX_EPISODE):
    
            observation = env.no_op_reset(options.no_op_max)
            done = False
            score = 0
            sum_loss_value = 0
            epi_step = 0
    
            while not done:
                global_step += 1
                epi_step += 1
    
                if options.LINEAR_EPS_DECAY:
                    if global_step > options.replay_start_size and eps > options.FINAL_EPS:
                        eps -= (options.INIT_EPS - options.FINAL_EPS)/options.EPS_ANNEAL_STEPS
                        eps = max(eps, options.FINAL_EPS)
                else:
                    if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                        eps *= options.EPS_DECAY_RATE
                        eps = max(eps, options.FINAL_EPS)
    
                if options.render:
                    env.render()
    
                obs_queue[exp_pointer] = observation
    
                if env.env_type == 'Atari':
                    action = agent.sample_action(Q1, {obs: np.reshape(observation, tuple([-1] + options.OBSERVATION_DIM))}, eps, options)
                else:
                    action = agent.sample_action(Q1, {obs: np.reshape(observation, (1, -1))}, eps, options)
    
                act_queue[exp_pointer] = action
                observation, reward, done, _ = env.step_repeat(np.argmax(action), options.action_repeat)
                
                if reward > 0:
                    score += reward
                # reward += score / 100  # Reward will be the accumulative score divied by 100
    
                if GAME == 'Acrobot-v0':
                    if done and epi_step < TIMESTEP_LIMIT:
                        reward = 10  # If make it, send a big reward
                        observation = np.zeros_like(observation)
                if GAME == 'CartPole-v0':
                    if done:
                        observation = np.zeros_like(observation)
                if env.env_type == 'Atari':
                    if done:
                        observation = np.zeros_like(observation)
                        reward = -100
                if env.env_type == 'AtariRAM':
                    if done:
                        observation = np.zeros_like(observation)
                        reward = -1
                
                if epi_step > TIMESTEP_LIMIT:
                    done = True
                    
                rwd_queue[exp_pointer] = reward
                next_obs_queue[exp_pointer] = observation
    
                if done:
                    is_done_queue[exp_pointer] = 1
                else:
                    is_done_queue[exp_pointer] = 0
    
                exp_pointer += 1
                exp_mem_size += 1
    
    
                if exp_pointer == options.MAX_EXPERIENCE:
                    exp_pointer = 0  # Refill the replay memory if it is full
                if exp_mem_size > options.MAX_EXPERIENCE:
                    exp_mem_size = options.MAX_EXPERIENCE
    
                if global_step > options.replay_start_size:
                    rand_indexs = np.random.choice(exp_mem_size, options.BATCH_SIZE)
                    feed.update({obs: obs_queue[rand_indexs]})
                    feed.update({act: act_queue[rand_indexs]})
                    feed.update({rwd: rwd_queue[rand_indexs]})
                    feed.update({is_done: is_done_queue[rand_indexs]})
                    feed.update({next_obs: next_obs_queue[rand_indexs]})
                    if not learning_finished:  # If not solved, we train and get the step loss
                        step_loss_value, _ = sess.run([loss, train_step], feed_dict=feed)
                    else:  # If solved, we just get the step loss
                        step_loss_value = sess.run(loss, feed_dict=feed)
                    # Use sum to calculate average loss of this episode
                    sum_loss_value += step_loss_value
    
                if global_step % options.target_net_update_freq == 0:   # copy network (Q- = Q)
                    if options.env_type == 'Atari':
                        agent.copy_network(sess)
                    #elif options.env_type == 'AtariRAM':
                        #agent.copy_network(sess)
                    
                    
            if i_episode % options.test_freq == 0:
                result_txt = "====== Episode {} ended with score = {}, avg_loss = {}, total_step = {}, eps = {} ======".format(i_episode + 1, score,
                                                                                                   sum_loss_value / epi_step, global_step,
                                                                                        eps)
                    
                print result_txt                
                with open(filename, "a") as f:
                    f.write(result_txt + "\n")
    
                global actionset
                global test_num
                
                actionset=[0,0,0,0,0,0]
                test_score, test_step = test(env, agent, Q1, obs, i_episode+1)
    
                test_num += 1
        
                result_txt = "====== Policy test score = {}, step = {}======".format(test_score, test_step)
                
                print result_txt                
                with open(filename, "a") as f:
                    f.write(result_txt + "\n")
                
                t = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                t = "local time = "+ t
                print t
                with open(filename, "a") as f:
                    f.write(t + "\n")
                
                #print actionset
    
                # options.render = True
            else:
                options.render = False
    
    
            score_queue.append(score)
            if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
                score_queue.pop(0)
                if GAME == 'CartPole-v0':
                    if np.mean(score_queue) > 195:  # The threshold of being solved
                        learning_finished = True
                    else:
                        learning_finished = False
                if GAME == 'Acrobot-v0':
                    if np.mean(score_queue) > -100:  # The threshold of being solved
                        learning_finished = True
                    else:
                        learning_finished = False
                if GAME == 'Breakout-v0':
                    if np.mean(score_queue) > 100:  # The threshold of being solved
                        learning_finished = True
                    else:
                        learning_finished = False
                if GAME == 'Breakout-ram-v0':
                    if np.mean(score_queue) > 100:  # The threshold of being solved
                        learning_finished = True
                    else:
                        learning_finished = False
            if learning_finished:
                print "Testing !!!"
    
        del(obs_queue)
        del(act_queue)
        del(rwd_queue)
        del(is_done_queue)
        del(next_obs_queue)
    
    
    env = gym_AtariRAM_Environment(GAME,fast_restart=True)
    #env = gym_ClassicControl_Environment(GAME)
    #env.monitor.start(OUT_DIR, force=True)
    train(env)
    #env.monitor.close()
