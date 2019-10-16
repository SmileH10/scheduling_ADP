"""
version 1.
Post-state (action 직후 변화하는 node 속성을 input으로 q-hat(output)을 구함.
19.10.01 세미나 발표 버전.
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Add, Activation, Lambda, Concatenate #, Multiply, Dot
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import optimizers

import json

# import keras
# from keras.layers import Input, Dense, Add  #, concatenate, Multiply, Dot
# from keras.models import Model, model_from_json
# from keras import activations

from time import time
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import numpy as np
import logging, os
import sys
print(sys.version)
print(tf.__version__)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# W_SCALE = 0.01  # init weights with rand normal(0, w_scale)


class QNet(object):
    def __init__(self, args, ptrn_info, mc_info, g, simenv, model=False, test=False):
        self.args = args
        self.ptrn_info = ptrn_info
        self.mc_info = mc_info
        self.g = g
        # self.gd = args['gd']
        if not test:
            self.model_dir = args['dir']['qnet_model_dir']
            self.loss_fig_dir = args['dir']['loss_fig_dir']
        self.simenv = simenv
        self.test = test

        self.epsilon = 0.05  # random action prob.
        self.short_memory = []
        self.memory = []
        self.cur_epoch = 0

        self.T = 3
        self.embed_dim = 64
        self.initialize_model(model=model)

        self.gamma_unit = 0.9999  # 0.9999 ** 60  # 시간단위를 60초에 0.9999로 하려고. (=0.994)
        self.loss_history = []
        self.loss_history_zoomin = []
        self.loss_history_zoomin_raw = []

        self.time = {'predict': 0.0, 'train': 0.0, 'initialize_model': 0.0}

    def initialize_model(self, model=False):
        if model:
            print("loading QNet model.. ", end=' ')
            self.model = load_model(model['model_loaded'])
            if not self.test:
                with open(model['memory'], "r") as read_file:
                    self.memory = json.load(read_file)
                self.cur_epoch = model['load_epoch']
                print('restart from epochs %d ..' % self.cur_epoch)
            print("QNet model loaded.")
            return
        print("running QNet::initialize_model..")
        mu = {(0, nkey): Input(shape=(self.embed_dim,), name='mu_%d%s' % (0, nkey))
              for nkey in self.g.n_x.keys()}
        x = {nkey: Input(shape=(len(self.g.n_features),), name='x_%s' % nkey) for nkey in self.g.n_x.keys() if nkey != 'S' and nkey != 'T'}
        x['S'] = Input(shape=(2,), name='x_%s' % 'S')
        x['T'] = Input(shape=(2,), name='x_%s' % 'T')

        # random_normal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.layer_septa1_1 = Dense(self.embed_dim, name='1-1')
        self.layer_septa1_2 = Dense(self.embed_dim, name='1-2')
        self.layer_septa2 = Dense(self.embed_dim, name='2')
        self.layer_septa3 = Dense(self.embed_dim, name='3')
        self.layer_septa4 = Dense(self.embed_dim, name='4')
        self.layer_septa5 = Dense(self.embed_dim, activation='relu', name='5')
        self.layer_septa6 = Dense(1, name='6')

        hop = 0
        new_mu = {}
        while hop < self.T:  # mu[0][n]: 초기값. mu[1][n] ~ mu[T][n] 까지 새로 구함
            hop += 1
            for n1 in self.g.n_x.keys():
                if n1 == 'S':
                    output1 = self.layer_septa1_2(x[n1])
                    # conjunctive_next-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[0] == n1) == 1:  # 100% 여기
                        input3 = [mu[(hop - 1, key[1])] for key in self.g.e_cj.keys() if key[0] == n1][0]  # tf.convert_to_tensor
                    # else:  # else 방문 안 함. 지금 case에서 next_key는 1개니까.
                        # input3 = Add()([Lambda(lambda xm: xm * self.g.e_cj[key])(mu[(hop - 1, key[1])])
                        #                 for key in self.g.e_cj.keys() if key[0] == n1])
                    output3 = self.layer_septa3(input3)
                    input_mu = Add()([output1, output3])
                elif n1 == 'T':
                    output1 = self.layer_septa1_2(x[n1])
                    # conjunctive_previous-nodes
                    # 지금 case에서는 self.g.e_cj[key] 가 모두 1이라서 안곱해줘도 됨.
                    input4 = Add()([mu[(hop - 1, key[0])] for key in self.g.e_cj.keys() if key[1] == n1])
                    # input4 = Add()([self.g.e_cj[key] * mu[(hop - 1, key[0])] for key in self.g.e_cj.keys() if key[1] == n1])
                    output4 = self.layer_septa4(input4)
                    input_mu = Add()([output1, output4])
                else:
                    output1 = self.layer_septa1_1(x[n1])
                    # disjunctive nodes
                    if len(self.g.e_dj[n1]) == 1:
                        input2 = mu[(hop - 1, n1)]
                    else:
                        input2 = Add()([mu[(hop - 1, n2)] for n2 in self.g.e_dj[n1]])
                    output2 = self.layer_septa2(input2)
                    # conjunctive_next-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[0] == n1) == 1:
                        input3 = [mu[(hop - 1, key[1])] for key in self.g.e_cj.keys() if key[0] == n1][0]
                    else:
                        for key in self.g.e_cj.keys():
                            if key[0] == n1:
                                temp_prob = self.g.e_cj[key]
                                new_mu[(hop - 1, key[1])] = Lambda(lambda xm: xm * temp_prob)(mu[(hop - 1, key[1])])
                        input3 = Add()([new_mu[(hop - 1, key[1])] for key in self.g.e_cj.keys() if key[0] == n1])
                    output3 = self.layer_septa3(input3)
                    # conjunctive_previous-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[1] == n1) == 1:  # 100% 여기
                        input4 = [mu[(hop - 1, key[0])] for key in self.g.e_cj.keys() if key[1] == n1][0]
                    # else:  # 무조건 1개
                        # input4 = Add()([self.g.e_cj[key] * mu[(hop - 1, key[0])] for key in self.g.e_cj.keys() if key[1] == n1])
                        # input4 = Add()([mu[(hop - 1, key[0])] for key in self.g.e_cj.keys() if key[1] == n1])
                    output4 = self.layer_septa4(input4)
                    input_mu = Add()([output1, output2, output3, output4])
                mu[(hop, n1)] = Activation('relu', name='act_hop%d_node%s' % (hop, n1))(input_mu)

        input5 = Add()([mu[(self.T, n)] for n in self.g.n_x.keys()])
        output5 = self.layer_septa5(input5)
        qhat = self.layer_septa6(output5)

        self.model = Model(inputs=[x[n] for n in self.g.n_x.keys()] + [mu[(0, n)] for n in self.g.n_x.keys()],
                           outputs=qhat)
        # adam = optimizers.Adam(lr=0.0001)
        # self.model.compile(optimizer=adam, loss='mean_squared_error')
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # checkpoint_path = self.model_dir  # "training_1/cp.ckpt"
        # self.cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path + "cp.ckpt",
        #                                                       save_weights_only=True,
        #                                                       verbose=1,
        #                                                       period=100)
        # print(self.model.summary())
        print("ends QNet::initialize_model")

    @staticmethod
    def get_post_state(s_nx, action):
        if action is None:
            return s_nx
        else:
            post_s_nx = deepcopy(s_nx)
            if post_s_nx[action]['wait'] > 0:
                post_s_nx[action]['wait'] -= 1
                post_s_nx[action]['srvd'] += 1
            # elif len(action) > 2 and s_nx[action[:len(action)-1]]['srvd'] == 1 and s_nx[action]['wait'] == 0:
            #     post_s_nx[action]['rsvd'] += 1
            return post_s_nx

    def run_job_selection(self, mc, simenv_now):
        s_nx = deepcopy(self.g.n_x)
        action_cands_wait = [n for n in s_nx.keys() if n[-1] == mc and s_nx[n]['wait'] > 0]
        # action_cands_rsvd = [n for n in s_nx.keys() if len(n) > 2 and s_nx[n[:len(n)-1]]['srvd'] == 1 and s_nx[n]['wait'] == 0 and n[-1] == mc]
        if np.random.random() < self.epsilon:
            action_cands = action_cands_wait  # + action_cands_rsvd
            if action_cands:
                best_a = np.random.choice(action_cands)
            else:
                best_a = None
        else:
            best_q = float("inf")
            best_a = None
            for a in action_cands_wait:
                post_s_nx = self.get_post_state(s_nx, a)
                x = self.make_input_shape(post_s_nx)
                start_time = time()
                qhat = self.model.predict(x)[0][0]  # np.expand_dims(x, axis=0)
                self.time['predict'] += time() - start_time
                if qhat < best_q:
                    best_q = qhat
                    best_a = a
            # for a in action_cands_rsvd:
            #     post_s_nx = self.get_post_state(s_nx, a)
            #     x = self.make_input_shape(post_s_nx)
            #     qhat = self.model.predict(x)[0][0]
            #     if qhat < best_q:
            #         best_q = qhat
            #         best_a = a

            # print('best_q: %.2f' % best_q)
        best_post_s = self.get_post_state(s_nx, best_a)
        if best_a:
            start_time = time()
            self.save_state_action(best_post_s, simenv_now)
            self.time['train'] += time() - start_time
        # print("predict: %.1f, train: %.1f, model: %.1f" % (self.time['predict'], self.time['train'], self.time['initialize_model']))
        return best_a

    def make_input_shape(self, n_x):  # qnet으로 옮기기
        x = []
        for n in n_x.keys():
            if n == 'S':
                x.append([[1, 0]])
            elif n == 'T':
                x.append([[0, 1]])
            else:
                x.append([[n_x[n][att] for att in n_x[n].keys()]])
        for n in n_x.keys():
            if n == 'S':
                x.append([[0 for _ in range(self.embed_dim)]])
            elif n == 'T':
                x.append([[0 for _ in range(self.embed_dim)]])
            else:
                x.append([[0 for _ in range(self.embed_dim)]])
        return x

    def save_state_action(self, post_n_x, t):
        if self.test:
            return
        else:
            num_waitjob = sum(post_n_x[n]['wait'] for n in post_n_x.keys() if n != 'S' and n != 'T')
            if len(self.short_memory) == 0:
                self.short_memory.append((post_n_x, 0, t))  # (s^a, r, t)
            else:
                gamma = self.gamma_unit ** (t - self.short_memory[-1][2])
                reward = num_waitjob * (1 + gamma)/2  # 직전 시점부터 현재 t 시점까지의 reward
                # print('t: %.1f, one_time_reward: %.2f, gamma: %.4f' % (t, reward, gamma))
                self.short_memory.append((post_n_x, reward, t))  # (현재 t 시점의 상태, t-1~t 누적 보상, 현재 시점 t)
            self.save_nstep_memory(step=self.args['NSTEP'])

    def save_nstep_memory(self, step):
        if len(self.short_memory) > step:
            post_s_t_minus_n = self.short_memory[-1-step][0]
            reward = sum(self.short_memory[i][1] for i in range(-step, 0))
            gamma = self.gamma_unit ** (self.short_memory[-1][2] - self.short_memory[-1-step][2])
            post_s_t = self.short_memory[-1][0]
            if len(self.memory) > self.args['MEMORY_SIZE']:
                del self.memory[0]
            self.memory.append((post_s_t_minus_n, reward, post_s_t, gamma))
            del self.short_memory[0]
            # if len(self.memory) % 100 == 0:
            #     print('memory_storage: ', len(self.memory))
            if len(self.memory) > self.args['BATCH_SIZE']:
                self.train_model()

    def train_model(self):
        memory_sample = random.sample(self.memory, self.args['BATCH_SIZE'])
        y = []
        x_train = [[] for _ in range(88)]
        for m in memory_sample:
            s_t_minus_n = m[0]
            x = self.make_input_shape(s_t_minus_n)
            for i in range(len(x)):
                x_train[i] += x[i]
            r = m[1]
            s_t = m[2]
            gamma = m[3]
            y.append([r + gamma * self.model.predict(self.make_input_shape(s_t))[0][0]])

        self.cur_epoch += 1
        # self.epsilon = max(0.05 * (1 - self.cur_epoch / 10000), 0)

        # train_on_batch로 돌리기
        loss = self.model.train_on_batch(x_train, [y])  # learning rate 지정해야.
        self.loss_history.append(loss)
        loss_avg = sum(self.loss_history[i] for i in range(len(self.loss_history))) / len(self.loss_history)
        if loss_avg > 100000:
            self.loss_history_zoomin.append(101000)
        else:
            self.loss_history_zoomin.append(loss_avg)

        # model.fit으로 돌리기
        # history = self.model.fit(x_train, [y], batch_size=BATCH_SIZE, verbose=2, callbacks=[self.cp_callback])
        # self.loss_history.append(history.history['loss'][0])

        if (self.cur_epoch <= 100 and self.cur_epoch % 10 == 0) or self.cur_epoch % 500 == 0:
            self.model.save(self.model_dir + 'model_epoch%d.h5' % self.cur_epoch)
            with open(self.model_dir + 'memory_epoch%d.json' % self.cur_epoch, 'w') as fout:
                json.dump(self.memory, fout)
            with open(self.model_dir + 'time at epoch.txt', 'a') as file:
                file.write("[epoch %d] cur_t: %.1f(%dd-%dh-%dm-%ds)\n"
                           % (self.cur_epoch,
                              self.simenv.now, self.simenv.now / (24 * 60 * 60), self.simenv.now % (24 * 60 * 60) / (60.0 * 60),
                              self.simenv.now % (60 * 60) / 60.0,
                              self.simenv.now % 60))
            # 전체 Loss 그래프
            plt.plot(self.loss_history)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig(self.loss_fig_dir + 'Epoch-{}.png'.format(self.cur_epoch))
            plt.close()
            # Loss 20000 이상 짜른 Loss 그래프
            plt.plot(self.loss_history_zoomin)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig(self.loss_fig_dir + 'zoomin_Epoch-{}.png'.format(self.cur_epoch))
            plt.close()

            if len(self.loss_history) >= 100:
                print('avg_100/%d_loss: %.1f' % (len(self.loss_history), sum(self.loss_history[-100:][i] for i in range(100)) / 100))
