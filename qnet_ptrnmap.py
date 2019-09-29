import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, concatenate, Multiply, Dot
from tensorflow.keras.models import Model
from time import time
from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BATCH_SIZE = 4 #  64
MEMORY_SIZE = 500000
EPSILON = 0.01
# W_SCALE = 0.01  # init weights with rand normal(0, w_scale)


class QNet(object):
    def __init__(self,  ptrn_info, mc_info, g):
        self.ptrn_info = ptrn_info
        self.mc_info = mc_info
        self.g = g

        self.T = 3
        self.embed_dim = 32
        self.initialize_model()

        self.epsilon = 0.05  # random action prob.
        self.short_memory = []
        self.memory = []
        self.gamma_unit = 0.9999 ** 60  # 시간단위를 60초에 0.9999로 하려고. (=0.994)

    def initialize_model(self):
        print("running QNet::initialize_model..")
        mu = defaultdict(dict)
        x = defaultdict(lambda: 0)
        for n in self.g.n_x.keys():
            if n == 'S' or n == 'T':
                x[n] = Input(shape=(2,))
                mu[0][n] = Input(shape=(self.embed_dim,))
            else:
                x[n] = Input(shape=(len(self.g.n_features),))
                mu[0][n] = Input(shape=(self.embed_dim,))
        # random_normal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.layer_septa1_1 = Dense(self.embed_dim)
        self.layer_septa1_2 = Dense(self.embed_dim)
        self.layer_septa2 = Dense(self.embed_dim)
        self.layer_septa3 = Dense(self.embed_dim)
        self.layer_septa4 = Dense(self.embed_dim)
        self.layer_septa5 = Dense(self.embed_dim, activation='relu')
        self.layer_septa6 = Dense(1)

        hop = 0
        while hop < self.T:  # mu[0][n]: 초기값. mu[1][n] ~ mu[T][n] 까지 새로 구함
            hop += 1
            for n1 in self.g.n_x.keys():
                if n1 == 'S':
                    output1 = self.layer_septa1_2(x[n1])
                    # conjunctive_next-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[0] == n1) == 1:
                        input3 = [mu[hop - 1][key[1]] for key in self.g.e_cj.keys() if key[0] == n1][0]
                    else:
                        input3 = Add()([self.g.e_cj[key] * mu[hop - 1][key[1]] for key in self.g.e_cj.keys() if key[0] == n1])
                    output3 = self.layer_septa3(input3)
                    input_mu = Add()([output1, output3])
                elif n1 == 'T':
                    output1 = self.layer_septa1_2(x[n1])
                    # conjunctive_previous-nodes
                    input4 = Add()([self.g.e_cj[key] * mu[hop - 1][key[0]] for key in self.g.e_cj.keys() if key[1] == n1])
                    output4 = self.layer_septa4(input4)
                    input_mu = Add()([output1, output4])
                else:
                    output1 = self.layer_septa1_1(x[n1])
                    # disjunctive nodes
                    if len(self.g.e_dj[n1]) == 1:
                        input2 = [mu[hop - 1][n2] for n2 in self.g.e_dj[n1]][0]  # n2: 자기 자신
                    else:
                        input2 = Add()([mu[hop - 1][n2] for n2 in self.g.e_dj[n1]])
                    output2 = self.layer_septa2(input2)
                    # conjunctive_next-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[0] == n1) == 1:
                        input3 = [mu[hop - 1][key[1]] for key in self.g.e_cj.keys() if key[0] == n1][0]
                    else:
                        input3 = Add()([self.g.e_cj[key] * mu[hop - 1][key[1]] for key in self.g.e_cj.keys() if key[0] == n1])
                    output3 = self.layer_septa3(input3)
                    # conjunctive_previous-nodes
                    if sum(1 for key in self.g.e_cj.keys() if key[1] == n1) == 1:
                        input4 = [mu[hop - 1][key[0]] for key in self.g.e_cj.keys() if key[1] == n1][0]
                    else:
                        input4 = Add()([self.g.e_cj[key] * mu[hop - 1][key[0]] for key in self.g.e_cj.keys() if key[1] == n1])
                    output4 = self.layer_septa4(input4)
                    input_mu = Add()([output1, output2, output3, output4])
                mu[hop][n1] = tf.keras.activations.relu(input_mu)  # from keras.layers import Activation '\n' model.add(Activation('relu'))
        input5 = Add()([mu[self.T][n] for n in self.g.n_x.keys()])
        output5 = self.layer_septa5(input5)
        qhat = self.layer_septa6(output5)

        self.model = Model(inputs=[x[n] for n in self.g.n_x.keys()] + [mu[0][n] for n in self.g.n_x.keys()],
                           outputs=qhat)
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        print(self.model.summary())
        print("ends QNet::initialize_model")

    @staticmethod
    def get_post_state(s_nx, action):
        if action is None:
            return s_nx
        else:
            post_s_nx = deepcopy(s_nx)
            print('post_s_nx: ', post_s_nx)
            print('post_s_nx[action]: ', post_s_nx[action])
            if post_s_nx[action]['wait'] > 0:
                post_s_nx[action]['wait'] -= 1
                post_s_nx[action]['srvd'] += 1
            # elif len(action) > 2 and s_nx[action[:len(action)-1]]['srvd'] == 1 and s_nx[action]['wait'] == 0:
            #     post_s_nx[action]['rsvd'] += 1
            return post_s_nx

    def run_job_selection(self, mc, simenv_now):
        s_nx = deepcopy(self.g.n_x)
        # action_cands_wait = []
        # for n in s_nx.keys():
        #     if n[-1] == mc:
        #         if s_nx[n]['wait'] > 0:
        #             action_cands_wait.append(n)
        action_cands_wait = [n for n in s_nx.keys() if n[-1] == mc and s_nx[n]['wait'] > 0]
        # action_cands_rsvd = [n for n in s_nx.keys() if len(n) > 2 and s_nx[n[:len(n)-1]]['srvd'] == 1 and s_nx[n]['wait'] == 0 and n[-1] == mc]
        print('action_cands_wait: ', action_cands_wait)
        if np.random.random() < self.epsilon:
            action_cands = action_cands_wait # + action_cands_rsvd
            if action_cands:
                best_a = np.random.choice(action_cands)[0]
            else:
                best_a = None
        else:
            best_q = float("inf")
            best_a = None
            for a in action_cands_wait:
                post_s_nx = self.get_post_state(s_nx, a)
                x = self.make_input_shape(post_s_nx)
                qhat = self.model.predict(x)  # np.expand_dims(x, axis=0)
                if qhat < best_q:
                    best_q = qhat
                    best_a = a
            # for a in action_cands_rsvd:
            #     post_s_nx = self.get_post_state(s_nx, a)
            #     x = self.make_input_shape(post_s_nx)
            #     qhat = self.model.predict(x)
            #     if qhat < best_q:
            #         best_q = qhat
            #         best_a = a

        best_post_s = self.get_post_state(s_nx, best_a)
        if best_a:
            self.save_state_action(best_post_s, simenv_now)
            self.g.n_x = deepcopy(best_post_s)
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
        num_waitjob = sum(post_n_x[n]['wait'] for n in post_n_x.keys() if n != 'S' and n != 'T')
        if len(self.short_memory) == 0:
            self.short_memory.append((post_n_x, 0, t))  # (s^a, r, t)
        else:
            gamma = self.gamma_unit ** (t - self.short_memory[-1][2])
            reward = num_waitjob * (1 + gamma)/2  # 직전 시점부터 현재 t 시점까지의 reward
            self.short_memory.append((post_n_x, reward, t))  # (현재 t 시점의 상태, t-1~t 누적 보상, 현재 시점 t)
        self.save_nstpe_memory(step=1)
        if len(self.memory) > BATCH_SIZE:
            self.train_model()

    def save_nstpe_memory(self, step=1):
        if len(self.short_memory) > step:
            post_s_t_minus_n = self.short_memory[-1-step][0]
            reward = sum(self.short_memory[i][1] for i in range(-step, 0))
            gamma = self.gamma_unit ** (self.short_memory[-1][2] - self.short_memory[-1-step][2])
            post_s_t = self.short_memory[-1][0]
            if len(self.memory) > MEMORY_SIZE:
                del self.memory[0]
            self.memory.append((post_s_t_minus_n, reward, post_s_t, gamma))
            del self.short_memory[0]

    def train_model(self):
        memory_sample = random.sample(self.memory, BATCH_SIZE)
        print('memory_sample: ', memory_sample)
        y = []
        q = []
        for m in memory_sample:
            s_t_minus_n = m[0]
            r = m[1]
            s_t = m[2]
            gamma = m[3]
            y.append(r + gamma * self.model.predict(self.make_input_shape(s_t)))
            # q.append(self.model.predict(self.make_input_shape(s_t_minus_n)))

            # next_action_list = self.get_action_list(state=s_t_plus_n)
            # action_list는 미리 저장할 수 있는데, qhat(s_t_plus_n, a)는 미리 저장하면 안되고,
                                   # 그때그때 업데이트된 세타를 반영해야
            # y.append(r + self.gamma * max(self.get_qvalue(s_t_plus_n, action_node) for action_node in next_action_list))
            # q.append(self.get_qvalue(s_t, v_t))
        print([self.make_input_shape(m[0]) for m in memory_sample])
        print(np.shape([self.make_input_shape(m[0]) for m in memory_sample]))  # 4 * 88 * 1 이 나오는데, 88 * 4 * 1 이 나오게끔 해야 해.
        self.model.fit([self.make_input_shape(m[0]) for m in memory_sample], y)
        # self.model.fit(y, q, batch_size=BATCH_SIZE, epochs=100)  # 진짜 데이터 입력
