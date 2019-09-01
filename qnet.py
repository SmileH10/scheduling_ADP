import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, concatenate, Multiply, Dot
from tensorflow.keras.models import Model
from tqdm import tqdm
from time import sleep
from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
import os

BATCH_SIZE = 64
MEMORY_SIZE = 500000
EPSILON = 0.01
# W_SCALE = 0.01  # init weights with rand normal(0, w_scale)


class QNet(object):
    def __init__(self, sim_env, mc_name_list, mc_sqc, proc_t, num_ptrn):
        self.env = sim_env
        self.mc_name_list = mc_name_list
        self.mc_sqc = mc_sqc
        self.proc_t = proc_t
        self.num_ptrn = num_ptrn

        self.num_nodes = sum(len(mc_sqc[ptrn]) for ptrn in range(num_ptrn))  # 종료 dummy node 포함
        self.time_interval = 0.0
        self.machine = "Not_initialized"

        self.T = 1
        self.embed_dim = 32
        self.gamma = 1
        self.memory = []
        self.epi_memory = []
        self.build_basic_state()
        self.initialize_model()

        self.num_ptrn_que = [[0 for oper in range(len(self.mc_sqc[ptrn]) + 1)] for ptrn in range(num_ptrn)]
        self.num_ptrn_res = [[0 for oper in range(len(self.mc_sqc[ptrn]) + 1)] for ptrn in range(num_ptrn)]

    def build_basic_state(self):
        print("running QNet::build_basic_state..")
        self.nodes = []
        for ptrn in range(self.num_ptrn):
            for oper in range(len(self.mc_sqc[ptrn])):  # 'A' in  ['A', 'B', 'C']
                self.nodes.append((ptrn, oper))
        assert len(self.nodes) == self.num_nodes

        conj_edges = []
        for ptrn in range(24):
            for oper in range(len(self.mc_sqc[ptrn])):  # 'A' in  ['A', 'B', 'C']
                if oper != 'END':
                    conj_edges.append(((ptrn, oper), (ptrn, oper + 1)))
                    conj_edges.append(((ptrn, oper + 1), (ptrn, oper)))

        disj_edges = []
        node_with_same_mc = defaultdict(list)
        for n in self.nodes:
            mc = self.mc_sqc[n[0]][n[1]]
            node_with_same_mc[mc].append(n)
        for mc in self.mc_name_list:
            for n1 in node_with_same_mc[mc]:
                for n2 in node_with_same_mc[mc]:
                    if node_with_same_mc[mc].index(n1) < node_with_same_mc[mc].index(n2):
                        disj_edges.append((n1, n2))
                        disj_edges.append((n2, n1))

        self.neighbor = defaultdict(list)
        for n1 in self.nodes:
            for n2 in self.nodes:
                if (n1, n2) in conj_edges or (n1, n2) in disj_edges:
                    self.neighbor[n1].append(n2)
                    self.neighbor[n2].append(n1)
        print("QNet::build_basic_state Ends")

    def initialize_model(self):
        print("running QNet::initialize_model..")
        mu = defaultdict(dict)
        x, w_conj, w_disj = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0)
        print("1. generating Input layers..")
        sleep(0.1)
        pbar = tqdm(total=self.num_nodes)
        for n in self.nodes:
            x[n] = Input(shape=(2,))
            mu[0][n] = Input(shape=(self.embed_dim,))
            for n2 in self.nodes:
                if n != n2:
                    w_conj[(n, n2)] = Input(shape=(1,))
                    w_disj[(n, n2)] = Input(shape=(1,))
            pbar.update(1)
        pbar.close()
        sleep(0.1)

        print("2. generating Dense layers..")
        # random_normal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.layer_septa1 = Dense(self.embed_dim)
        self.layer_septa2 = Dense(self.embed_dim)
        self.layer_septa3 = Dense(self.embed_dim)
        self.layer_septa4 = Dense(self.embed_dim, activation='relu')
        self.layer_septa3_2 = Dense(self.embed_dim)
        self.layer_septa4_2 = Dense(self.embed_dim, activation='relu')
        self.layer_septa5 = Dense(1)
        self.layer_septa6 = Dense(self.embed_dim, activation='relu')
        self.layer_septa7 = Dense(self.embed_dim, activation='relu')

        print("3. connecting layers..")
        sleep(0.1)
        pbar = tqdm(total=self.num_nodes * self.T + 1)
        hop = 0
        while hop < self.T:  # mu[0][n]: 초기값. mu[1][n] ~ mu[T][n] 까지 새로 구함
            hop += 1
            for n1 in self.nodes:
                output1 = self.layer_septa1(x[n1])
                input2 = Add()([mu[hop - 1][n2] for n2 in self.neighbor[n1]])  # action에 따라 node 바뀌는데, neighbor가 어느위치인지 모르니까 음........ 고민... 아닌가
                output2 = self.layer_septa2(input2)
                output4_for_edges = []
                for n2 in self.neighbor[n1]:
                    output4_for_edges.append(self.layer_septa4(w_conj[n1, n2]))
                output4 = Add()(output4_for_edges)
                output3 = self.layer_septa3(output4)
                output4_2_for_edges = []
                for n2 in self.neighbor[n1]:
                    output4_2_for_edges.append(self.layer_septa4(w_disj[n1, n2]))
                output4_2 = Add()(output4_2_for_edges)
                output3_2 = self.layer_septa3(output4_2)
                mu[hop][n1] = Add()([output1, output2, output3, output3_2])
                pbar.update(1)

        input6 = Add()([mu[self.T][n] for n in self.nodes])

        output6 = self.layer_septa6(input6)
        action = Input(shape=(self.num_nodes, ))
        # mu가 dict이라서 생기는 오류 아닐까? 하지만 dict이어야 원하는 node를 찾을 수 있는데.
        # mu_action = Multiply()([mu[self.T], action])  # mu[T][i] 과 action[i] 가 곱해지는데, mu[T][i] : 64 x 1 이고, action[i]: 1 인데...
        # mu_action = Dot(mu[self.T], action)
        # output7 = self.layer_septa7(mu_action)
        output7 = self.layer_septa7(mu[self.T][(0, 0)])
        input5 = concatenate([output6, output7])
        qhat = self.layer_septa5(input5)
        pbar.update(1)
        pbar.close()

        sleep(0.1)
        print("4. generating model..")
        self.model = Model(inputs=[x[n] for n in self.nodes] + [mu[0][n] for n in self.nodes] +
                                 [w_conj[(n, n2)] for n in self.nodes for n2 in self.nodes if n != n2] +
                                 [w_disj[(n, n2)] for n in self.nodes for n2 in self.nodes if n != n2],
                           outputs=qhat)
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        print("QNet::initialize_model Ends")

    def get_state(self):
        print("running QNet::get_state..")
        node_att = {}
        for ptrn in range(self.num_ptrn):
            for oper in range(len(self.mc_sqc[ptrn])):  # 'A' in  ['A', 'B', 'C']
                node_att[(ptrn, oper)] = [self.num_ptrn_que[ptrn][oper], self.num_ptrn_res[ptrn][oper]]
            node_att[(ptrn, len(self.mc_sqc[ptrn]))] = [0, 0]

        conj_edges = []
        conj_edge_att = defaultdict(lambda: 0)
        for ptrn in range(self.num_ptrn):
            for oper in range(len(self.mc_sqc[ptrn]) - 1):  # 'A' in  ['A', 'B', 'C']
                conj_edges.append(((ptrn, oper), (ptrn, oper + 1)))
                conj_edge_att[((ptrn, oper), (ptrn, oper + 1))] = self.proc_t[ptrn][oper]
                conj_edge_att[((ptrn, oper + 1), (ptrn, oper))] = -1 * self.proc_t[ptrn][oper]

        disj_edges = []
        disj_edge_att = defaultdict(lambda : 0)
        node_with_same_mc = defaultdict(list)
        for node in self.nodes:
            mc = self.mc_sqc[node[0]][node[1]]
            node_with_same_mc[mc].append(node)
        for mc in self.mc_name_list:
            for node1 in node_with_same_mc[mc]:
                for node2 in node_with_same_mc[mc]:
                    if node_with_same_mc[mc].index(node1) < node_with_same_mc[mc].index(node2):
                        disj_edges.append((node1, node2))
                        if len(self.machine[mc].users) > 0:
                            disj_edge_att[(node1, node2)] = self.env.now - self.machine[mc].users[0].obj.oper_start_t
                        else:
                            disj_edge_att[(node1, node2)] = self.proc_t[ptrn][oper]
                        disj_edge_att[(node2, node1)] = disj_edge_att[(node1, node2)]

        mu = defaultdict(dict)
        for n in self.nodes:
            mu[0][n] = 0

        state = [node_att[self.nodes[i]] for i in range(len(self.nodes))] + [mu[0][n] for n in self.nodes] + \
                [conj_edge_att[(n, n2)] for n in self.nodes for n2 in self.nodes if n != n2] + \
                [disj_edge_att[(n, n2)] for n in self.nodes for n2 in self.nodes if n != n2]

        print("QNet::get_state Ends")
        return state

    def get_action_list(self, state=False, mc_name=0):
        action_list = []
        if mc_name:
            for qjob in self.machine[mc_name].queue:
                if not (qjob.obj.pattern, qjob.obj.oper_num) in action_list:
                    action_list.append((qjob.obj.pattern, qjob.obj.oper_num))

        elif state:
            mc_numque = {}
            mc_numuser = {}
            for mc in self.mc_name_list:
                mc_numque[mc] = 0
                mc_numuser[mc] = 0
            for i in range(len(self.nodes)):
                mc_numque[self.mc_sqc[self.nodes[i][0]][self.nodes[i][1]]] += state[i][0]
                mc_numuser[self.mc_sqc[self.nodes[i][0]][self.nodes[i][1]]] += state[i][1]

            for mc in self.mc_name_list:
                if mc_numque[mc] >= 2 and  mc_numuser[mc] == 0:
                    mc_name = mc
                    break

            for i in range(len(self.nodes)):
                if self.mc_sqc[ptrn][oper_num] == mc_name:
                    ptrn = self.nodes[i][0]
                    oper_num = self.nodes[i][1]
                    if not (ptrn, oper_num) in action_list:
                        action_list.append((ptrn, oper_num))
        else:
            raise Exception ("[Error] QNet::get_action_list")

        return action_list

    def get_qvalue(self, state, action_node):
        # node_idx 찾기
        state_action = deepcopy(state)
        state_action[self.nodes.index(action_node)][0] -= 1
        state_action[self.nodes.index(action_node)][1] += 1
        assert state_action[self.nodes.index(action_node)][0] >= 0
        assert state_action[self.nodes.index(action_node)][1] == 1
        return self.model.predict(state_action)

    def save_state_action(self, S, v, t):
        self.epi_memory.append((S, v, t))

    def update_nstpe_memory(self, R, T, step=1):
        for i in range(len(self.epi_memory)):
            if i + step < len(self.epi_memory):
                s = self.epi_memory[i][0]
                v = self.epi_memory[i][1]
                r = R * (self.epi_memory[i+step][2] - self.epi_memory[i][2]) / T
                sp = self.epi_memory[i+1][0]
                self.save_memory(s, v, r, sp)
            else:
                break
        self.epi_memory = []

    def save_memory(self, S_t_minus_n, v_t_minus_n, n_cumulative_R, S_t):
        if len(self.memory) > MEMORY_SIZE:
            del self.memory[0]
        self.memory.append((S_t_minus_n, v_t_minus_n, n_cumulative_R, S_t))
        assert len(self.memory) <= MEMORY_SIZE

    def train_model(self):
        assert self.memory
        memory_sample = random.sample(self.memory, BATCH_SIZE)
        y = []
        q = []
        for m in memory_sample:
            s_t = m[0]
            v_t = m[1]
            cum_r = m[2]
            s_t_plus_n = m[3]
            next_action_list = []  # action_list는 미리 저장할 수 있는데, qhat(s_t_plus_n, a)는 미리 저장하면 안되고,
                                   # 그때그때 업데이트된 세타를 반영해야
            y.append(cum_r + self.gamma * max(self.get_qvalue(s_t_plus_n, action_node) for action_node in next_action_list))
            q.append(self.get_qvalue(s_t, v_t))
        self.model.fit(y, q, batch_size=BATCH_SIZE, epochs=100)  # 진짜 데이터 입력


    # def reward(self, S, v):
    #     return 10
    #
    # def qhat(self, S, v):
    #     # S[v] 정의 필요  (v 노드를 0번째 위치로 옮기기)
    #     return self.model.predict(S[v])
    #
    # def max_qhat(self, S):
    #     action_list = []  # 수정 필요
    #     max_val = 0.0
    #     for v in action_list:
    #         v_val = self.model.predict(S, v)
    #         if max_val < v_val:
    #             max_val = v_val
    #
    # def action(self):
    #     assert
    #     if random.random() < EPSILON:
    #         action_node = random.sample(nodes)
    #     else:
    #         predictions = self.model.predict(test_images)