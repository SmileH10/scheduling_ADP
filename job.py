import numpy as np
from time import time


class Job(object):
    def __init__(self, id, simenv, prior_rule, sim_mcrsc, ptrn_info, mc_info, g):
        self.id = id
        self.simenv = simenv
        self.prior_rule = prior_rule
        self.sim_mcrsc = sim_mcrsc
        self.ptrn_info = ptrn_info
        self.mc_info = mc_info
        self.g = g
        self.arrt = simenv.now
        self.qnet = False
        self.waiting_t = 0.0
        self.n_key = 'S'

        # Run
        self.update_nkey_and_g()
        self.run_nkey()

    def job_select(self, mc):
        if self.prior_rule == "RL":
            action = self.qnet.run_job_selection(mc, self.simenv.now)
            if not action:
                pass
            else:
                best_job_idx = "None"
                longest_arrt = float("inf")
                for qjob in self.sim_mcrsc[mc].queue:
                    if qjob.obj.n_key == action:
                        if qjob.obj.arrt < longest_arrt:
                            best_job_idx = self.sim_mcrsc[mc].queue.index(qjob)
                            longest_arrt = qjob.obj.arrt
                assert best_job_idx != "None"
                temp = self.sim_mcrsc[mc].queue[0]
                self.sim_mcrsc[mc].queue[0] = self.sim_mcrsc[mc].queue[best_job_idx]
                self.sim_mcrsc[mc].queue[best_job_idx] = temp

        elif self.prior_rule == 'SPT':
            if len(self.sim_mcrsc[mc].queue) >= 2:
                best_value = 999
                best_qloc = 0
                qloc = 0
                for user_req in self.sim_mcrsc[mc].queue:
                    if self.g.proct[user_req.obj.n_key] < best_value:
                        best_value = user_req.obj.id
                        best_qloc = qloc
                        qloc += 1
                temp = self.sim_mcrsc[mc].queue[0]
                self.sim_mcrsc[mc].queue[0] = self.sim_mcrsc[mc].queue[best_qloc]
                self.sim_mcrsc[mc].queue[best_qloc] = temp

        elif self.prior_rule == 'LPT':
            if len(self.sim_mcrsc[mc].queue) >= 2:
                best_value = 0
                best_qloc = 0
                qloc = 0
                for user_req in self.sim_mcrsc[mc].queue:
                    if self.g.proct[user_req.obj.n_key] > best_value:
                        best_value = user_req.obj.id
                        best_qloc = qloc
                        qloc += 1
                temp = self.sim_mcrsc[mc].queue[0]
                self.sim_mcrsc[mc].queue[0] = self.sim_mcrsc[mc].queue[best_qloc]
                self.sim_mcrsc[mc].queue[best_qloc] = temp

    def update_nkey_and_g(self):
        cand_ns = [key[1] for key in self.g.e_cj.keys() if key[0] == self.n_key]
        next_nkey = np.random.choice(cand_ns, 1, p=[self.g.e_cj[(self.n_key, n_next)] for n_next in cand_ns])[0]
        self.n_key = next_nkey
        if self.n_key != 'T':
            self.g.n_x[self.n_key]['wait'] += 1

    def run_nkey(self):
        if self.n_key != 'T':
            next_mc = self.n_key[-1]
            self.simenv.process(self.run_mc(next_mc))
        else:
            # 결과 기록
            print("job %d ends at %.1f with WT %.1f" % (self.id, self.simenv.now, self.waiting_t))

    def run_mc(self, mc):
        mc_arrt = self.simenv.now
        # print('job %d (nkey:%s) arrives at mc %s at t = %.2f' % (self.id, self.n_key, self.n_key[-1], self.simenv.now))
        # print('[graph_info] srvd: %d, wait: %d, rsvd: %d'
        #       % (self.g.n_x[self.n_key]['srvd'], self.g.n_x[self.n_key]['wait'], self.g.n_x[self.n_key]['rsvd']))
        with self.sim_mcrsc[mc].request() as req:
            req.obj = self
            yield req
            if self.g.n_x[self.n_key]['rsvd'] == 1:
                assert len(self.sim_mcrsc[mc].user) == 0
                self.g.n_x[self.n_key]['rsvd'] -= 1
            else:
                self.g.n_x[self.n_key]['wait'] -= 1
            self.g.n_x[self.n_key]['srvd'] += 1
            self.waiting_t += self.simenv.now - mc_arrt
            assert 1 == sum(self.g.n_x[n]['srvd'] for n in self.g.n_x.keys() if n[-1] == mc)
            yield self.simenv.timeout(np.random.exponential(self.g.proct[self.n_key]))
            self.g.n_x[self.n_key]['srvd'] -= 1
            # print('job %d (nkey:%s) ends at mc %s at t = %.2f' % (self.id, self.n_key, self.n_key[-1], self.simenv.now))
            self.update_nkey_and_g()
            # print('job %d moves at mc %s (nkey:%s) at t = %.2f' % (self.id, self.n_key[-1], self.n_key, self.simenv.now))
            self.job_select(mc)
        # with 절을 나오면서 self.sim_mcrsc[mc].release(req) 를 한다고 함.
        self.run_nkey()
