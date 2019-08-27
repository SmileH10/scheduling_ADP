import simpy
from simulation_GUI import GraphicDisplay
import numpy as np  # 나중에 random seed 설정할 때, proc_t 확률적으로 쓸 때
from collections import defaultdict
from datetime import datetime
import os


ITERATION = 3
NUM_JOBS = 100
NUM_PATTERN = 24


def load_job_features(mc_name_list):
    oper_sqc = {  # 비응급환자
        0: ['A', 'B', 'I'],
        1: ['A', 'B', 'D', 'I'],
        2: ['A', 'B', 'E', 'I'],
        3: ['A', 'B', 'C', 'D', 'I'],
        4: ['A', 'B', 'C', 'J'],
        5: ['A', 'B', 'D', 'C', 'J'],
        6: ['A', 'B', 'C', 'I'],
        7: ['A', 'B', 'D', 'C', 'I'],
        8: ['A', 'B', 'F', 'I'],
        9: ['A', 'B', 'C', 'D', 'J'],
        10: ['A', 'B', 'C', 'E', 'I'],
        11: ['A', 'B', 'E', 'C', 'J'],
        # 응급환자
        12: ['A', 'B', 'C', 'D', 'E', 'J'],
        13: ['A', 'B', 'C', 'D', 'E', 'I'],
        14: ['A', 'B', 'C', 'D', 'I'],  # ABCDI 3번과 같음
        15: ['A', 'B', 'E', 'C', 'D', 'J'],
        16: ['A', 'B', 'C', 'E', 'D', 'J'],
        17: ['A', 'B', 'C', 'E', 'D', 'I'],
        18: ['A', 'B', 'E', 'C', 'D', 'I'],
        19: ['A', 'B', 'C', 'D', 'E', 'D', 'J'],
        20: ['A', 'B', 'C', 'D', 'F', 'E', 'I'],
        21: ['A', 'B', 'C', 'D', 'F', 'E', 'J'],
        22: ['A', 'B', 'C', 'D', 'G', 'E', 'J'],
        23: ['A', 'B', 'D', 'C', 'E', 'J']
    }
    pattern_freq = {0: 2699, 1: 1390, 2: 876, 3: 694, 4: 358, 5: 292,
                6: 288, 7: 228, 8: 210, 9: 114, 10: 74, 11: 62,
                12: 885, 13: 523, 14: 256, 15: 206, 16: 197, 17: 116,
                18: 98, 19: 76, 20: 63, 21: 46, 22: 45, 23: 44}
    for k in pattern_freq.keys():
        pattern_freq[k] /= 61.0

    pattern_prob = []
    sum_pattern_freq = sum(pattern_freq[k] for k in pattern_freq.keys())
    for k in range(NUM_PATTERN):
        pattern_prob.append(pattern_freq[k] / sum_pattern_freq)

    util = 0.85
    proc_t_ratio = 2  # 응급/비응급환자의 process time 비율 (>=1)
    assert proc_t_ratio >= 1
    mc_freq, avg_proc_t = {}, {}
    for mc in mc_name_list:
        mc_freq[mc] = {'ug': 0.0, 'nug': 0.0}
        avg_proc_t[mc] = {'ug': 0.0, 'nug': 0.0}
    for mc in mc_name_list:
        for p in range(NUM_PATTERN):
            if p < 12:
                mc_freq[mc]['nug'] += (pattern_freq[p] * oper_sqc[p].count(mc))
            else:
                mc_freq[mc]['ug'] += (pattern_freq[p] * oper_sqc[p].count(mc))
        avg_proc_t[mc]['nug'] = min(10800.0, util * (24 * 60 * 60) / (mc_freq[mc]['nug'] + proc_t_ratio * mc_freq[mc]['ug']))
        avg_proc_t[mc]['ug'] = proc_t_ratio * avg_proc_t[mc]['nug']
    proc_t = defaultdict(dict)
    for p in range(NUM_PATTERN):
        for mc in oper_sqc[p]:
            if p < 12:
                proc_t[p][mc] = avg_proc_t[mc]['nug']
            else:
                proc_t[p][mc] = avg_proc_t[mc]['ug']
    # proc_t = {0: {'A': 6, 'B': 4, 'I': 5},
    #           1: {'A': 10, 'B': 10, 'D': 10, 'I': 10}, ...}
    return oper_sqc, proc_t, pattern_freq, pattern_prob


def main():
    # # 결과파일 출력 주소 지정
    # log_dir = "./logs/{}-{}/".format('rule_name', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # os.makedirs(log_dir)

    env = simpy.Environment()
    mc_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']
    gd = GraphicDisplay(env, mc_name_list)

    machine = {}
    for mc in mc_name_list:
        machine[mc] = simpy.Resource(env, capacity=1)

    oper_sqc, proc_t, pattern_freq, pattern_prob = load_job_features(mc_name_list)
    seed = 1
    LoS, Tard = [0.0 for n in range(ITERATION)], [0.0 for n in range(ITERATION)]
    for n in range(ITERATION):
        env = simpy.Environment()
        machine = {}
        for mc in mc_name_list:
            machine[mc] = simpy.Resource(env, capacity=1)
        jobs = []
        arrt = 0
        np.random.seed(seed=seed)
        for id in range(NUM_JOBS):
            interval = np.random.exponential((24 * 60 * 60) / sum(pattern_freq[k] for k in pattern_freq.keys()))
            job_pattern = np.random.choice(NUM_PATTERN, 1, p=pattern_prob)[0]  # list로 반환해서 [0] 붙여줌
            arrt += interval
            jobs.append(Job(env, id, machine, arrt, job_pattern, oper_sqc, proc_t))
        while env.peek() < float("inf"):  # for i in range(1, 300): env.run(until=i)
            env.step()
            # GUI sentence. e.g., progressbar.update(i)
            status = {'mc_users': {}, 'mc_queue': {}}
            for mc in mc_name_list:
                status['mc_users'][mc] = machine[mc].users[0].obj if len(machine[mc].users) > 0 else 'empty'
            for mc in mc_name_list:
                if len(machine[mc].queue) > 0:
                    status['mc_queue'][mc] = {}
                    for job in range(len(machine[mc].queue)):
                        status['mc_queue'][mc][job] = machine[mc].queue[job].obj
                else:
                    status['mc_queue'][mc] = 'empty'
            gd.save_status(n, env.now, status)
        print("Simulation [%d] Complete" % n)
        for id in range(NUM_JOBS):
            LoS[n] += jobs[id].LoS
            Tard[n] += jobs[id].tardiness
        LoS[n] = round(LoS[n] / 60.0, 2)  # minute으로 변환
        Tard[n] = round(Tard[n] / 60.0, 2)
        gd.event_cnt = 0
        seed += 10
    print("All Simulations Complete")
    print("LoS_sum(min): ", LoS)
    print("LoS/patient(min): ", np.array(LoS) / NUM_JOBS)
    print("Tard_sum (min): ", Tard)
    print("Tard/pateint (min): ", np.array(Tard) / NUM_JOBS)

    # f = open("%s%s.csv" % (log_dir, 'filename'), "w")  # file_name.csv 파일 만들기
    gd.run_reset()
    gd.mainloop()


class Job(object):
    def __init__(self, env, id, machine, arrt, job_pattern, oper_sqc, proc_t):
        self.env = env
        self.id = id
        self.machine = machine
        self.pattern = job_pattern
        self.oper_sqc = oper_sqc[self.pattern][:]  # ['A', 'B', 'C']
        self.next_oper = self.oper_sqc[0]
        self.proc_t = proc_t[self.pattern]
        self.arrt = arrt
        self.waiting_t = 0.0
        if self.pattern < 12:
            self.severity = 'nug'
        else:
            self.severity = 'ug'
        self.due_rate = {'nug': 5, 'ug': 2}
        self.due_time = self.arrt + self.due_rate[self.severity] * sum(self.proc_t[oper] for oper in oper_sqc[self.pattern][:])
        self.env.process(self.run_arrival(arrt))

    def job_select(self, mc_name, rule='SPT'):
        if rule == 'SPT':
            if len(self.machine[mc_name].queue) >= 2:
                best_value = 999
                best_qloc = 0
                qloc = 0
                for user_req in self.machine[mc_name].queue:
                    if user_req.obj.proc_t[mc_name] < best_value:
                        best_value = user_req.obj.id
                        best_qloc = qloc
                        qloc += 1
                temp = self.machine[mc_name].queue[0]
                self.machine[mc_name].queue[0] = self.machine[mc_name].queue[best_qloc]
                self.machine[mc_name].queue[best_qloc] = temp
        elif rule == 'LPT':
            if len(self.machine[mc_name].queue) >= 2:
                best_value = 0
                best_qloc = 0
                qloc = 0
                for user_req in self.machine[mc_name].queue:
                    if user_req.obj.proc_t[mc_name] > best_value:
                        best_value = user_req.obj.id
                        best_qloc = qloc
                        qloc += 1
                temp = self.machine[mc_name].queue[0]
                self.machine[mc_name].queue[0] = self.machine[mc_name].queue[best_qloc]
                self.machine[mc_name].queue[best_qloc] = temp
        elif rule == "reinforcement_learning":
            pass

    def run_arrival(self, arrt):
        yield self.env.timeout(arrt)
        self.next_oper = self.oper_sqc[0]
        self.env.process(self.run_mc(self.next_oper))

    def run_mc(self, mc_name):
        mc_arrt = self.env.now
        # print('[%s] pattern %d (id:%d) arrives at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
        with self.machine[mc_name].request() as req:
            req.obj = self
            yield req
            self.waiting_t += self.env.now - mc_arrt
            # print('[%s] pattern %d (id:%d) starting at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            yield self.env.timeout(np.random.exponential(self.proc_t[mc_name]))
            # print('[%s] pattern %d (id:%d) leaving at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            self.job_select(mc_name, rule='SPT')
        del (self.oper_sqc[0])
        if len(self.oper_sqc) > 0:
            self.next_oper = self.oper_sqc[0]
            self.env.process(self.run_mc(self.next_oper))
        else:
            self.LoS = self.env.now - self.arrt
            self.tardiness = max(0, self.env.now - self.due_time)


if __name__ == '__main__':
    main()
