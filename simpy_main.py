import simpy
from simulation_GUI import GraphicDisplay
import numpy as np  # 나중에 random seed 설정할 때, proc_t 확률적으로 쓸 때
from collections import defaultdict
from qnet import QNet
from job import Job
from datetime import datetime
import os


ITERATION = 3
NUM_JOBS = 100
NUM_PATTERN = 5
assert NUM_PATTERN <= 24


def load_job_features(mc_name_list):
    mc_sqc = {  # 비응급환자
        0: ['A', 'B', 'I', 'END'],
        1: ['A', 'B', 'D', 'I', 'END'],
        2: ['A', 'B', 'E', 'I', 'END'],
        3: ['A', 'B', 'C', 'D', 'I', 'END'],
        4: ['A', 'B', 'C', 'J', 'END'],
        5: ['A', 'B', 'D', 'C', 'J', 'END'],
        6: ['A', 'B', 'C', 'I', 'END'],
        7: ['A', 'B', 'D', 'C', 'I', 'END'],
        8: ['A', 'B', 'F', 'I', 'END'],
        9: ['A', 'B', 'C', 'D', 'J', 'END'],
        10: ['A', 'B', 'C', 'E', 'I', 'END'],
        11: ['A', 'B', 'E', 'C', 'J', 'END'],
        # 응급환자
        12: ['A', 'B', 'C', 'D', 'E', 'J', 'END'],
        13: ['A', 'B', 'C', 'D', 'E', 'I', 'END'],
        14: ['A', 'B', 'C', 'D', 'I', 'END'],  # ABCDI 3번과 같음
        15: ['A', 'B', 'E', 'C', 'D', 'J', 'END'],
        16: ['A', 'B', 'C', 'E', 'D', 'J', 'END'],
        17: ['A', 'B', 'C', 'E', 'D', 'I', 'END'],
        18: ['A', 'B', 'E', 'C', 'D', 'I', 'END'],
        19: ['A', 'B', 'C', 'D', 'E', 'D', 'J', 'END'],
        20: ['A', 'B', 'C', 'D', 'F', 'E', 'I', 'END'],
        21: ['A', 'B', 'C', 'D', 'F', 'E', 'J', 'END'],
        22: ['A', 'B', 'C', 'D', 'G', 'E', 'J', 'END'],
        23: ['A', 'B', 'D', 'C', 'E', 'J', 'END']
    }
    pattern_freq = {0: 2699, 1: 1390, 2: 876, 3: 694, 4: 358, 5: 292,
                6: 288, 7: 228, 8: 210, 9: 114, 10: 74, 11: 62,
                12: 885, 13: 523, 14: 256, 15: 206, 16: 197, 17: 116,
                18: 98, 19: 76, 20: 63, 21: 46, 22: 45, 23: 44}
    pattern_freq = {k: pattern_freq[k] for k in pattern_freq.keys() if k < NUM_PATTERN}
    for k in pattern_freq.keys():
        pattern_freq[k] /= 61.0

    pattern_prob = []
    sum_pattern_freq = sum(pattern_freq[k] for k in pattern_freq.keys())
    for k in range(NUM_PATTERN):
        pattern_prob.append(pattern_freq[k] / sum_pattern_freq)

    util = 0.85
    proc_t_ratio = 2  # 응급/비응급환자의 process time 비율 (>=1)
    assert proc_t_ratio >= 1
    # dictionary initialize
    mc_freq, avg_proc_t = {}, {}
    for mc in mc_name_list:
        mc_freq[mc] = {'ug': 0.0, 'nug': 0.0}
        avg_proc_t[mc] = {'ug': 0.0, 'nug': 0.0}
    for mc in mc_name_list:
        for ptrn in range(NUM_PATTERN):
            if ptrn < 12:
                mc_freq[mc]['nug'] += (pattern_freq[ptrn] * mc_sqc[ptrn].count(mc))
            else:
                mc_freq[mc]['ug'] += (pattern_freq[ptrn] * mc_sqc[ptrn].count(mc))
        if mc_freq[mc]['nug'] > 0:
            avg_proc_t[mc]['nug'] = min(10800.0, util * (24 * 60 * 60) / (mc_freq[mc]['nug'] + proc_t_ratio * mc_freq[mc]['ug']))
        else:
            avg_proc_t[mc]['nug'] = 0
        avg_proc_t[mc]['ug'] = proc_t_ratio * avg_proc_t[mc]['nug']

    proc_t = defaultdict(list)
    for ptrn in range(NUM_PATTERN):
        for oper_num in range(len(mc_sqc[ptrn]) - 1):
            mc = mc_sqc[ptrn][oper_num]
            if ptrn < 12:
                proc_t[ptrn].append(avg_proc_t[mc]['nug'])
            else:
                proc_t[ptrn].append(avg_proc_t[mc]['ug'])
    # proc_t = {0: {'A': 6, 'B': 4, 'I': 5},
    #           1: {'A': 10, 'B': 10, 'D': 10, 'I': 10}, ...}
    return mc_sqc, proc_t, pattern_freq, pattern_prob


def main(gd=False):
    # # 결과파일 출력 주소 지정
    # log_dir = "./logs/{}-{}/".format('rule_name', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # os.makedirs(log_dir)

    env = simpy.Environment()
    mc_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']
    machine = {}
    for mc in mc_name_list:
        machine[mc] = simpy.Resource(env, capacity=1)
    mc_sqc, proc_t, pattern_freq, pattern_prob = load_job_features(mc_name_list)

    if gd:
        gd = GraphicDisplay(env, mc_name_list)
    qnet = QNet(env, mc_name_list, mc_sqc, proc_t, NUM_PATTERN)

    seed = 1
    LoS, Tard = [0.0 for n in range(ITERATION)], [0.0 for n in range(ITERATION)]

    for n in range(ITERATION):
        env = simpy.Environment()
        machine = {}
        for mc in mc_name_list:
            machine[mc] = simpy.Resource(env, capacity=1)
        qnet.machine = machine
        jobs = []
        arrt = 0
        np.random.seed(seed=seed)
        for id in range(NUM_JOBS):
            interval = np.random.exponential((24 * 60 * 60) / sum(pattern_freq[k] for k in pattern_freq.keys()))
            job_ptrn = np.random.choice(NUM_PATTERN, 1, p=pattern_prob)[0]  # list로 반환해서 [0] 붙여줌
            jobs.append(Job(env, qnet, 'RL', id, machine, arrt, job_ptrn, mc_sqc[job_ptrn], proc_t[job_ptrn]))
            arrt += interval
        while env.peek() < float("inf"):  # for i in range(1, 300): env.run(until=i)
            env.step()
            # GUI sentence. e.g., progressbar.update(i)
            if gd:
                gd_status = {'mc_users': {}, 'mc_queue': {}}
                for mc in mc_name_list:
                    gd_status['mc_users'][mc] = machine[mc].users[0].obj if len(machine[mc].users) > 0 else 'empty'
                for mc in mc_name_list:
                    if len(machine[mc].queue) > 0:
                        gd_status['mc_queue'][mc] = {}
                        for job in range(len(machine[mc].queue)):
                            gd_status['mc_queue'][mc][job] = machine[mc].queue[job].obj
                    else:
                        gd_status['mc_queue'][mc] = 'empty'
                gd.save_status(n, env.now, gd_status)
        print("Simulation [%d] Complete" % n)
        # 시뮬레이션 1회 결과 기록
        for id in range(NUM_JOBS):
            LoS[n] += jobs[id].LoS
            Tard[n] += jobs[id].tardiness
        LoS[n] = round(LoS[n] / 60.0, 2)  # minute으로 변환
        Tard[n] = round(Tard[n] / 60.0, 2)
        qnet.update_nstpe_memory(LoS[n], env.now, step=1)
        qnet.train_model()
        if gd:
            gd.event_cnt = 0
        seed += 10

    print("All Simulations Complete")
    print("LoS_sum(min): ", LoS)
    print("LoS/patient(min): ", np.array(LoS) / NUM_JOBS)
    print("Tard_sum (min): ", Tard)
    print("Tard/pateint (min): ", np.array(Tard) / NUM_JOBS)

    # f = open("%s%s.csv" % (log_dir, 'filename'), "w")  # file_name.csv 파일 만들기
    if gd:
        gd.run_reset()
        gd.mainloop()


if __name__ == '__main__':
    main()
