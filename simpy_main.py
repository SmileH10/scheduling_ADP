import simpy
# from simulation_GUI import GraphicDisplay
# from sim_ptrnmapGUI import PtrnMapGraphicDisplay
import numpy as np  # 나중에 random seed 설정할 때, proc_t 확률적으로 쓸 때
from qnet_ptrnmap import QNet
from job import Job
from dp_graph import DisjunctivePatternGraph
from time import time
from datetime import datetime
import os


ITERATION = 1
NUM_JOBS = 100


def load_job_features():
    ptrn_name = ['SABIT', 'SABDIT', 'SABEIT', 'SABCDIT', 'SABCJT', 'SABDCJT', 'SABCIT', 'SABDCIT', 'SABFIT', 'SABCDJT', 'SABCEIT', 'SABECJT',
                 'SABCDEJT', 'SABCDEIT', 'SABECDJT', 'SABCEDJT', 'SABCEDIT', 'SABECDIT', 'SABCDEDJT', 'SABCDFEIT', 'SABCDFEJT', 'SABCDGEJT',
                 'SABDCEJT']

    # PATTERN 빈도
    temp_ptrn_freq = {0: 2699, 1: 1390, 2: 876, 3: 694 + 256, 4: 358, 5: 292, 6: 288, 7: 228, 8: 210, 9: 114, 10: 74, 11: 62,
                      12: 885, 13: 523, 14: 206, 15: 197, 16: 116, 17: 98, 18: 76, 19: 63, 20: 46, 21: 45, 22: 44}
    ptrn_freq = {ptrn_name[i]: temp_ptrn_freq[i] / 61.0 for i in range(23)}

    # PATTERN 비율 (합: 1)
    temp_sum_ptrn_freq = sum(ptrn_freq[ptrn] for ptrn in ptrn_name)
    ptrn_prob = {ptrn: ptrn_freq[ptrn] / temp_sum_ptrn_freq for ptrn in ptrn_name}

    ptrn_info = {'name': ptrn_name, 'freq': ptrn_freq, 'prob': ptrn_prob}
    return ptrn_info


def sim_setup(simenv, prior_rule, sim_mcrsc, ptrn_info, mc_info, qnet, g, fig_dir, report):
    jobs = []
    jobid = 0
    report['num_job_in_system'] = 0
    while True:
        jobs.append(Job(jobid, simenv, prior_rule, sim_mcrsc, ptrn_info, mc_info, g, fig_dir, report))
        if prior_rule == 'RL':
            jobs[-1].qnet = qnet
        # interval time of arrival
        yield simenv.timeout(np.random.exponential((24 * 60 * 60) / sum(ptrn_info['freq'][ptrn] for ptrn in ptrn_info['name'])))
        jobid += 1
        if jobid == 10000:
            print("@@@@@****************************** %.2f ******************************@@@@@" % simenv.now)
            break


def main(gd=False, prior_rule='SPT', load_ckpt=False, test=False):
    # 결과파일 출력 주소 지정
    print("RULE: ", prior_rule)
    log_dir = "./logs/{}-{}/".format(prior_rule, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir)
    fig_dir = log_dir + "output_fig/waiting_time/"
    os.makedirs(fig_dir)

    # load data: ptrn/mc info
    mc_info = {'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']}
    ptrn_info = load_job_features()

    # load classes
    g = DisjunctivePatternGraph(ptrn_info, mc_info)
    # if gd:
    #     gd = GraphicDisplay(mc_info['name'])
    if prior_rule == 'RL':
        qnet = QNet(ptrn_info, mc_info, g, log_dir, ckpt=load_ckpt, test=test)  # qnet part 나중에 확인##########################
    else:
        qnet = False

    # 결과 기록 변수 생성
    report = {'WT': [], 'WT_warmup': []}

    # 알고리즘 초기값 설정
    seed = 1
    start = time()

    # Iteration 초기값 설정
    simenv = simpy.Environment()
    sim_mcrsc = {mc: simpy.Resource(simenv, capacity=1) for mc in mc_info['name']}
    if prior_rule == 'RL':
        qnet.sim_mcrsc = sim_mcrsc  # qnet part 나중에 확인##########################

    np.random.seed(seed=seed)
    simenv.process(sim_setup(simenv, prior_rule, sim_mcrsc, ptrn_info, mc_info, qnet, g, fig_dir, report))

    while simenv.peek() < float("inf"):  # for i in range(1, 300): env.run(until=i)
        simenv.step()

        if gd:  # GUI sentence. e.g., progressbar.update(i)
            gd_status = {'mc_users': {}, 'mc_queue': {}}
            for mc in mc_info['name']:
                gd_status['mc_users'][mc] = sim_mcrsc[mc].users[0].obj if len(sim_mcrsc[mc].users) > 0 else 'empty'
            for mc in mc_info['name']:
                if len(sim_mcrsc[mc].queue) > 0:
                    gd_status['mc_queue'][mc] = {}
                    for job in range(len(sim_mcrsc[mc].queue)):
                        gd_status['mc_queue'][mc][job] = sim_mcrsc[mc].queue[job].obj
                else:
                    gd_status['mc_queue'][mc] = 'empty'
            gd.save_status(n, simenv.now, gd_status)
    print("Simulation Complete")
    # 시뮬레이션 1회 결과 기록

    # # job.py 보고나서 이 밑에서부터 다시 시작
    # for id in range(NUM_JOBS):
    #     Total_WT[n] += jobs[id].LoS
    # LoS[n] = round(LoS[n] / 60.0, 2)  # minute으로 변환
    if gd:
        gd.event_cnt = 0
    seed += 10

    # print("All Simulations Complete. 소요시간: %.2f" % (time() - start))
    # print("WT_sum(minutes): ", Total_WT)
    # print("WT/patient(minutes): ", np.array(Total_WT) / NUM_JOBS)

    # f = open("%s%s.csv" % (log_dir, 'filename'), "w")  # file_name.csv 파일 만들기
    if gd:
        gd.run_reset()
        gd.mainloop()


if __name__ == '__main__':
    ckpt = r'C:\Users\weird\OneDrive - 연세대학교 (Yonsei University)\PycharmProjects_Dropbox\19ADP_EDscheduling\logs\RL-2019-09-30_17-46-18\model\cp.ckpt'
    main(gd=False, prior_rule='RL', load_ckpt=ckpt, test=False)
