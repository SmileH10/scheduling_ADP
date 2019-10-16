import simpy
# from simulation_GUI import GraphicDisplay
# from sim_ptrnmapGUI import PtrnGraphicDisplay
import numpy as np  # 나중에 random seed 설정할 때, proc_t 확률적으로 쓸 때
# from qnet_ptrnmap import QNet
from qnet_ver1 import QNet
from job import Job
from dp_graph import DisjunctivePatternGraph
from time import time
from datetime import datetime
import os


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


def sim_setup(args, simenv, sim_mcrsc, ptrn_info, mc_info, qnet, g, test):
    jobs = []
    jobid = 0
    args['report']['num_job_in_system'] = 0
    if test:
        sim_seed = args['test_id']
    else:
        sim_seed = 0
    while True:
        jobs.append(Job(jobid, args, simenv, sim_mcrsc, ptrn_info, mc_info, g, test))
        if args['prior_rule'] == 'RL':
            jobs[-1].qnet = qnet
        # interval time of arrival
        sim_seed += 100
        np.random.seed(seed=sim_seed)
        yield simenv.timeout(np.random.exponential((24 * 60 * 60) / sum(ptrn_info['freq'][ptrn] for ptrn in ptrn_info['name'])))
        jobid += 1


def main(args, test=False, model=False):
    # 결과출력 dir 정하기
    if not test:  # args['prior_rule'] == 'RL' 인 경우에만 생길 수 있음.
        args['dir']['qnet_model_dir'] = args['dir']['instance_dir'] + "trained_model/"
        args['dir']['wt_fig_dir'] = args['dir']['instance_dir'] + "RLtrain_fig_waiting_time/"
        args['dir']['loss_fig_dir'] = args['dir']['instance_dir'] + "RLtrain_fig_model_loss/"
    else:
        args['dir']['test_dir'] = args['dir']['instance_dir'] + 'test\\test%d\\' % (args['test_id'])
        if args['prior_rule'] == 'RL':
            args['dir']['test_epoch_dir'] = args['dir']['test_dir'] + 'test_epoch_%s\\' % (model['load_epoch'])  #

    # directory 생성
    for key in args['dir'].keys():
        if not os.path.exists(args['dir'][key]):
            os.makedirs(args['dir'][key])
        else:
            if key == 'instance_dir' and not test:
                raise Exception("INSTANCE DIR ALREADY EXISTS")
            # else:
            #     print("WARNING: " + str(key) + " ALREADY EXISTS")

    # configuration 기록
    if not test:
        with open(args['dir']['instance_dir'] + 'config.txt', 'a') as file:
            file.write("time_log: {}\n".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            for key in args.keys():
                if key != 'dir' and key != 'report':
                    file.write("{}: {}\n".format(key, args[key]))
    else:
        with open(args['dir']['test_dir'] + 'config.txt', 'a') as file:
            file.write("time_log: {}\n".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            for key in args.keys():
                if key != 'dir' and key != 'report':
                    file.write("{}: {}\n".format(key, args[key]))

    # load data: ptrn/mc info
    mc_info = {'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']}
    ptrn_info = load_job_features()

    # 결과 기록 변수 생성
    args['report'] = {'WT': [], 'WT_warmup': [], 'best_alltime_wt': float("inf"), 'test_wt_raw': []}

    # 알고리즘 초기값 설정
    # Iteration 초기값 설정
    simenv = simpy.Environment()
    sim_mcrsc = {mc: simpy.Resource(simenv, capacity=1) for mc in mc_info['name']}

    # Load classes
    g = DisjunctivePatternGraph(args, ptrn_info, mc_info)
    if args['prior_rule'] == 'RL':
        qnet = QNet(args, ptrn_info, mc_info, g, simenv, model=model, test=test)  # qnet part 나중에 확인##########################
    else:
        qnet = False

    simenv.process(sim_setup(args, simenv, sim_mcrsc, ptrn_info, mc_info, qnet, g, test=test))

    while simenv.peek() < (args['WARMUP_DAYS'] + args['TEST_DAYS']) * (24*60*60):  # 42 * (24*60*60):  # while simenv.peek() < float("inf"): # for i in range(1, 300): env.run(until=i)
        simenv.step()
    print("Simulation Complete")
    return args


def write_data(args, data, opt):
    if opt == 'summary':
        f = open("%s%s.csv" % (args['dir']['instance_dir'] + 'test\\', 'summary.csv'), "w")
        f.write("test_id, avg_WT\n")
        for key_testid in data.keys():
            f.write("%d, %.1f,\n" % (key_testid, data[key_testid]['avg_WT']))
        f.write("AVG, %.1f" % (sum(data[key_testid]['avg_WT'] for key_testid in data.keys()) / len(list(data.keys()))))
        f.close()
    elif opt == 'epoch':
        f = open("%s%s.csv" % (args['dir']['instance_dir'] + 'test\\', 'epoch.csv'), "w")
        f.write("test_id, epoch, avg_WT\n")
        for key_testid in data.keys():
            for key in data[key_testid].keys():
                if key[0] == 'e':
                    f.write('%d, %s, %.1f,\n' % (key_testid, key, data[key_testid][key]['avg_WT']))
        f.close()
        f = open("%s%s.csv" % (args['dir']['instance_dir'] + 'test\\', 'epoch_summary.csv'), "w")
        f.write("epoch, avg_WT\n")
        sumval = 0.0
        cnt = 0
        for key in data[0].keys():
            if key[0] == 'e':
                avg = sum(data[key_testid][key]['avg_WT'] for key_testid in data.keys()) / len(list(data.keys()))
                f.write("%s, %.1f,\n" % (key, avg))
                sumval += avg
                cnt += 1
        f.write("AVG, %.1f" % (sumval/cnt))
        f.close()
    elif opt == 'detail':
        f = open("%s%s.csv" % (args['dir']['test_dir'], 'wt_detail.csv'), "w")
        f.write("wt_raw,\n")
        for i in range(len(data[args['test_id']]['test_wt_raw'])):
            f.write("%.1f,\n" % data[args['test_id']]['test_wt_raw'][i])
        f.close()
    elif opt == 'epoch_detail':
        f = open("%s%s.csv" % (args['dir']['test_epoch_dir'], 'wt_detail.csv'), "w")
        f.write("wt_raw,\n")
        if args['name'] == 'best':
            for i in range(len(data[args['test_id']]['test_wt_raw'])):
                f.write("%.1f,\n" % data[args['test_id']]['test_wt_raw'][i])
        else:
            for i in range(len(data[args['test_id']][args['name']]['test_wt_raw'])):
                f.write("%.1f,\n" % data[args['test_id']][args['name']]['test_wt_raw'][i])
        f.close()


if __name__ == '__main__':
    args = {'prior_rule': 'RL',
            'WARMUP_DAYS': 0.5,
            'TEST_DAYS': 0.1,
            'MEMORY_SIZE': 2000,
            'NSTEP': 5,
            'BATCH_SIZE': 64,
            'dir': {}
            }
    print("RULE: ", args['prior_rule'])

    avg_util_list = [0.6, 0.75, 0.9]
    avg_util_list = [0.75]
    num_scenario = 1  # avg_util = 0.xx 가 주어졌을 때, 몇 가지 proct variation을 만들 것인가?
    num_test = 1  # 28일 wamrup + 28일 결과기록 test: 몇 번 반복할 것인가?

    for util_id in avg_util_list:
        for sce_id in range(num_scenario):
            print("INSTANCE UTIL: %.2f, SCENARIO: %d" % (util_id, sce_id))
            # instance 생성
            args['util_id'] = util_id
            args['sce_id'] = sce_id
            args['dir']['instance_dir'] = "./logs/{}-util{}-sce{}/".format(args['prior_rule'], args['util_id'], args['sce_id'])
            # if args['prior_rule'] == 'RL':
            #     # RL은 학습 필요
            #     args_return = main(args=args, test=False, model=False)
            #     print("TRAINING ENDS. TEST STARTS...")
            # # TEST
            data = {}
            # RL인 경우
            if args['prior_rule'] == 'RL':
                model = {'load_epoch': 'best'}
                model['model_loaded'] = args['dir']['instance_dir'] + 'trained_model\\best_alltime_wt_model.h5'
                # print("EPOCH BEST STARTS...")
                for test_id in range(num_test + 1):
                    args['test_id'] = test_id
                    # print("TEST %d STARTS..." % test_id)
                    args_return = main(args=args, model=model, test=True)
                    data[test_id] = {'avg_WT': args_return['report']['WT_warmup'][-1], 'test_wt_raw': args_return['report']['test_wt_raw']}
                    args_return['name'] = 'best'
                    args_return['test_id'] = test_id
                    write_data(args_return, data, opt='epoch_detail')
                i = 1
                while True:
                    model = {'load_epoch': 50 * i}
                    model['model_loaded'] = args['dir']['instance_dir'] + 'trained_model\model_epoch%d.h5' % model['load_epoch']
                    if not os.path.exists(model['model_loaded']):
                        break
                    # print("EPOCH %d STARTS..." % model['load_epoch'])
                    for test_id in range(num_test + 1):
                        args['test_id'] = test_id
                        # print("TEST %d STARTS..." % test_id)
                        args_return = main(args=args, model=model, test=True)
                        data[test_id]['epoch' + str(model['load_epoch'])] = \
                            {'avg_WT': args_return['report']['WT_warmup'][-1], 'test_wt_raw': args_return['report']['test_wt_raw']}
                        args_return['name'] = 'epoch' + str(model['load_epoch'])
                        args_return['test_id'] = test_id
                        write_data(args_return, data, opt='epoch_detail')
                    i += 1
            else:
                # 기타 다른 dispatching rules (SPT, FCFS 등)
                for test_id in range(num_test + 1):
                    args['test_id'] = test_id
                    # print("TEST %d STARTS..." % test_id)
                    args_return = main(args=args, model=False, test=True)
                    data[test_id] = {'avg_WT': args_return['report']['WT_warmup'][-1], 'test_wt_raw': args_return['report']['test_wt_raw']}
                    args_return['test_id'] = test_id
                    write_data(args_return, data, opt='detail')

            write_data(args_return, data, opt='summary')
            if args['prior_rule'] == 'RL':
                write_data(args_return, data, opt='epoch')



    # # 'C:\Users\Aidan\OneDrive - 연세대학교 (Yonsei University)\PycharmProjects_Dropbox\19ADP_EDscheduling\logs\RL-is1-2019-10-10_17-32-57_dim64_T3'
    # model['memory_name'] = 'trained_model\memory_epoch%d.json' % model['load_epoch']
    # model['instance_dir'] = r'.\logs\RL-is1-2019-10-10_17-32-57_dim64_T3\\'
    # model['memory_loaded'] = model['load_dir'] + model['memory_name']
    # # 이어서 Training 할 때
    # # else:
    # #     model['save_moretrain_dir'] = model['instance_dir'] + 'train_from_model_epoch%d\\' % model['load_epoch']
    #
    # # input: model / 문제(proct) scenario 번호(개수 말고) / test 횟수(arrival/proct/oper prob.)
