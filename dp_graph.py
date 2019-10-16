import numpy as np
from collections import OrderedDict


class DisjunctivePatternGraph(object):
    def __init__(self, args, ptrn_info, mc_info):
        self.n_x = OrderedDict()
        self.e_cj = {}
        self.e_dj = {}
        # [status(0:작업중, 1:예약, 2:대기중, 3:종료노드) + proct(4:평균/5:remaining)]
        self.n_features = ['srvd', 'rsvd', 'wait', 'proct']
        self.make_initial_graph(ptrn_info)
        self.set_st_nodes()
        util = {mc: args['util_id'] for mc in mc_info['name']}
        self.proct = self.make_initial_proct(ptrn_info, mc_info, util, scale=0.5, proct_instance=args['sce_id'])

    def set_st_nodes(self):
        self.n_x['S'] = [1, 0]
        self.n_x['T'] = [0, 1]

    def make_initial_graph(self, ptrn_info):
        # load pattern_info
        ptrn_name = ptrn_info['name']
        ptrn_prob = ptrn_info['prob']
        # node의 key 생성
        temp_e_cj_keys = []
        for ptrn in ptrn_name:
            n_key = ""
            for mc in ptrn:
                next_nkey = n_key + mc
                if n_key not in self.n_x.keys() and n_key != "":
                    self.n_x[n_key] = OrderedDict({att: 0 for att in self.n_features})
                if not (n_key, next_nkey) in temp_e_cj_keys:
                    if mc == 'T':
                        temp_e_cj_keys.append((n_key, 'T'))
                    else:
                        temp_e_cj_keys.append((n_key, next_nkey))
                n_key = next_nkey
        # edges_conjunctive 생성
        for n in self.n_x.keys():  # "AB"
            next_ns = [e[1] for e in temp_e_cj_keys if e[0] == n]  # "ABC" "ABD"(="ABDT" + "ABDCT")
            prob_nn = {n_next: sum(ptrn_prob[ptrn] for ptrn in ptrn_name if n_next in ptrn) for n_next in next_ns}
            total_prob = sum(prob_nn[n_next] for n_next in next_ns)
            for n_next in next_ns:
                prob_nn[n_next] /= total_prob
                self.e_cj[(n, n_next)] = prob_nn[n_next]
        # edges_disjunctive 생성
        for n in self.n_x.keys():
            self.e_dj[n] = [n2 for n2 in self.n_x.keys() if n2[-1] == n[-1]]

    def make_initial_proct(self, ptrn_info, mc_info, util, scale, proct_instance):    # 평균 근처에서 다들 비슷하게 하는 건 별로..
        # self.proct[n] 만들기
        np.random.seed(814 * (1 + proct_instance))
        proct = {}
        for mc in mc_info['name']:
            n_with_same_mc = [n for n in self.n_x.keys() if n[-1] == mc]
            mc_freq = sum(ptrn_info['freq'][ptrn] * ptrn.count(mc) for ptrn in ptrn_info['name'])
            mc_proct_mean = min(10800.0, util[mc] * (24 * 60 * 60) / mc_freq)
            while True:
                for n in n_with_same_mc:
                    proct[n] = round(np.random.uniform(mc_proct_mean - scale * mc_proct_mean, mc_proct_mean + scale * mc_proct_mean))
                if sum(proct[n] * sum(ptrn_info['freq'][ptrn] for ptrn in ptrn_info['name'] if n in ptrn) for n in n_with_same_mc)\
                        / (24 * 60 * 60) < min(0.95, util[mc] + 0.1):
                    if mc_proct_mean == 10800:
                        break
                    elif sum(proct[n] * sum(ptrn_info['freq'][ptrn] for ptrn in ptrn_info['name'] if n in ptrn) for n in n_with_same_mc)\
                            / (24 * 60 * 60) > util[mc] - 0.1:
                        break
            for n in n_with_same_mc:
                self.n_x[n]['proct'] = proct[n]
            # print('mc: %s / ' % mc, ["%.1f" % proct[n] for n in n_with_same_mc])
        return proct
