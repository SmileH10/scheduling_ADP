import numpy as np


class Job(object):
    def __init__(self, env, qnet, rule, id, machine, arrt, job_ptrn, oper_sqc, proc_t):
        self.rule = rule
        self.env = env
        self.qnet = qnet
        self.id = id
        self.machine = machine
        self.pattern = job_ptrn
        self.oper_sqc = oper_sqc[:]  # ['A', 'B', 'C', 'END']
        self.next_oper = self.oper_sqc[0]
        self.proc_t = proc_t[:]  # [754.44, 754.44, 193.22]
        self.arrt = arrt
        self.waiting_t = 0.0
        if self.pattern < 12:
            self.severity = 'nug'
        else:
            self.severity = 'ug'
        self.due_rate = {'nug': 5, 'ug': 2}
        self.due_time = self.arrt + self.due_rate[self.severity] * sum(self.proc_t[oper_num] for oper_num in range(len(self.oper_sqc) - 1))
        self.env.process(self.run_arrival(arrt))

    def job_select(self, mc_name, rule='SPT'):
        if rule == "RL":
            if len(self.machine[mc_name].queue) >= 2:
                state = self.qnet.get_state()
                action_list = self.qnet.get_action_list(mc_name=mc_name)  # [[ptrn1, oper_num1], [ptrn2, oper_num2], ..., [ptrn_#inque, oper_num_#inque]]
                best_value = float("inf")
                best_action = "None"
                for action in action_list:
                    val = self.qnet.get_qvalue(state, action)
                    if val < best_value:
                        best_value = val
                        best_action = action_list.index(action)  # 이거 수정해야
                assert best_action != "None"
                best_qloc = "None"
                longest_arrt = float("inf")
                for qjob in self.machine[mc_name].queue:
                    if (qjob.obj.pattern, qjob.obj.oper_num) == best_action:
                        if qjob.obj.arrt < longest_arrt:
                            best_qloc = self.machine[mc_name].queue.index(qjob)
                            longest_arrt = qjob.obj.arrt
                assert best_qloc != "None"
                temp = self.machine[mc_name].queue[0]
                self.machine[mc_name].queue[0] = self.machine[mc_name].queue[best_qloc]
                self.machine[mc_name].queue[best_qloc] = temp

                self.qnet.save_state_action(state, best_action, self.env.now)

        elif rule == 'SPT':
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

    def run_arrival(self, arrt):
        yield self.env.timeout(arrt)
        self.next_oper = self.oper_sqc[0]
        self.oper_num = 0
        self.env.process(self.run_mc(self.next_oper))

    def run_mc(self, mc_name):
        mc_arrt = self.env.now
        self.qnet.num_ptrn_que[self.pattern][self.oper_num] += 1
        # print('[%s] pattern %d (id:%d) arrives at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
        with self.machine[mc_name].request() as req:
            req.obj = self
            yield req
            self.waiting_t += self.env.now - mc_arrt
            self.qnet.num_ptrn_que[self.pattern][self.oper_num] -= 1
            self.qnet.num_ptrn_res[self.pattern][self.oper_num] += 1
            self.oper_start_t = self.env.now
            # print('[%s] pattern %d (id:%d) starting at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            yield self.env.timeout(np.random.exponential(self.proc_t[self.oper_num]))
            # print('[%s] pattern %d (id:%d) leaving at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            self.qnet.num_ptrn_res[self.pattern][self.oper_num] -= 1
            self.qnet.num_ptrn_que[self.pattern][self.oper_num + 1] += 1

            self.job_select(mc_name, rule=self.rule)
        del (self.oper_sqc[0])
        if len(self.oper_sqc) >= 2:
            self.next_oper = self.oper_sqc[0]
            self.oper_num += 1
            self.env.process(self.run_mc(self.next_oper))
        else:
            self.env.process(self.run_exit())

    def run_exit(self):
        self.LoS = self.env.now - self.arrt
        self.tardiness = max(0, self.env.now - self.due_time)