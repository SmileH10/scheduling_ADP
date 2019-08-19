import simpy
import random
from simulation_GUI import GraphicDisplay

ITERATION = 3


def main():
    env = simpy.Environment()
    mc_name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']
    gd = GraphicDisplay(env, mc_name_list)

    machine = {}
    for mc in mc_name_list:
        machine[mc] = simpy.Resource(env, capacity=1)

    arrt = 0
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
                13: ['A', 'B', 'C', 'D', 'E', 'I']
                }
    proc_t = {0: {'A': 6, 'B': 4, 'I': 5},
              1: {'A': 10, 'B': 10, 'D': 10, 'I': 10},
              2: {'A': 10, 'B': 10, 'E': 15, 'I': 10}
              }

    for n in range(ITERATION):
        env = simpy.Environment()
        machine = {}
        for mc in mc_name_list:
            machine[mc] = simpy.Resource(env, capacity=1)
        for id in range(10):
            arrt += 1
            Job(env, id, machine, arrt, oper_sqc, proc_t)
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
        gd.event_cnt = 0
    print("All Simulations Complete")
    gd.run_reset()
    gd.mainloop()


class Job(object):
    def __init__(self, env, id, machine, arrt, oper_sqc, proc_t):
        self.env = env
        self.id = id
        self.machine = machine
        self.pattern = random.randint(0, 2)
        self.oper_sqc = oper_sqc[self.pattern][:]  # ['A', 'B', 'C']
        self.next_oper = self.oper_sqc[0]
        self.proc_t = proc_t[self.pattern]
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

    def run_arrival(self, arrt):
        yield self.env.timeout(arrt)
        self.next_oper = self.oper_sqc[0]
        self.env.process(self.run_mc(self.next_oper))

    def run_mc(self, mc_name):
        print('[%s] pattern %d (id:%d) arrives at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
        with self.machine[mc_name].request() as req:
            req.obj = self
            yield req
            print('[%s] pattern %d (id:%d) starting at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            yield self.env.timeout(self.proc_t[mc_name])
            print('[%s] pattern %d (id:%d) leaving at t = %.2f' % (mc_name, self.pattern, self.id, self.env.now))
            self.job_select(mc_name, rule='SPT')
        del (self.oper_sqc[0])
        if len(self.oper_sqc) > 0:
            self.next_oper = self.oper_sqc[0]
            self.env.process(self.run_mc(self.next_oper))


if __name__ == '__main__':
    main()
