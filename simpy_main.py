import simpy
import random

data = []


def main():
    env = simpy.Environment()
    machine = {}
    machine['A'] = simpy.Resource(env, capacity=2)
    machine['B'] = simpy.Resource(env, capacity=1)
    machine['C'] = simpy.Resource(env, capacity=1)

    # jobs = []
    arrt = 0
    oper_sqc = {1: ['A', 'B', 'C'], 2: ['A', 'C']}
    proc_t = {1: {'A':6, 'B':4, 'C':5}, 2: {'A':10, 'C':10}}
    for id in range(5):
        arrt += 1
        Job(env, id, machine, arrt, oper_sqc, proc_t)
        # jobs.append(Job(env, id, machine, arrt, oper_sqc, proc_t))
    for i in range(1, 300):
        env.run(until=i)
        # GUI sentence. e.g., progressbar.update(i)
    print("Simulation Complete")


class Job(object):
    def __init__(self, env, id, machine, arrt, oper_sqc, proc_t):
        self.env = env
        self.id = id
        self.machine = machine
        self.pattern = random.randint(1, 2)
        self.oper_sqc = oper_sqc[self.pattern][:]  # ['A', 'B', 'C']
        self.next_oper = self.oper_sqc[0]
        self.proc_t = proc_t[self.pattern]
        self.run_oper = {'A': self.runA(arrt), 'B': self.runB(), 'C': self.runC()}

        self.env.process(self.run_oper[self.next_oper])

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

    def runA(self, arrt):
        yield self.env.timeout(arrt)
        print('[A] id: %d (pattern %d) arrives at t = %.2f' % (self.id, self.pattern, self.env.now), self.oper_sqc)
        with self.machine['A'].request() as req:
            req.obj = self
            yield req
            print('[A] id: %d starting at t = %.2f' % (self.id, self.env.now))
            yield self.env.timeout(self.proc_t['A'])
            print('[A] id: %d leaving at t = %.2f' % (self.id, self.env.now))
            self.job_select('A', rule='SPT')

        del (self.oper_sqc[0])
        if len(self.oper_sqc) > 0:
            self.next_oper = self.oper_sqc[0]
            self.env.process(self.run_oper[self.next_oper])

    def runB(self):
        print('[B] id: %d arrives at t = %.2f' % (self.id, self.env.now))
        with self.machine['B'].request() as req:
            req.obj = self
            yield req
            print('[B] id: %d starting at t = %.2f' % (self.id, self.env.now))
            yield self.env.timeout(self.proc_t['B'])
            print('[B] id: %d leaving at t = %.2f' % (self.id, self.env.now))
            self.job_select('B', rule='SPT')

        del (self.oper_sqc[0])
        if len(self.oper_sqc) > 0:
            self.next_oper = self.oper_sqc[0]
            self.env.process(self.run_oper[self.next_oper])

    def runC(self):
        print('[C] id: %d arrives at t = %.2f' % (self.id, self.env.now))
        with self.machine['C'].request() as req:
            req.obj = self
            yield req
            print('[C] id: %d starting at t = %.2f' % (self.id, self.env.now))
            yield self.env.timeout(self.proc_t['C'])
            print('[C] id: %d leaving at t = %.2f' % (self.id, self.env.now))
            self.job_select('C', rule='SPT')

        del (self.oper_sqc[0])
        if len(self.oper_sqc) > 0:
            self.next_oper = self.oper_sqc[0]
            self.env.process(self.run_oper[self.next_oper])

    # def run_mc(self, mc_name):
    #     print('[%s] id: %d arrives at t = %.2f' % (mc_name, self.id, self.env.now))
    #     with self.machine[mc_name].request() as req:
    #         req.obj = self
    #         yield req
    #         print('[%s] id: %d starting at t = %.2f' % (mc_name, self.id, self.env.now))
    #         yield self.env.timeout(self.proc_t[mc_name])
    #         print('[%s] id: %d leaving at t = %.2f' % (mc_name, self.id, self.env.now))
    #         self.job_select(mc_name, rule='SPT')
    #     del (self.oper_sqc[0])
    #     if len(self.oper_sqc) > 0:
    #         self.next_oper = self.oper_sqc[0]
    #         # self.env.process(self.run_oper[self.next_oper])
    #         self.env.process(self.run_mc(self.next_oper))


if __name__ == '__main__':
    main()
