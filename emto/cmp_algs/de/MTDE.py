import numpy as np
from copy import deepcopy


class MTDE_Base:
    def __init__(self, env):
        '''
        Implementation of multitask differential evolution with base vector transfer.
        Ref: Jin, C., Tsai, P. W., & Qin, A. K. (2019, June). A study on knowledge reuse strategies in multitasking
        differential evolution. In 2019 IEEE Congress on Evolutionary Computation (CEC) (pp. 1564-1571). IEEE.
        Parameters
        ----------
        env
        '''
        self.env = env
        self.tasks = None
        self.s = 10   # maximal archive size
        self.p = 0.1  # knowledge utilization probability

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))
        self.archive = [[]] * self.env.n_tasks
        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            for task_id in range(self.env.n_tasks):
                cur_pop = self.env.ev_procs[task_id].cur_pop
                offspring_pop = self.env.ev_procs[task_id].sample()
                other_task_ids = self.task_ids[:task_id] + self.task_ids[task_id + 1:]
                transfer_gbx, _ = self.env.ev_procs[task_id].memory.get('gbest')

                # transfer the current best solution of to archives of other tasks
                for other_task_id in other_task_ids:
                    if len(self.archive[other_task_id]) >= self.s:
                        self.archive[other_task_id].pop(0)
                    self.archive[other_task_id].append(transfer_gbx)

                cur_archive_size = len(self.archive[task_id])
                for i in range(self.env.pop_size_per_task):
                    if np.random.rand() <= self.p and cur_archive_size > 0:
                        mask_binocr = np.random.binomial(1, self.env.ev_procs[task_id]._cr, self.env.n_dims)
                        mask_binocr[np.random.randint(self.env.n_dims)] = 1
                        ra_ = np.random.randint(cur_archive_size)
                        rt_ = np.random.permutation(self.env.pop_size_per_task)
                        offspring_pop[i] = mask_binocr * (self.archive[task_id][ra_] +
                                                          self.env.ev_procs[task_id]._f * (cur_pop[rt_[0]] - cur_pop[rt_[1]])) + \
                                           (1 - mask_binocr) * self.env.ev_procs[task_id].cur_pop[i]

                offspring_pop = np.clip(offspring_pop, self.env.xlb, self.env.xub)
                offspring_fit = self.env.tasks[task_id](offspring_pop)
                self.nfe += offspring_pop.shape[0]
                self.env.ev_procs[task_id].update(offspring_pop, offspring_fit)

        y_trajectory = []
        y_final = np.array([np.min(task._y_trajectory[:self.task_max_nfe]) - task.f.fopt for task in self.env.tasks])
        for task_id in range(self.env.n_tasks):
            task_y_trajectory = np.array(self.env.tasks[task_id]._y_trajectory) - self.env.tasks[task_id].f.fopt
            for i in range(1, len(task_y_trajectory)):
                task_y_trajectory[i] = np.min([task_y_trajectory[i], task_y_trajectory[i - 1]])
            y_trajectory.append(task_y_trajectory[:self.task_max_nfe:self.rec_nfe])
        y_trajectory = np.array(y_trajectory).T
        self.clean()
        return y_final, y_trajectory, env_info

    def clean(self):
        self.tasks = None


if __name__ == '__main__':
    import emto
    import gymnasium as gym

    env = gym.make('l2t_emto-v6', problem_name='bbob-v1')
    algo = MTDE_Base(env)
    for env_seed in range(100):
        algo.run(env_seed)