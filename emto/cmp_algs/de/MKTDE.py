import numpy as np
from copy import deepcopy


class MKTDE:
    '''
        Implementation of the meta-knowledge trasnfer-based differential evolution algorithm.
        Ref: Li, J. Y., Zhan, Z. H., Tan, K. C., & Zhang, J. (2021). A meta-knowledge transfer-based
        differential evolution for multitask optimization. IEEE Transactions on Evolutionary Computation, 26(4), 719-734.
    '''
    def __init__(self, env):
        self.env = env
        self.tasks = None

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))
        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            pop_mus = [np.mean(self.env.ev_procs[task_id].cur_pop, axis=0) for task_id in range(self.env.n_tasks)]
            for task_id in range(self.env.n_tasks):
                other_task_ids = self.task_ids[:task_id] + self.task_ids[task_id + 1:]
                source_task_id = np.random.choice(other_task_ids)
                pop_f = np.r_[self.env.ev_procs[task_id].cur_pop,
                              self.env.ev_procs[source_task_id].cur_pop + pop_mus[task_id] - pop_mus[source_task_id]]
                offspring_pop = []
                for i in range(self.env.pop_size_per_task):
                    mask_binocr = np.random.binomial(1, self.env.ev_procs[task_id]._cr, self.env.n_dims)
                    mask_binocr[np.random.randint(self.env.n_dims)] = 1
                    ri_ = np.random.randint(self.env.pop_size_per_task)
                    rf_ = np.random.permutation(pop_f.shape[0])
                    offspring_pop.append(mask_binocr * (self.env.ev_procs[task_id].cur_pop[ri_] +
                                                        self.env.ev_procs[task_id]._f * (pop_f[rf_[0]] - pop_f[rf_[1]])) +
                                                        (1 - mask_binocr) * self.env.ev_procs[task_id].cur_pop[i])
                offspring_pop = np.clip(offspring_pop, self.env.xlb, self.env.xub)
                offspring_fit = self.env.tasks[task_id](offspring_pop)
                self.nfe += offspring_pop.shape[0]
                self.env.ev_procs[task_id].update(offspring_pop, offspring_fit)

                transfer_gbx, _ = self.env.ev_procs[source_task_id].memory.get('gbest')
                transfer_gbx = transfer_gbx[np.newaxis, :]
                transfer_fit = self.env.tasks[task_id](transfer_gbx)
                self.nfe += transfer_gbx.shape[0]
                self.env.ev_procs[task_id].update(transfer_gbx, transfer_fit)

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
    algo = MKTDE(env)
    for env_seed in range(100):
        algo.run(env_seed)