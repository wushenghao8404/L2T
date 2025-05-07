import numpy as np
from copy import deepcopy


class MFDE:
    '''
        Implementation of multifactorial differential evolution algorithm.
        Ref: Feng, L., Zhou, W., Zhou, L., Jiang, S. W., Zhong, J. H., Da, B. S., ... & Wang, Y. (2017, June).
        An empirical study of multifactorial PSO and multifactorial DE. In 2017 IEEE Congress on evolutionary
        computation (CEC) (pp. 921-928). IEEE.
    '''
    def __init__(self, env):
        self.env = env
        self.tasks = None
        self.rmp = 0.3

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))
        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            for task_id in range(self.env.n_tasks):
                offspring_pop = self.env.ev_procs[task_id].sample()
                other_task_ids = self.task_ids[:task_id] + self.task_ids[task_id + 1:]
                source_task_id = np.random.choice(other_task_ids)
                source_pop = self.env.ev_procs[source_task_id].cur_pop

                for i in range(self.env.pop_size_per_task):
                    if np.random.rand() <= self.rmp:
                        mask_binocr = np.random.binomial(1, self.env.ev_procs[task_id]._cr, self.env.n_dims)
                        mask_binocr[np.random.randint(self.env.n_dims)] = 1
                        ri_ = np.random.randint(self.env.pop_size_per_task)
                        rs_ = np.random.permutation(source_pop.shape[0])
                        offspring_pop[i] = + (1 - mask_binocr) * self.env.ev_procs[task_id].cur_pop[i] + \
                            mask_binocr * (self.env.ev_procs[task_id].cur_pop[ri_] +
                                           self.env.ev_procs[task_id]._f * (source_pop[rs_[0]] - source_pop[rs_[1]]))

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
    algo = MFDE(env)
    for env_seed in range(100):
        algo.run(env_seed)