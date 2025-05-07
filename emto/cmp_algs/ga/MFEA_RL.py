import numpy as np
from copy import deepcopy
import emto.evol_operator as eo


class MFEA_RL:
    '''
        Implementation of multifactorial evolutionary algorithm with reinforcement learning
        Ref: @article{li2023evolutionary,
                      title={Evolutionary multitasking via reinforcement learning},
                      author={Li, Shuijia and Gong, Wenyin and Wang, Ling and Gu, Qiong},
                      journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
                      volume={8},
                      number={1},
                      pages={762--775},
                      year={2023},
                      publisher={IEEE}}
    '''

    def __init__(self, env):
        self.env = env
        self.tasks = None
        self.rmps = None

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))
        self.cand_rmp_list = np.arange(0.1, 0.8)
        self.q_tbl = np.zeros((self.env.n_tasks, 3, len(self.cand_rmp_list)))
        self.states = np.array([1, 1])
        self.rmps = np.random.rand(self.env.n_tasks)
        self.alpha = 0.1 # q-learning rate
        self.beta = 0.9  # discount rate
        self.f = 0.5
        self.cr = 0.6

        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])
            offspring_pops = np.full(shape=(self.env.n_tasks, self.env.pop_size_per_task, self.env.n_dims),
                                     fill_value=np.nan)
            offspring_fits = np.full(shape=(self.env.n_tasks, self.env.pop_size_per_task), fill_value=np.nan)

            if np.random.rand() <= 0.5:
                eo_ops = ["DE/rand/1", "DE/best/1"]
            else:
                eo_ops = ["DE/best/1", "DE/rand/1"]

            for task_id in range(self.env.n_tasks):
                for i in range(self.env.pop_size_per_task):
                    if eo_ops[task_id] == "DE/rand/1":
                        mutant = eo.diff_evol_rand_1_mutation(self.env.ev_procs[task_id].cur_pop, f=self.f)
                    else:
                        mutant = eo.diff_evol_best_1_mutation(self.env.ev_procs[task_id].cur_pop,
                                                              self.env.ev_procs[task_id].memory.get('gbest')[0], f=self.f)
                    child = eo.binomial_crossover(mutant, self.env.ev_procs[task_id].cur_pop[i], cr=self.cr)
                    offspring_pops[task_id, i] = deepcopy(child)

            for task_id in range(self.env.n_tasks):
                # clip the population to satisfy box constraint
                offspring_pops[task_id] = np.clip(offspring_pops[task_id], self.env.xlb, self.env.xub)

                # fitness evaluation
                offspring_fits[task_id] = self.env.tasks[task_id](offspring_pops[task_id])

                # update population by selection
                self.env.ev_procs[task_id].update(offspring_pops[task_id], offspring_fits[task_id])

            rl_offspring_pops = np.full(shape=(self.env.n_tasks, self.env.pop_size_per_task, self.env.n_dims),
                                        fill_value=np.nan)
            rl_offspring_fits = np.full(shape=(self.env.n_tasks, self.env.pop_size_per_task), fill_value=np.nan)

            for task_id in range(self.env.n_tasks):
                for i in range(self.env.pop_size_per_task):
                    r_ = np.random.permutation(self.env.pop_size_per_task)
                    if eo_ops[task_id] == "DE/rand/1":
                        p1 = self.env.ev_procs[task_id].cur_pop[r_[0]]
                    else:
                        p1 = self.env.ev_procs[task_id].memory.get('gbest')[0]
                    if np.random.rand() <= self.rmps[task_id]:
                        p2 = self.env.ev_procs[1 - task_id].cur_pop[r_[1]]
                        p3 = self.env.ev_procs[1 - task_id].cur_pop[r_[2]]
                    else:
                        p2 = self.env.ev_procs[task_id].cur_pop[r_[1]]
                        p3 = self.env.ev_procs[task_id].cur_pop[r_[2]]
                    mutant = p1 + self.f * (p2 - p3)
                    child = eo.binomial_crossover(mutant, self.env.ev_procs[task_id].cur_pop[i], cr=self.cr)
                    rl_offspring_pops[task_id, i] = deepcopy(child)

            next_states = np.array([2, 2])
            rewards = np.array([0, 0])
            for task_id in range(self.env.n_tasks):
                # clip the population to satisfy box constraint
                rl_offspring_pops[task_id] = np.clip(rl_offspring_pops[task_id], self.env.xlb, self.env.xub)

                # fitness evaluation
                rl_offspring_fits[task_id] = self.env.tasks[task_id](rl_offspring_pops[task_id])

                # get transitioned state and reward
                if np.min(rl_offspring_fits[task_id]) < np.min(self.env.ev_procs[task_id].cur_fit):
                    next_states[task_id] = 0
                    rewards[task_id] = 10
                elif np.min(rl_offspring_fits[task_id]) == np.min(self.env.ev_procs[task_id].cur_fit):
                    next_states[task_id] = 1
                    rewards[task_id] = 5

                # get current action id
                act_id = np.argmin(np.abs(self.rmps[task_id] - self.cand_rmp_list))

                # update q-table
                self.q_tbl[task_id, self.states[task_id], act_id] += \
                    self.alpha * (rewards[task_id] + self.beta * np.max(self.q_tbl[task_id, next_states[task_id], :]) -
                                  self.q_tbl[task_id, self.states[task_id], act_id])

                # update state and select action
                self.states[task_id] = next_states[task_id]
                self.rmps[task_id] = self.cand_rmp_list[np.argmax(self.q_tbl[task_id, self.states[task_id]])]

            if np.array([len(task._y_trajectory) > self.task_max_nfe for task in self.env.tasks]).all():
                break

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

    env = gym.make('l2t_emto-v1', env_mode='test', problem_name='bbob-v1', base_solver_name='ga', max_gen=100, rec_freq=5,
                    env_kwargs=dict(alg_name='ppo-v0'))
    algo = MFEA_RL(env)
    for env_seed in range(2):
        print(algo.run(env_seed))

