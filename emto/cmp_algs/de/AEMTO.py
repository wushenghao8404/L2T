import numpy as np
from copy import deepcopy
from emto.evol_operator import pairwise_selection

class AEMTO:
    '''
        Implementation of adaptive evolutionary multitask optimization algorithm
        Ref:  Xu, H., Qin, A. K., & Xia, S. (2021). Evolutionary multitask optimization with adaptive knowledge transfer.
        IEEE Transactions on Evolutionary Computation, 26(2), 290-303.
    '''

    def __init__(self, env):
        self.env = env
        self.tasks = None
        self.alpha = 0.5  # quality update coefficient
        self.ptlb = 0.05  # lower bound of probability of knowledge transfer
        self.ptub = 0.7   # upper bound of probability of knowledge transfer
        self.pbase = 0.3


    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))

        self.min_pts = self.pbase / (self.env.n_tasks - 1)  # The minimum source task selection probability
        qsel = np.zeros((self.env.n_tasks, self.env.n_tasks))
        psel = np.ones((self.env.n_tasks, self.env.n_tasks)) / (self.env.n_tasks - 1)
        qs = np.zeros(self.env.n_tasks)
        qo = np.zeros(self.env.n_tasks)
        pt = 0.5 * np.ones(self.env.n_tasks) * (self.ptlb + self.ptub)

        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            for task_id in range(self.env.n_tasks):
                other_task_ids = self.task_ids[:task_id] + self.task_ids[task_id + 1:]
                if np.random.rand() < pt[task_id]:
                    # inter-task knowledge transfer
                    offspring_pop = []
                    transfer_pop = []
                    nt = np.zeros(self.env.n_tasks - 1)
                    pseli = np.clip(psel[task_id, other_task_ids], 1e-25, 1e25)
                    sel_source_ids = stochastic_universal_selection(other_task_ids, pseli/np.sum(pseli), self.env.pop_size_per_task)
                    sel_source_ids = np.array(sel_source_ids)
                    for i in range(len(other_task_ids)):
                        nt[i] = np.sum(sel_source_ids == other_task_ids[i])
                    nt = nt.astype(np.int_)
                    for i in range(len(other_task_ids)):
                        if nt[i] == 0:
                            continue
                        sorted_ids = np.argsort(self.env.ev_procs[other_task_ids[i]].cur_fit)
                        source_fit_scores = self.env.pop_size_per_task - np.argsort(sorted_ids)  # using solution score
                        p_sel_sol = source_fit_scores / np.sum(source_fit_scores)
                        sel_solution_ids = roulette_wheel_selection(list(np.arange(self.env.pop_size_per_task)),p_sel_sol, nt[i])
                        transfer_pop += list(self.env.ev_procs[other_task_ids[i]].cur_pop[sel_solution_ids])
                    transfer_pop = np.array(transfer_pop)
                    # Perform knowledge transfer between source and target individuals with crossover
                    kk = 0
                    for i in range(len(other_task_ids)):
                        if nt[i] == 0:
                            continue
                        for ii in range(nt[i]):
                            mask_binocr = np.random.binomial(1, self.env.ev_procs[task_id]._cr, self.env.n_dims)
                            mask_binocr[np.random.randint(self.env.n_dims)] = 1
                            offspring_pop.append(mask_binocr * transfer_pop[kk] + (1 - mask_binocr) * self.env.ev_procs[task_id].cur_pop[kk])
                            kk += 1
                    offspring_pop = np.clip(offspring_pop, self.env.xlb, self.env.xub)
                    offspring_fit = self.env.tasks[task_id](offspring_pop)
                    # Update comparison flag between the transferred individuals and the parent
                    ns = np.zeros(self.env.n_tasks - 1)
                    kk = 0
                    for i in range(len(other_task_ids)):
                        if nt[i] == 0:
                            continue
                        for ii in range(nt[i]):
                            if offspring_fit[kk] < self.env.ev_procs[task_id].cur_fit[kk]:
                                ns[i] += 1
                            kk += 1
                    # Update source task selection quality values
                    for i in range(len(other_task_ids)):
                        if nt[i] == 0:
                            continue
                        qsel[task_id, other_task_ids[i]] = self.alpha * qsel[task_id, other_task_ids[i]] +  \
                                                           (1 - self.alpha) * ns[i] / nt[i]
                    # Update source task selection probabilities
                    for i in range(len(other_task_ids)):
                        if nt[i] == 0:
                            continue
                        pp = qsel[task_id, other_task_ids[i]] / (np.sum(qsel[task_id, other_task_ids]) + 1e-15)
                        psel[task_id, other_task_ids[i]] = self.min_pts + (1 - (self.env.n_tasks - 1) * self.min_pts) ** pp
                    # Calculate reward
                    reward = np.sum(ns) / np.sum(nt)
                    qo[task_id] = self.alpha * qo[task_id] + (1 - self.alpha) * reward
                    # Update population
                    self.env.ev_procs[task_id].memory.update_all_stat(data=(self.env.ev_procs[task_id].cur_pop,
                                                                            self.env.ev_procs[task_id].cur_fit,
                                                                            offspring_pop,
                                                                            offspring_fit))
                    cur_pop, cur_fit = pairwise_selection(self.env.ev_procs[task_id].cur_pop,
                                                           self.env.ev_procs[task_id].cur_fit, offspring_pop, offspring_fit)
                    self.env.ev_procs[task_id].cur_pop, self.env.ev_procs[task_id].cur_fit = cur_pop, cur_fit
                else:
                    offspring_pop = self.env.ev_procs[task_id].sample()
                    offspring_pop = np.clip(offspring_pop, self.env.xlb, self.env.xub)
                    offspring_fit = self.env.tasks[task_id](offspring_pop)
                    sorted_ids = np.argsort(np.r_[self.env.ev_procs[task_id].cur_fit, offspring_fit])
                    next_pop_ids = sorted_ids[:self.env.pop_size_per_task]
                    reward = np.sum(next_pop_ids>=self.env.pop_size_per_task)
                    qs[task_id] = self.alpha * qs[task_id] + (1 - self.alpha) * reward
                    self.env.ev_procs[task_id].update(offspring_pop, offspring_fit)
                pt[task_id] = self.ptlb + (self.ptub - self.ptlb) * qo[task_id] / (qs[task_id] + qo[task_id] + 1e-15)
                self.nfe += offspring_pop.shape[0]

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


def roulette_wheel_selection(items: list, probs: np.array, n_select_items: int=1) -> list:
    assert (probs >= 0).all(), 'The input probability vector should not contain negative values'
    items_ = np.asarray(items)
    selected_items = []
    cumulative_probs = np.cumsum(probs/ np.sum(probs))
    for i in range(n_select_items):
        selected_items.append(items_[np.random.rand() < cumulative_probs][0])
    return selected_items


def stochastic_universal_selection(items: list, probs: np.array, n_select_items: int=1) -> list:
    assert (probs >= 0).all(), 'The input probability vector should not contain negative values'
    items_ = np.asarray(items)
    selected_items = []
    cumulative_probs = np.cumsum(probs / np.sum(probs))
    seg_len = 1.0 / n_select_items
    rs_ = np.random.rand() * seg_len  # a random starting pointer
    for i in range(n_select_items):
        ptr = rs_ + i * seg_len
        selected_items.append(items_[ptr < cumulative_probs][0])
    return selected_items


if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make('l2t_emto-v6', problem_name='bbob-v1')
    algo = AEMTO(env)
    for env_seed in range(100):
        algo.run(env_seed)