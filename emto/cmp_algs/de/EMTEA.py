import numpy as np
from copy import deepcopy


class EMTEA:
    '''
        Implementation of evolutionary multitasking algorithm with explicit auto-encoding
        Ref: Feng, L., Zhou, L., Zhong, J., Gupta, A., Ong, Y. S., Tan, K. C., & Qin, A. K. (2018). Evolutionary
        multitasking via explicit autoencoding. IEEE transactions on cybernetics, 49(9), 3457-3470.
    '''

    def __init__(self, env):
        self.env = env
        self.tasks = None
        self.n_tr_sols = 10  # number of transfer solutions
        self.tr_freq = 10    # periodically transfer frequency measured by the number of generations

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))

        # calibrate the setting to approximate the original paper
        self.n_tr_sols = np.min([self.n_tr_sols, self.env.pop_size_per_task])  # should not be larger than the pop size
        self.tr_freq = int(np.ceil(self.env.max_gen / 25))  # should not be larger than the pop size

        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            for task_id in range(self.env.n_tasks):
                offspring_pop = self.env.ev_procs[task_id].sample()
                other_task_ids = self.task_ids[:task_id] + self.task_ids[task_id + 1:]
                source_task_id = np.random.choice(other_task_ids)
                source_pop = deepcopy(self.env.ev_procs[source_task_id].cur_pop)
                target_pop = deepcopy(self.env.ev_procs[task_id].cur_pop)

                # perform domain adaptation-based knowledge transfer
                if gen % self.tr_freq == 0:
                    # sort the population in descending order based on fitness
                    target_pop = target_pop[np.argsort(self.env.ev_procs[task_id].cur_fit)]
                    source_pop = source_pop[np.argsort(self.env.ev_procs[source_task_id].cur_fit)]
                    source_elit = source_pop[:self.n_tr_sols]
                    transfer_pop = explicit_autoencoding_transform(target_pop, source_pop, source_elit)
                    offspring_pop[:self.n_tr_sols] = transfer_pop

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


def explicit_autoencoding_transform(pop_t: np.ndarray, pop_s: np.ndarray, elit_s: np.ndarray)->np.array:
    '''
    Parameters
    ----------
    pop_t, pop_s
        target population and source population, both in the form of n*d matrix. %n is the number of individual, and d
        is the variable dimension. They do not have to be with the same d. We assume they have the same n (same population size)
    elit_s
        best solutions from source task
    Returns
    -------
    transformed solutions
    '''
    X_t = deepcopy(pop_t)
    X_s = deepcopy(pop_s)
    X_s_elit = deepcopy(elit_s)
    if X_t.shape[1] < X_s.shape[1]:
        X_t = np.c_[X_t, np.zeros((X_t.shape[0], X_s.shape[1] - X_t.shape[1]))]
    elif X_t.shape[1] > X_s.shape[1]:
        X_s = np.X_s[X_s, np.zeros((X_s.shape[0], X_t.shape[1] - X_s.shape[1]))]
    X_t = np.r_[X_t.T, np.ones((1,X_t.shape[0]))]
    X_s = np.r_[X_s.T, np.ones((1,X_s.shape[0]))]
    P = np.dot(X_t, X_s.T)
    Q = np.dot(X_s, X_s.T)
    reg = 1e-5 * np.eye(X_t.shape[0])
    reg[-1,-1] = 0
    W = P * np.linalg.inv(Q + reg)
    W = W[:-1,:-1]
    if X_t.shape[1] <= X_s.shape[1]:
        tr_X = np.dot(W, X_s_elit.T).T
        tr_X = tr_X[:, :X_t.shape[1]]
    else:
        X_s_elit = np.c_[X_s_elit, np.zeros((X_s_elit.shape[0], X_t.shape[1] - X_s.shape[1]))]
        tr_X = np.dot(W, X_s_elit.T).T
    return tr_X


if __name__ == '__main__':
    import emto
    import gymnasium as gym

    env = gym.make('l2t_emto-v6', problem_name='bbob-v1')
    algo = EMTEA(env)
    for env_seed in range(100):
        algo.run(env_seed)