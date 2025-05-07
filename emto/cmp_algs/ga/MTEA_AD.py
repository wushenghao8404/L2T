import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal as mvn


class MTEA_AD:
    '''
        Implementation of the multitask differential evolution algorithm with anomaly detection
        Ref: Wang, C., Liu, J., Wu, K., & Wu, Z. (2021). Solving multitask optimization problems with adaptive knowledge
        transfer via anomaly detection. IEEE Transactions on Evolutionary Computation, 26(2), 304-318.
    '''
    def __init__(self, env):
        self.env = env
        self.tasks = None
        self.trp = 0.1

    def run(self, env_seed):
        _, env_info = self.env.reset(seed=env_seed)
        self.task_max_nfe = self.env.task_max_nfe
        self.rec_nfe = self.env.rec_nfe
        self.max_nfe = self.env.max_nfe
        self.nfe = 0
        self.task_ids = list(np.arange(self.env.n_tasks))

        eps = np.zeros(self.env.n_tasks)
        for gen in range(self.env.max_gen):
            # print('gen',gen,'error',[task.get_error() for task in self.env.tasks])

            for task_id in range(self.env.n_tasks):
                # produce next offspring population and evaluate them first
                offspring_pop = self.env.ev_procs[task_id].sample()
                offspring_pop = np.clip(offspring_pop, self.env.xlb, self.env.xub)
                offspring_fit = self.env.tasks[task_id](offspring_pop)
                self.nfe += offspring_pop.shape[0]

                if np.random.rand() < self.trp:
                    # anomaly detection-based knowledge transfer
                    if gen == 0:
                        nl = 1
                    else:
                        nl = eps[task_id]

                    # collect all populations from other tasks as candidate solutions for transfer
                    other_task_ids = np.asarray(self.task_ids[:task_id] + self.task_ids[task_id + 1:])
                    source_task_ids = other_task_ids[np.random.permutation(self.env.n_tasks - 1)[:min(self.env.n_tasks - 1, 10)]]
                    source_pops = []
                    for source_task_id in source_task_ids:
                        source_pops += list(self.env.ev_procs[source_task_id].cur_pop)
                    source_pops = np.array(source_pops)

                    # select transferred individuals from candidate pool based on population distribution
                    transfer_pop = learn_anomaly_detection(self.env.ev_procs[task_id].cur_pop, source_pops, nl)
                    transfer_pop = np.clip(transfer_pop, self.env.xlb, self.env.xub)

                    # evaluate and update fitness call counter
                    transfer_fit = self.env.tasks[task_id](transfer_pop)
                    self.nfe += transfer_pop.shape[0]

                    # calculate number of successful transferred individuals
                    sorted_ids = np.argsort(np.concatenate((self.env.ev_procs[task_id].cur_fit, offspring_fit, transfer_fit)))
                    next_pop_ids = sorted_ids[:self.env.pop_size_per_task]
                    n_success = np.sum(next_pop_ids >= (len(self.env.ev_procs[task_id].cur_fit) + len(offspring_fit)))

                    # update anomaly detection model parameter
                    eps[task_id] = n_success / transfer_pop.shape[0]

                    # update next population by elitist selection
                    self.env.ev_procs[task_id].update(np.r_[offspring_pop, transfer_pop], np.r_[offspring_fit, transfer_fit])
                else:
                    # update next population by elitist selection
                    self.env.ev_procs[task_id].update(offspring_pop, offspring_fit)

        y_trajectory = []
        y_final = np.array([np.min(task._y_trajectory[:self.task_max_nfe]) - task.f.fopt for task in self.env.tasks])
        for task_id in range(self.env.n_tasks):
            task_y_trajectory = np.array(self.env.tasks[task_id]._y_trajectory) - self.env.tasks[task_id].f.fopt
            # print(task_y_trajectory.shape)
            for i in range(1, len(task_y_trajectory)):
                task_y_trajectory[i] = np.min([task_y_trajectory[i], task_y_trajectory[i - 1]])
            y_trajectory.append(task_y_trajectory[:self.task_max_nfe:self.rec_nfe])
        y_trajectory = np.array(y_trajectory).T

        self.clean()
        return y_final, y_trajectory, env_info

    def clean(self):
        self.tasks = None


def learn_anomaly_detection(pop_t: np.ndarray, pop_s: np.ndarray, nl)->np.array:
    '''
    Parameters
    ----------
    pop_t, pop_s
        target population and source population, both in the form of n*d matrix. %n is the number of individual, and d
        is the variable dimension. They do not have to be with the same d. We assume they have the same n (same population size)
    nl
        anomaly detection parameter
    Returns
    -------
    transformed solutions
    '''
    X_t = deepcopy(pop_t)
    X_s = deepcopy(pop_s)
    n_sample = int(np.ceil(0.01 * X_t.shape[0]))
    # if X_t.shape[0] <= X_t.shape[1]:
    #     n_sample += X_t.shape[1] - X_t.shape[0]
    X_t = np.r_[X_t, np.random.rand(n_sample, X_t.shape[1])]
    mu = np.mean(X_t,axis=0)
    sig = np.cov(X_t,rowvar=False) + 1e-6 * np.eye(X_t.shape[1])
    scores = mvn.pdf(X_s, mu, sig)
    sorted_ids = np.argsort(-scores)
    if nl == 0:
        threshold = scores[sorted_ids[0]]
    else:
        threshold = scores[sorted_ids[int(np.ceil(len(scores) * nl) - 1)]]
    return X_s[scores >= threshold]


if __name__ == '__main__':
    import emto
    import gymnasium as gym
    env = gym.make('l2t_emto-v5', problem_name='bbob-v1')
    algo = MTEA_AD(env)
    for env_seed in range(2):
        print(algo.run(env_seed))