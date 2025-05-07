from copy import deepcopy
import numpy as np
import emto.evol_operator as eo


def get_de_setting(config):
    pass


class Memory:
    def __init__(self, stat_names):
        self.gbx = None
        self.gby = np.inf
        self.gbx_improved = False
        self.n_stag = 0  # number of stagnation
        # archive for storing the best solutions so far
        self.archive_size = 100
        self.archive_x = None
        self.archive_y = None
        # context for recording evolutionary trajectory within a time window
        self.context_size = 100
        self.context_x = None
        self.context_y = None
        self.prev_pop = None
        self.prev_fit = None
        self.offspring_pop = None
        self.offspring_fit = None
        self.stat_names = stat_names

    def setup(self, params):
        self.params = params

    def get(self, stat_name):
        if stat_name == 'gbest':
            return self.gbx, self.gby
        elif stat_name == 'gbx_improved':
            return self.gbx_improved
        elif stat_name == 'n_stag':
            return self.n_stag
        elif stat_name == 'context':
            return self.context_x, self.context_y
        elif stat_name == 'archive':
            return self.archive_x, self.archive_y
        elif stat_name == 'prev':
            return self.prev_pop, self.prev_fit
        elif stat_name == 'offspring':
            return self.offspring_pop, self.offspring_fit
        else:
            raise Exception('Invalid statistics name')
        pass

    def update_all_stat(self, data):
        for stat_name in self.stat_names:
            self.update(stat_name, data)

    def update(self, stat_name, data):
        self.prev_pop, self.prev_fit, self.offspring_pop, self.offspring_fit = deepcopy(data)
        if self.prev_pop is None:
            self.prev_pop = deepcopy(self.offspring_pop)
            self.prev_fit = deepcopy(self.offspring_fit)
        if stat_name == 'gbest':
            self.n_stag += 1
            self.gbx_improved = False
            best_index = np.argmin(self.offspring_fit)
            if self.offspring_fit[best_index] < self.gby:
                self.gbx = self.offspring_pop[best_index, :]
                self.gby = self.offspring_fit[best_index]
                self.gbx_improved = True
                self.n_stag = 0
        elif stat_name == 'archive':
            if self.archive_x is None:
                self.archive_x = deepcopy(self.offspring_pop)
                self.archive_y = deepcopy(self.offspring_fit)
            elif self.archive_x.shape[0] < self.archive_size:
                self.archive_x = np.r_[self.archive_x, self.offspring_pop]
                self.archive_y = np.r_[self.archive_y, self.offspring_fit]
            else:
                self.archive_x = self.archive_x[:self.archive_size]
                self.archive_y = self.archive_y[:self.archive_size]
                self.archive_x, self.archive_y = eo.elitist_selection(self.archive_x, self.archive_y, 
                                                                      self.offspring_pop, self.offspring_fit)
        elif stat_name == 'context':
            if self.context_x is None:
                self.context_x = deepcopy(self.offspring_pop)
                self.context_y = deepcopy(self.offspring_fit)
            elif self.context_x.shape[0] < self.context_size:
                self.context_x = np.r_[self.context_x, self.prev_pop]
                self.context_y = np.r_[self.context_y, self.prev_fit]
            else:
                self.context_x = np.r_[self.context_x[:self.prev_pop.shape[0]], self.prev_pop]
                self.context_y = np.r_[self.context_y[:self.prev_fit.shape[0]], self.prev_fit]
        else:
            raise Exception('Invalid statistics name')
        pass


class EvolMultitaskOptimizer:
    def transfer(self):
        pass

    def sample(self):
        pass

    def update(self):
        pass


class EvolOptimizer:
    def update_param(self, param):
        self.param = param

    def ask(self):
        raise NotImplementedError

    def ask_one(self):
        raise NotImplementedError

    def tell(self, fit, ):
        raise NotImplementedError

    def get_alg_setting(self):
        raise NotImplementedError


class DiffEvol(EvolOptimizer):
    def __init__(self, prob_dim, pop_size, mutation_op=None, mutation_param=None, crossover_op=None, crossover_param=None,
                 selection_op=None, **kwargs):
        self.update_param({})
        self.dim = prob_dim
        self.pop_size = pop_size
        self.cur_pop = None
        self.cur_fit = None
        self.memory = Memory(['gbest', 'archive'])

        if mutation_op is None:
            self.mutation = eo.diff_evol_rand_1_mutation
            self._f = 0.5
            self.param['mutation'] = dict(f=self._f)
        else:
            self.mutation = mutation_op
            self.param['mutation'] = mutation_param

        if crossover_op is None:
            self.crossover = eo.binomial_crossover
            self._cr = 0.5
            self.param['crossover'] = dict(cr=self._cr)
        else:
            self.crossover = crossover_op
            self.param['crossover'] = crossover_param

        if selection_op is None:
            self.selection = eo.elitist_selection
        else:
            self.selection = selection_op

    def setup(self, task, init_pop=None, init_fit=None):
        if init_pop is not None:
            assert init_pop.shape[0] == self.pop_size
            assert (init_pop <= 1).all() and (init_pop >= 0).all(), 'input initial population is out of boundary'
            self.cur_pop = deepcopy(init_pop)
        else:
            self.cur_pop = np.random.rand(self.pop_size, self.dim)
        if init_fit is not None:
            assert len(init_fit) == self.pop_size
            self.cur_fit = init_fit
        else:
            self.cur_fit = task(self.cur_pop)
        self.memory.update_all_stat(data=(None, None, self.cur_pop, self.cur_fit))

    def sample(self):
        parent_ids = list(np.arange(self.pop_size))
        offspring_pop = []
        for parent_id in parent_ids:
            offspring_pop.append(self.crossover(self.mutation(self.cur_pop, **self.param['mutation']),
                                                self.cur_pop[parent_id, :], **self.param['crossover']))
        return np.array(offspring_pop)

    def update(self, offspring_pop, offspring_fit):
        self.memory.update_all_stat(data=(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit))
        self.cur_pop, self.cur_fit = self.selection(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit)
        return self.cur_pop, self.cur_fit

    def get_alg_setting(self):
        return {'alg_name': 'diff_evol',
                'mutation': self.mutation.__name__,
                'F': self._f,
                'crossover': self.crossover.__name__,
                'Cr': self._cr,
                'selection': self.selection.__name__,
                'pop_size': self.pop_size}


class SelfAdaptiveDiffEVol(DiffEvol):
    def __init__(self, prob_dim, pop_size):
        super().__init__(prob_dim, pop_size,
                         mutation_op=eo.diff_evol_current_to_pbest_mutation,
                         selection_op=eo.pairwise_selection)
        self.param['top_ratio'] = 0.1
        self.param['c'] = 0.1
        self.param['with_archive'] = False
        self.param['dynamic'] = dict(mean_f=0.5, mean_cr=0.5)
        self.trial_fs = []
        self.trial_crs = []

    def sample(self):
        if self.param['with_archive']:
            raise NotImplementedError

        parent_ids = list(np.arange(self.pop_size))
        sorted_ids = np.argsort(self.cur_fit)
        offspring_pop = []
        self.trial_fs = []
        self.trial_crs = []

        for parent_id in parent_ids:
            f = self.param['dynamic']['mean_f'] + np.random.standard_cauchy() * 0.1
            while f <= 0:
                f = self.param['dynamic']['mean_f'] + np.random.standard_cauchy() * 0.1
            f = min(f, 1.0)
            cr = max(min(self.param['dynamic']['mean_cr'] + np.random.randn() * 0.1, 1.0), 0.0)
            self.trial_fs.append(f)
            self.trial_crs.append(cr)
            mutated_child = self.mutation(self.cur_pop, parent_id, sorted_ids, p=self.param['top_ratio'],
                                          f=f, archive=None)
            offspring_pop.append(self.crossover(mutated_child, self.cur_pop[parent_id, :], cr=cr))
        return np.array(offspring_pop)

    def update(self, offspring_pop, offspring_fit):
        self.memory.update_all_stat(data=(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit))
        self.cur_pop, self.cur_fit, cmp_mask = self.selection(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit,
                                                              return_cmp_mask=True)
        if np.any(cmp_mask):
            success_fs = np.array(self.trial_fs)[cmp_mask]
            success_crs = np.array(self.trial_crs)[cmp_mask]
            mean_success_f = np.sum(success_fs ** 2) / np.sum(success_fs)

            self.param['dynamic']['mean_f'] = ((1 - self.param['c']) * self.param['dynamic']['mean_f'] +
                                               self.param['c'] * mean_success_f)

            self.param['dynamic']['mean_cr'] = ((1 - self.param['c']) * self.param['dynamic']['mean_cr'] +
                                                self.param['c'] * np.mean(success_crs))
        return self.cur_pop, self.cur_fit

    def get_alg_setting(self):
        return {'alg_name': 'self_adaptive_diff_evol',
                'mutation': self.mutation.__name__,
                'crossover': self.crossover.__name__,
                'selection': self.selection.__name__,
                'param'   : self.param,
                'pop_size': self.pop_size}


class GeneAlgo(EvolOptimizer):
    def __init__(self, prob_dim, pop_size):
        self.dim = prob_dim
        self.pop_size = pop_size
        assert self.pop_size % 2 == 0
        self.cur_pop = None
        self.cur_fit = None
        self.memory = Memory(['gbest', 'archive'])
        self.mutation = eo.polynomial_mutation
        self._yita_m = 5.0
        self.crossover = eo.simulated_binary_crossover
        self._yita_c = 2.0
        self.selection = eo.elitist_selection

    def setup(self, task, init_pop=None, init_fit=None):
        if init_pop is not None:
            assert init_pop.shape[0] == self.pop_size
            assert (init_pop <= 1).all() and (init_pop >= 0).all(), 'input initial population is out of boundary'
            self.cur_pop = deepcopy(init_pop)
        else:
            self.cur_pop = np.random.rand(self.pop_size, self.dim)
        if init_fit is not None:
            assert len(init_fit) == self.pop_size
            self.cur_fit = init_fit
        else:
            self.cur_fit = task(self.cur_pop)
        self.memory.update_all_stat(data=(None, None, self.cur_pop, self.cur_fit))

    def sample(self):
        offspring_pop = []
        shuffled_ids = np.random.permutation(self.pop_size)
        for i in range(int(self.pop_size / 2)):
            child1, child2 = self.crossover(self.cur_pop[shuffled_ids[i * 2], :], self.cur_pop[shuffled_ids[i * 2 + 1], :],
                                            yita_c=self._yita_c)
            child1 = self.mutation(child1, yita_m=self._yita_m)
            child2 = self.mutation(child2, yita_m=self._yita_m)
            offspring_pop.append(child1)
            offspring_pop.append(child2)
        return np.array(offspring_pop)

    def update(self, offspring_pop, offspring_fit):
        self.memory.update_all_stat(data=(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit))
        self.cur_pop, self.cur_fit = self.selection(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit)
        return self.cur_pop, self.cur_fit

    def get_alg_setting(self):
        return {'alg_name': 'gene_algo',
                'mutation': self.mutation.__name__,
                'yita_m': self._yita_m,
                'crossover': self.crossover.__name__,
                'yita_c': self._yita_c,
                'selection': self.selection.__name__,
                'pop_size': self.pop_size}


class EstimationOfDistributionAlgo(EvolOptimizer):
    def __init__(self, prob_dim, pop_size):
        self.dim = prob_dim
        self.pop_size = pop_size
        assert self.pop_size % 2 == 0
        self.cur_pop = None
        self.cur_fit = None
        self.memory = Memory(['gbest','archive'])
        self.mutation = eo.polynomial_mutation
        self.crossover = eo.simulated_binary_crossover
        self.selection = eo.elitist_selection

    def setup(self, task, init_pop=None, init_fit=None):
        if init_pop is not None:
            assert init_pop.shape[0] == self.pop_size
            assert (init_pop <= 1).all() and (init_pop >= 0).all(), 'input initial population is out of boundary'
            self.cur_pop = deepcopy(init_pop)
        else:
            self.cur_pop = np.random.rand(self.pop_size, self.dim)
        if init_fit is not None:
            assert len(init_fit) == self.pop_size
            self.cur_fit = init_fit
        else:
            self.cur_fit = task(self.cur_pop)
        self.memory.update_all_stat(data=(None, None, self.cur_pop, self.cur_fit))

    def sample(self):
        raise NotImplementedError

    def update(self, offspring_pop, offspring_fit):
        self.memory.update_all_stat(data=(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit))
        self.cur_pop, self.cur_fit = self.selection(self.cur_pop, self.cur_fit, offspring_pop, offspring_fit)
        return self.cur_pop, self.cur_fit
