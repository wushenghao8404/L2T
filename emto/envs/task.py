import os
from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from problems.base_func import FunctionCEC14, Sphere, Rastrigin, Weierstrass, Griewank, Rosenbrock, Ackley, Schwefel
import time


class Timer:

    elapsed_times: list
    t0: float
    t1: float
    eval_unit: float

    def setup(self, eval_unit=1e-6):
        self.eval_unit = eval_unit

    def sleep(self, n):
        time.sleep(n * self.eval_unit)

    def reset(self):
        self.elapsed_times = []

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.t1 = time.time()
        self.elapsed_times.append(self.t1 - self.t0)

    def get_elapsed_time(self):
        return np.sum(self.elapsed_times)


eval_timer = Timer()
_eval_sleep = False
_record_time = False


def set_timer(eval_unit=1e-6):
    global _eval_sleep, _record_time
    eval_timer.setup(eval_unit=eval_unit)
    eval_timer.reset()
    _eval_sleep = True
    _record_time = True


def get_recorded_time():
    # print(eval_timer.elapsed_times)
    return eval_timer.get_elapsed_time()


class Task(ABC):
    def __init__(self, prob_name, max_nfe, record_traj, ):
        self.prob_name = prob_name
        self.max_nfe = max_nfe
        self._y_best = np.inf
        self._x_trajectory = []
        self._y_trajectory = []
        self._trajectory = []
        self._targets = None
        self.record_traj = record_traj
        self._nfe = 0

    def __call__(self, x):
        # print("invoke __call__ function")

        if _record_time:
            eval_timer.tic()

        if x.ndim == 1:
            # Evaluate
            n = 1  # number of solutions to be evaluated
            y = self.evaluate(x)
            self._nfe += 1
            self._y_trajectory.append(y)
            self._trajectory.append(self._nfe)
            # Record trajectory
            if self.record_traj:
                self._x_trajectory.append(x)
        elif x.ndim == 2:
            # Evaluate
            n = x.shape[0]  # number of solutions to be evaluated
            y = self.evaluate(x)
            self._trajectory += list(np.arange(self._nfe, self._nfe + len(y)))
            self._nfe += len(y)
            self._y_trajectory += list(y)
            # Record trajectory
            if self.record_traj:
                self._x_trajectory.append(x)
        else:
            raise Exception('Wrong input dimension')
        self._y_best = np.min([self._y_best, np.min(y)])

        if _eval_sleep:
            eval_timer.sleep(n)

        if _record_time:
            eval_timer.toc()

        return y

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def nb_evaluations(self):
        return self._nfe

    @property
    def x_trajectory(self):
        return self._x_trajectory

    @property
    def y_trajectory(self):
        return self._y_trajectory

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass


class TaskBBOB(Task):
    def __init__(self):
        super().__init__(prob_name='bbob', max_nfe=1e25, record_traj=False)

    def __str__(self):
        return f'TaskBBOB => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, dim, targets_amount, targets_precision):
        # print('fid','iid',f_id,i_id)
        # http://cma.gforge.inria.fr/apidocs-pycma/cma.bbobbenchmarks.html
        from cma import bbobbenchmarks as bn
        self.f_id = f_id
        self.i_id = i_id
        self.dim = dim
        self.targets_amount = targets_amount
        self.targets_precision = targets_precision
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.f = eval(f'bn.F{self.f_id}({self.i_id})')
        return self

    @property
    def targets(self):
        if self._targets is None:
            fopt = self.f.fopt
            self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
        return self._targets

    def evaluate(self, x):
        assert (x <= 1).all() and (x >= 0).all(), 'input x is out of boundary'
        return self.f(x * (self.ub - self.lb) + self.lb)

    def get_error(self):
        # print(self._y_trajectory, self.f.fopt)
        return self._y_best - self.f.fopt

    def has_reach_target(self):
        return self.get_error() <= 10 ** self.targets_precision


class TaskHPO(Task):

    def __init__(self):
        super().__init__(prob_name='hpo', max_nfe=1e25, record_traj=False)

    def __str__(self):
        return f'TaskHPO => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id:int, i_id:int, targets_precision, noise=False):
        from emukit.examples.profet.meta_benchmarks import meta_forrester
        from emukit.examples.profet.meta_benchmarks import meta_svm
        from emukit.examples.profet.meta_benchmarks import meta_fcnet
        from emukit.examples.profet.meta_benchmarks import meta_xgboost
        self.probs = ['forrester', 'svm', 'xgboost', 'fcnet']
        self.f_id = f_id
        self.i_id = i_id
        self.noise = noise
        self.targets_precision = targets_precision
        self.prob_name = self.probs[self.f_id]
        self.path_prefix = '/public2/home/wushenghao/project/L2T/problem_data/profet/profet_data'
        path_objective = f'/{self.path_prefix}/samples/{self.prob_name}/sample_objective_{str(self.i_id)}.pkl'
        path_cost = f'/{self.path_prefix}/samples/{self.prob_name}/sample_cost_{str(self.i_id)}.pkl'

        if self.prob_name == "svm":
            self.f, parameter_space = meta_svm(path_objective, path_cost, self.noise)
        elif self.prob_name == "xgboost":
            self.f, parameter_space = meta_xgboost(path_objective, path_cost, self.noise)
        elif self.prob_name == "fcnet":
            self.f, parameter_space = meta_fcnet(path_objective, path_cost, self.noise)
        else:
            raise Exception(self.prob_name + " not implemented!")
        self.dim = len(parameter_space.parameter_names)
        if hasattr(self.f, 'xopt'):
            # In HPO benchmark, no ground-truth x optimum is provided so set it to None
            self.f.xopt = None
        else:
            setattr(self.f, 'xopt', None)
        if hasattr(self.f, 'fopt'):
            # In HPO benchmark, y corresponds to the validation error of an ML model, so a rough estimation of lower
            # bound is zero.
            self.f.fopt = 0.0
        else:
            setattr(self.f, 'fopt', 0.0)
        return self

    @property
    def targets(self):
        if self._targets is None:
            import json
            path_target_file = self.path_prefix + "/targets/meta_" + self.prob_name + "_noiseless_targets.json"
            # print(path_prefix)
            with open(path_target_file) as f:
                targets = np.array(json.load(f))
            targets = targets[self.i_id]
            # print(targets)
            traj = []
            curr = np.inf
            for t in targets:
                if t < curr:
                    curr = t
                traj.append(curr)
            traj = np.array(traj)
            self._targets = traj
        return self._targets

    def get_error(self):
        # print(self._y_trajectory, self.f.fopt)
        return np.max([0, self._y_best]) - self.f.fopt
        # return self._y_best - self.f.fopt

    def has_reach_target(self):
        return self.get_error() <= 10 ** self.targets_precision

    def evaluate(self, x):
        if x.ndim == 1:
            x = np.array(x)[np.newaxis]
        f_mean, f_std = self.f(x)
        if x.ndim == 1:
            # evaluate single solution
            return f_mean[0,0]
        else:
            # evaluate multiple solutions
            return f_mean[:,0]


class TaskCEC19(Task):
    def __init__(self):
        super().__init__(prob_name='cec19', max_nfe=1e25, record_traj=False)

    def __str__(self):
        return f'TaskCEC19 => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, benchmark_id, dim, lb, ub):
        assert f_id in list(np.arange(30) + 1)
        assert i_id in [1,2]
        self.f_id = f_id  # value should lie in [1, 30]
        self.i_id = i_id  # value should lie in [0, 1]
        self.benchmark_id = benchmark_id
        self.dim = dim
        self.lb = lb * np.ones(self.dim)
        self.ub = ub * np.ones(self.dim)
        self.targets_amount = 10
        self.targets_precision = -8
        self.f = FunctionCEC14(dim=self.dim, func_id=self.f_id, benchmark_id=self.benchmark_id, task_id=self.i_id)
        return self

    @property
    def targets(self):
        if self._targets is None:
            fopt = self.f.fopt
            self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
        return self._targets

    def evaluate(self, x):
        assert (x <= 1).all() and (x >= 0).all(), 'input x is out of boundary'
        return self.f(x[:, :self.dim] * (self.ub - self.lb) + self.lb)

    def get_error(self):
        # print(self._y_trajectory, self.f.fopt)
        return self._y_best - self.f.fopt

    def has_reach_target(self):
        return self.get_error() <= 10 ** self.targets_precision


class TaskCEC17(Task):
    function_map = {1: Ackley,
                    2: Griewank,
                    3: Rastrigin,
                    4: Rosenbrock,
                    5: Schwefel,
                    6: Sphere,
                    7: Weierstrass}

    def __init__(self):
        super().__init__(prob_name='cec17', max_nfe=1e25, record_traj=False)

    def __str__(self):
        return f'TaskCEC17 => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, benchmark_id, dim, shift, rotate, lb, ub):
        assert f_id in [1,2,3,4,5,6,7]
        assert i_id in [1,2]
        self.f_id = f_id  # value should lie in [1, 7]
        self.i_id = i_id  # value should lie in [0, 1]
        self.benchmark_id = benchmark_id
        self.dim = dim
        self.lb = lb * np.ones(self.dim)
        self.ub = ub * np.ones(self.dim)
        self.targets_amount = 10
        self.targets_precision = -8
        self.f = self.function_map[f_id](self.dim, shift, rotate, lb=lb, ub=ub)
        return self

    @property
    def targets(self):
        if self._targets is None:
            fopt = self.f.fopt
            self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
        return self._targets

    def evaluate(self, x):
        assert (x <= 1).all() and (x >= 0).all(), 'input x is out of boundary'
        return self.f(x[:, :self.dim] * (self.ub - self.lb) + self.lb)

    def get_error(self):
        # print(self._y_trajectory, self.f.fopt)
        return self._y_best - self.f.fopt

    def has_reach_target(self):
        return self.get_error() <= 10 ** self.targets_precision


class TaskCustom(Task):
    '''
        The class of configurable tasks that allows customization on the task characteristic such as function type,
        global optimum (if can), and rotation matrix
    '''
    function_map = {1: Ackley,
                    2: Griewank,
                    3: Rastrigin,
                    4: Rosenbrock,
                    5: Schwefel,
                    6: Sphere,
                    7: Weierstrass}

    function_boundary = {1: dict(lb=-50, ub=50),
                         2: dict(lb=-100, ub=100),
                         3: dict(lb=-50, ub=50),
                         4: dict(lb=-50, ub=50),
                         5: dict(lb=-500, ub=500),
                         6: dict(lb=-100, ub=100),
                         7: dict(lb=-0.5, ub=0.5),}

    def __init__(self):
        super().__init__(prob_name='func_custom', max_nfe=1e25, record_traj=False)

    def __str__(self):
        return f'TaskCustom => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, dim, shift, rotate):
        assert (shift <= 1).all() and (shift >= 0).all(), 'input x is out of boundary'
        assert f_id in [1, 2, 3, 4, 5, 6, 7]
        assert i_id in [1, 2]
        self.f_id = f_id  # value should lie in [1, 7]
        self.i_id = i_id
        self.dim = dim
        self.lb = self.function_boundary[f_id]['lb'] * np.ones(self.dim)
        self.ub = self.function_boundary[f_id]['ub'] * np.ones(self.dim)
        self.shift = shift * (self.ub -self.lb) + self.lb
        self.targets_amount = 10
        self.targets_precision = -8
        self.f = self.function_map[f_id](self.dim, self.shift, rotate, lb=self.function_boundary[f_id]['lb'],
                                         ub=self.function_boundary[f_id]['ub'])
        return self

    @property
    def targets(self):
        if self._targets is None:
            fopt = self.f.fopt
            self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
        return self._targets

    def evaluate(self, x):
        assert (x <= 1).all() and (x >= 0).all(), 'input x is out of boundary'
        return self.f(x[:, :self.dim] * (self.ub - self.lb) + self.lb)

    def get_error(self):
        # print(self._y_trajectory, self.f.fopt)
        return self._y_best - self.f.fopt

    def has_reach_target(self):
        return self.get_error() <= 10 ** self.targets_precision


if __name__ == "__main__":

    # # Example BBOB:
    # f = TaskBBOB().build(1, 1, 2, 10, -3)
    #
    # X = np.array([[0.0, 1.0], [0.2, 0.4], [0.3, 0.4]])
    # Y = [f(x) for x in X]
    #
    # print()
    # print('==================================================')
    # print(f'Task: {f}')
    # print('==================================================')
    # print('x_trajectory | y_trajectory')
    # for x, y in zip(f.x_trajectory, f.y_trajectory):
    #     print(f'{x}   | {y}')
    # print('--------------------------------------------------')
    # # print(bbob'Targets solved: {bbob.get_percentage_of_targets_solved()}')
    # print('Targets:')
    # print(f.targets)
    # print()

    # Example HPO:
    f = TaskHPO().build(3, 999, -3)
    print(f.dim)
    X = np.random.rand(20, f.dim)
    Y = f(X) # evaluate multiple solutions
    # Y = [f(x) for x in X] # evaluate single solution

    print()
    print('==================================================')
    print(f'Task: {f}')
    print('==================================================')
    print('x_trajectory | y_trajectory')
    for x, y in zip(f.x_trajectory, f.y_trajectory):
        print(f'{x}   | {y}')
    print('--------------------------------------------------')
    # print(f'Targets solved: {f.get_percentage_of_targets_solved()}')
    print('Task error:', f.get_error())
    print('Targets:')
    print(f.targets[0], f.targets[-1])
    print()