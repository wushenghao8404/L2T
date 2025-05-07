import numpy as np
import ctypes as ct
from copy import deepcopy
from numpy.linalg import inv
# void cec14_test_func(double *x, double *f, int nx, int mx, int func_num, int benchmark_id, int task_id)
cec14_func = ct.cdll.LoadLibrary('/public2/home/wushenghao/project/L2T/problems/cec14_func.so')['cec14_test_func']


def sr_func(x, Os, Mr):  # shift and rotate
    y = x[:, :Os.shape[-1]] - Os
    return np.matmul(Mr, y.transpose()).transpose()


def rotate_gen(dim, np_gen):  # Generate a rotate matrix
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = np_gen.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


class BaseFunction:
    def __init__(self, dim, shift, rotate, lb, ub):
        assert (shift <= ub).all() and (shift >= lb).all(), 'input shift is out of space boundary'
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.lb = lb
        self.ub = ub
        self.xopt = None
        self.fopt = 0.0

    def __call__(self, x):
        raise NotImplementedError


class FunctionCEC14:
    def __init__(self, dim:int, func_id:int, benchmark_id:int, task_id:int):
        self.dim = dim
        self.func_id = func_id
        self.benchmark_id = benchmark_id
        self.task_id = task_id
        self.fopt = 0.0
        self.xopt = None

    def __call__(self, x: np.ndarray):
        if x.ndim == 1:
            # Evaluate
            assert x.shape[0] == self.dim
            y = np.full(shape=(1,), fill_value=np.nan)
            xptr = x.ctypes.data_as(ct.c_char_p)
            yptr = y.ctypes.data_as(ct.c_char_p)
            cec14_func(xptr, yptr, self.dim, 1, self.func_id, self.benchmark_id, self.task_id)
        elif x.ndim == 2:
            # Evaluate
            assert x.shape[1] == self.dim
            y = np.full(shape=(x.shape[0],), fill_value=np.nan)
            xptr = x.flatten().ctypes.data_as(ct.c_char_p)
            yptr = y.ctypes.data_as(ct.c_char_p)
            cec14_func(xptr, yptr, self.dim, x.shape[0], self.func_id, self.benchmark_id, self.task_id)
        else:
            raise Exception('Wrong input dimension')
        return y - self.func_id * 100


class Sphere(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-100.0, ub=100.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis,:], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = np.sum(z ** 2, axis=-1)
        return y


class Rastrigin(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-50.0, ub=50.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis,:], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1)
        return y


class Griewank(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-100.0, ub=100.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis, :], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = 1.0 + np.sum(z ** 2, axis=-1) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1,self.dim+1))), axis=-1)
        return y


class Rosenbrock(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-50.0, ub=50.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift + inv(self.rotate).dot(np.ones((self.dim,1))).flatten()
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis, :], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1)
        return y


class Ackley(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-50, ub=50.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis, :], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = -20.0 * np.exp(-0.3 * np.sqrt(np.sum(z ** 2, axis=-1) / self.dim)) - \
            np.exp(np.sum(np.cos(2 * np.pi * z), axis=-1) / self.dim) + 20 + np.exp(1)
        return y


class Schwefel(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-500.0, ub=500.0):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = np.ones(self.dim) * 420.96874633
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        # shift and rotate transformation do not take effect on this function,
        z = deepcopy(x)
        y = 418.9828872724339 * self.dim - np.sum(z * np.sin(np.sqrt(np.abs(z))),axis=-1)
        return y


class Weierstrass(BaseFunction):
    def __init__(self, dim, shift, rotate, lb=-0.5, ub=0.5):
        super().__init__(dim, shift, rotate, lb, ub)
        # xopt in unified space [0,1]
        self.raw_xopt = self.shift
        self.xopt = (self.raw_xopt - self.lb) / (self.ub - self.lb)
        self.a = 0.5
        self.b = 3
        self.kmax = 20
        self.b_p = np.power(self.b, np.arange(self.kmax + 1))
        self.a_p = np.power(self.a, np.arange(self.kmax + 1))
        self.c_ = self.dim * np.sum(self.a_p * np.cos(np.pi * self.b_p))

    def __call__(self, x: np.ndarray):
        assert (x <= self.ub).all() and (x >= self.lb).all(), 'input x is out of boundary'
        if x.ndim == 1:
            # Evaluate
            z = sr_func(x[np.newaxis, :], self.shift, self.rotate)
        else:
            z = sr_func(x, self.shift, self.rotate)
        y = np.sum(np.sum(self.a_p * np.cos(2 * np.pi * self.b_p *
                                            np.expand_dims(z + 0.5, 2).repeat(self.b_p.shape[0], 2)), axis=-1), axis=-1)
        return y - self.c_


if __name__ == '__main__':
    np_gen = np.random.RandomState(3)
    f = Sphere(5,np_gen.uniform(-80,80,size=(5,)),rotate_gen(5,np_gen),-100,100)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Griewank(5,np_gen.uniform(-80,80,size=(5,)),rotate_gen(5,np_gen),-100,100)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Ackley(5,np_gen.uniform(-40,40,size=(5,)),rotate_gen(5,np_gen),-50,50)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Weierstrass(5,np_gen.uniform(-0.4,0.4,size=(5,)),rotate_gen(5,np_gen),-0.5,0.5)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Rastrigin(5,np_gen.uniform(-40,40,size=(5,)),rotate_gen(5,np_gen),-50,50)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Rosenbrock(5,np_gen.uniform(-40,40,size=(5,)),rotate_gen(5,np_gen),-50,50)
    print(f(f.raw_xopt.reshape(1,-1)))
    f = Schwefel(5,np_gen.uniform(-400,400,size=(5,)),rotate_gen(5,np_gen),-500,500)
    print(f(f.raw_xopt.reshape(1,-1)))
    # f = FunctionCEC14(50, 16, 9, 2)
    # file = open('/public2/home/wushenghao/project/L2T/problems/CEC19MultiTasks/benchmark_9/bias_2', 'r')
    # x = np.random.rand(10,50)
    # x[0] = np.array(file.read().split(), dtype=np.float64)
    # print(f(x))