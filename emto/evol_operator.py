import numpy as np
from copy import deepcopy


def diff_evol_current_to_pbest_mutation(pop, current_id, sorted_ids, p=0.05, f=0.5, archive=None):
    """ Differential Evolution current_to_pbest/1 with/without archive Mutation

    Parameters
    ----------
    pop
        trial individuals
    f
        scale factor
    Returns
    -------
    c
        child individual
    """
    assert pop.ndim == 2
    pbest_id = sorted_ids[np.random.randint(np.ceil(p * pop.shape[0]))]
    if archive is None:
        r_ = np.random.permutation(pop.shape[0])
        c = pop[current_id] + f * (pop[pbest_id] - pop[current_id]) + f * (pop[r_[0]] - pop[r_[1]])
    else:
        r1 = np.random.randint(pop.shape[0])
        pop_arhicve = np.r_[pop, archive]
        r2 = np.random.randint(pop_arhicve.shape[0])
        while r2 == current_id or r2 == r1:
            r2 = np.random.randint(pop_arhicve.shape[0])
        c = pop[current_id] + f * (pop[pbest_id] - pop[current_id]) + f * (pop[r1] - pop_arhicve[r2])
    return c


def diff_evol_rand_1_mutation(pop, f=0.5):
    """ Differential Evolution Rand/1 Mutation

    Parameters
    ----------
    p1
        trial individuals
    F
        scale factor
    Returns
    -------
    c
        child individual
    """
    assert pop.ndim == 2
    r_ = np.random.permutation(pop.shape[0])
    c = pop[r_[0],:] + f * (pop[r_[1],:] - pop[r_[2],:])
    return c


def diff_evol_best_1_mutation(pop, gbest, f=0.5):
    """ Differential Evolution Best/1 Mutation

    Parameters
    ----------
    p1
        trial individuals
    F
        scale factor
    Returns
    -------
    c
        child individual
    """
    assert pop.ndim == 2
    r_ = np.random.permutation(pop.shape[0])
    c = gbest + f * (pop[r_[1], :] - pop[r_[2], :])
    return c


def binomial_crossover(p1, p2, cr=0.5):
    """ Binomial Crossover

    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals
    cr
        crossover rate

    Returns
    -------
    c
        child individuals
    """
    assert p1.shape == p2.shape
    if p1.ndim == 1:
        mask_binocr = np.random.binomial(1, cr, p1.shape[0])  # for binomial crossover
        mask_binocr[np.random.randint(p1.shape[0])] = 1
        c = mask_binocr * p1 + (1 - mask_binocr) * p2
    elif p1.ndim == 2:
        mask_binocr = np.random.binomial(1, cr, p1.shape)  # for binomial crossover
        rep_ = np.random.randint(p1.shape[1],size=p1.shape[0])
        mask_binocr[np.arange(p1.shape[0]), rep_] = 1
        c = mask_binocr * p1 + (1 - mask_binocr) * p2
    else:
        raise Exception('Unexpected input dimension, should be smaller than 3')
    return c


def two_point_crossover(p1, p2):
    """ Two-point Crossover

    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals

    Returns
    -------
    c
        child individuals
    """
    assert p1.shape == p2.shape
    if p1.ndim == 1:
        r_ = np.random.randint(p1.shape[0], size=(2,))
        pt1, pt2 = np.min(r_), np.max(r_)
        c = np.concatenate((p1[:pt1], p2[pt1:pt2], p1[pt2:]))
    elif p1.ndim == 2:
        r_ = np.random.randint(p1.shape[1], size=(p1.shape[0], 2))
        pt1, pt2 = np.array([np.min(r__) for r__ in r_]), np.array([np.max(r__) for r__ in r_])
        c = np.array([np.concatenate((p1[i,:pt1[i]], p2[i,pt1[i]:pt2[i]], p1[i,pt2[i]:])) for i in range(p1.shape[0])])
    else:
        raise Exception('Unexpected input dimension, should be smaller than 3')
    return c


def uniform_crossover(p1, p2):
    """ Uniform Crossover
    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals

    Returns
    -------
        child individuals
    """
    return binomial_crossover(p1, p2, cr=0.5)


def arithmetic_crossover(p1, p2, lambda_=0.5):
    """ Arithmetic Crossover
    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals
    lambda_
        controls the mixing ratio
    Returns
    -------
        child individuals
    """
    return lambda_ * p1 + (1 - lambda_) * p2


def geometric_crossover(p1, p2, omega_=0.5):
    """ Geometric Crossover
    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals
    omega_
        controls the mixing ratio
    Returns
    -------
        child individuals
    """
    return (p1 ** omega_) * (p2 ** (1 - omega_))


def blx_alpha_crossover(p1, p2, alpha_=0.5):
    """ BLX-alpha Crossover
    Parameters
    ----------
    p1
        trial individuals
    p2
        parent individuals
    alpha_
        controls the mixing ratio
    Returns
    -------
    c
        child individuals
    """
    I_ = np.abs(p1 - p2)
    pmin = np.minimum(p1,p2)
    c = (pmin - I_ * alpha_) + np.random.random(p1.shape) * (I_ + 2 * I_ * alpha_)
    return c


def simulated_binary_crossover(p1, p2, yita_c=2.0):
    """ Simulated Binary Crossover

    Parameters
    ----------
    p1, p2
        parent individuals
    yita_c
        tunable parameter

    Returns
    -------
    c1, c2
        child individuals
    """
    assert p1.ndim == 1 and p2.ndim == 1
    assert p1.shape == p2.shape
    u = np.random.rand(len(p1))
    gamma = np.zeros(len(p1))
    gamma[u <= 0.5] = np.power(2 * u[u <= 0.5], 1.0 / (yita_c + 1))
    gamma[u > 0.5] = np.power(0.5 / (1 - u[u > 0.5]), 1.0 / (yita_c + 1))
    c1 = 0.5 * ((1 + gamma) * p1 + (1 - gamma) * p2)  # child 1
    c2 = 0.5 * ((1 - gamma) * p1 + (1 + gamma) * p2)
    return c1, c2


def polynomial_mutation(p, yita_m=5.0):
    """ Polynomial Mutation

    Parameters
    ----------
    p
        parent individual
    yita_m
        tunable parameter

    Returns
    -------
    c
        child individual
    """
    assert p.ndim == 1
    dim = len(p)
    pm = 1.0 / dim
    u = np.random.random(dim)
    rd = np.random.random(dim)
    beta = np.zeros(dim)
    index = (u < 0.5) & (rd < pm)
    beta[index] = np.power(2 * u[index], 1.0 / (yita_m + 1)) - 1
    index = (u >= 0.5) & (rd < pm)
    beta[index] = 1 - np.power(2 * (1 - u[index]), 1.0 / (yita_m + 1))
    c = p + beta
    return c


def pairwise_selection(p1, f1, p2, f2, return_cmp_mask=False):
    """ Pairwise Selection

    Parameters
    ----------
    p1, p2
        parent population and offspring population
    f1, f2
        parent fitness and offspring fitness

    Returns
    -------
    np, np
        new population and new fit
    """
    assert p1.ndim == 2 and p2.ndim == 2 and p1.shape[1] == p2.shape[1]
    mask_cmp = f2 < f1
    new_pop = deepcopy(p1)
    new_fit = deepcopy(f1)
    new_pop[mask_cmp, :] = p2[mask_cmp, :]
    new_fit[mask_cmp] = f2[mask_cmp]
    if return_cmp_mask:
        return new_pop, new_fit, mask_cmp
    else:
        return new_pop, new_fit


def elitist_selection(p1, f1, p2, f2):
    """ Elitist Selection

    Parameters
    ----------
    p1, p2
        parent population and offspring population
    f1, f2
        parent fitness and offspring fitness

    Returns
    -------
    np, np
        new population and new fit
    """
    assert p1.ndim == 2 and p2.ndim == 2 and p1.shape[1] == p2.shape[1]
    union_pop = np.r_[p1, p2]
    union_fit = np.r_[f1, f2]
    sorted_id = np.argsort(union_fit)
    new_pop = union_pop[sorted_id[:p1.shape[0]], :]
    new_fit = union_fit[sorted_id[:p1.shape[0]]]
    return new_pop, new_fit


if __name__ == '__main__':
    a = np.random.random(10)
    print(a)
    a[a > 0.5] = np.zeros(len(a[a > 0.5]))
    print(a)
