import pickle
import config
from copy import deepcopy
from typing import Callable, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_obs_features(alg_name):
    if alg_name not in config.ALG_NAMES and alg_name.replace('_best', '') not in config.ALG_NAMES:
        raise ValueError
    common_features = ['oc1', 'oc2', 'oc3', 'oc4']
    task_features = ['oi1', 'oi2', 'oi3', 'oi4']
    if 'ppo-v0-s1' in alg_name:
        common_features = []
        task_features = ['oi1', 'oi2', 'oi3', 'oi4']
    elif 'ppo-v0-s2' in alg_name:
        common_features = ['oc1', 'oc2', 'oc3', 'oc4']
        task_features = []
    elif 'ppo-v0-spop' in alg_name:
        common_features = []
        task_features = []
    else:
        print(alg_name, 'use default observation feature set', common_features + task_features)
    return common_features, task_features


def load_actions(alg_name):
    if alg_name not in config.ALG_NAMES and alg_name.replace('_best', '') not in config.ALG_NAMES:
        raise ValueError
    actions = ['a1', 'a3', 'a5']
    if 'ppo-v0-a1' in alg_name:
        actions = ['a3', 'a5']
    elif 'ppo-v0-a2' in alg_name:
        actions = ['a1', 'a5']
    elif 'ppo-v0-a3' in alg_name:
        actions = ['a1', 'a3']
    else:
        print(alg_name, 'use default action set', actions)
    return actions


def load_hyper_param(alg_name, default_param):
    hyper_param = deepcopy(default_param)
    if alg_name not in config.ALG_NAMES and alg_name.replace('_best','') not in config.ALG_NAMES:
        raise ValueError
    if 'ppo-v0-b(1,0)' in alg_name:
        hyper_param['b2'] = 0.0
    elif 'ppo-v0-b(1,.01)' in alg_name:
        hyper_param['b2'] = 0.01
    elif 'ppo-v0-b(1,.05)' in alg_name:
        hyper_param['b2'] = 0.05
    elif 'ppo-v0-b(1,.1)' in alg_name:
        hyper_param['b2'] = 0.1
    elif 'ppo-v0-b(1,.5)' in alg_name:
        hyper_param['b2'] = 0.5
    elif 'ppo-v0-b(1,1)' in alg_name:
        hyper_param['b2'] = 1
    elif 'ppo-v0-b(1,5)' in alg_name:
        hyper_param['b2'] = 5
    elif 'ppo-v0-b(1,10)' in alg_name:
        hyper_param['b2'] = 10
    elif 'ppo-v0-b(1,50)' in alg_name:
        hyper_param['b2'] = 50
    elif 'ppo-v0-b(1,100)' in alg_name:
        hyper_param['b2'] = 100
    elif 'ppo-v0-b(0,1)' in alg_name:
        # in fact b2 is 10 by default, b1 the first term is eliminated, so the final results approx. equals to b2 only
        hyper_param['b1'] = 0
    else:
        print(alg_name, 'use default reward hyperparameter', hyper_param)
    return hyper_param


def get_pop_size_default(dim):
    if dim <= 3:
        return 10
    elif 3 < dim <= 50:
        return 50
    elif dim > 50:
        return 100


def load_agent_data(env_name, prob_name, agent_names, alg_name):
    result_data = {}
    file_names = []
    for agent_name in agent_names:
        filename = agent_name + '-' + alg_name if agent_name in ['mtde-l2t', 'mtga-l2t'] else agent_name
        with open('data/' + env_name + '/' + prob_name + '/' + filename + '.pkl', 'rb') as f:
            result_data[agent_name] = pickle.load(f)
        file_names.append(filename)
    print(file_names)
    return result_data


def load_algorithm_data(env_name, prob_name, agent_name, alg_names):
    result_data = {}
    for alg_name in alg_names:
        filename = agent_name + '-' + alg_name
        with open('data/' + env_name + '/' + prob_name + '/' + filename + '.pkl', 'rb') as f:
            result_data[alg_name] = pickle.load(f)
    print(result_data.keys())
    return result_data


def load_sto_expdata(f_ids, i_ids, n_dims, alg_name='de_elit'):
    opt_exp = []
    for f_id, i_id in zip(f_ids, i_ids):
        with open(config.PROJECT_PATH + '/data/bbob_opt_exp/sto/f' + str(f_id) + '_i' + str(i_id) + '_d' +
                  str(n_dims) + '_' + alg_name + '.pkl', 'rb') as f:
            opt_exp.append(pickle.load(f))
    return opt_exp


X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var
