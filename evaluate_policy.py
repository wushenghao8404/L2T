import os
import config
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
import numpy as np
import time
import ray
import pickle
import argparse
import emto
import emto.agent


@ray.remote
def rollout_remote(agt, env, env_seed):
    return agt.rollout(env, env_seed)


def get_evaluation_rollout_setting(prob_name):
    if prob_name[:4] == 'bbob':
        max_gen = 250
    elif prob_name[:3] == 'hpo':
        max_gen = 100
    elif prob_name[:3] == 'cec':
        max_gen = 500
    elif prob_name[:4] == 'cust':
        max_gen = 250
    else:
        raise ValueError
    rec_freq = 5
    return max_gen, rec_freq


def evaluate(env_name: str, base_solver_name: str, agent_names: list, alg_name: str, prob_name: str = 'bbob-v1',
             seed=0, n_runs=20, n_envs=100, n_cpus: int = 1, retrained=False):
    print(env_name, base_solver_name, agent_names, alg_name, prob_name, seed, n_runs, n_envs, n_cpus, retrained)
    res_dir = 'data/' + env_name + '/' + prob_name
    os.makedirs(res_dir, exist_ok=True)
    if n_cpus > 1:
        num_procs = n_cpus
        ray.shutdown()
        ray.init(num_cpus=num_procs)
    else:
        num_procs = 1
    max_gen, rec_freq = get_evaluation_rollout_setting(prob_name)
    env = gym.make(env_name,
                   env_mode='test',
                   problem_name=prob_name,
                   base_solver_name=base_solver_name,
                   max_gen=max_gen,
                   rec_freq=rec_freq,
                   env_kwargs=dict(alg_name=alg_name))
    result_data = {}
    for agent_name in agent_names:
        result_alg_data = {'y_final': np.full(shape=(n_envs, n_runs, env.n_tasks), fill_value=np.nan),
                           'y_trajectory': [],
                           'env_info': []}
        agent = emto.agent.get_agent(agent_name, alg_name=alg_name, retrained=retrained)
        result_data[agent_name] = {}
        episode_errors = []
        for env_id in range(n_envs):
            print('env_id',env_id)
            env_y_traj = []
            episode_errors.append(np.full((n_runs, env.n_tasks), fill_value=np.nan))
            if num_procs > 1:
                run_id = 0
                while run_id < n_runs:
                    batch_size = num_procs if run_id + num_procs <= n_runs else n_runs - run_id
                    remote_res = ray.get([rollout_remote.remote(agent, env, env_id + seed) for _ in range(batch_size)])
                    episode_errors[env_id][run_id: run_id+batch_size, :] = [single_res[0] for single_res in remote_res]
                    result_alg_data['y_final'][env_id, run_id: run_id+batch_size, :] = episode_errors[env_id][run_id: run_id+batch_size, :]
                    env_y_traj += [res[1] for res in remote_res]
                    env_info = remote_res[0][2]
                    run_id += batch_size
            else:
                for run_id in range(n_runs):
                    episode_errors[env_id][run_id, :], y_traj, env_info = agent.rollout(env, env_id + seed)
                    result_alg_data['y_final'][env_id, run_id, :] = episode_errors[env_id][run_id, :]
                    env_y_traj.append(y_traj)
            env_y_traj = np.array(env_y_traj)
            result_alg_data['y_trajectory'].append(env_y_traj)
            result_alg_data['env_info'].append(env_info)
            # print(env_y_traj.shape,len(result_alg_data['y_trajectory']),env_info)
        result_data[agent_name] = {'episode_errors': episode_errors}
        filename = agent_name + '-' + alg_name if agent_name in ['mtde-l2t', 'mtga-l2t'] else agent_name
        with open(res_dir + '/' + filename + '.pkl', 'wb') as f:
            pickle.dump(result_alg_data, f)
        print(f"{agent_name} policy: error:{np.mean(episode_errors):.2e} +/- {np.std(episode_errors):.2e}")
    return result_data


def evaluate_on_problem_set(env_name: str, base_solver_name: str, agent_names: list, alg_name: str, prob_names: list,
                            seed=0, n_runs=20, n_envs=100, n_cpus: int = 1, retrained=False):
    for prob_name in prob_names:
        evaluate(env_name=env_name, base_solver_name=base_solver_name, agent_names=agent_names, alg_name=alg_name,
                 prob_name=prob_name, seed=seed, n_runs=n_runs, n_envs=n_envs, n_cpus=n_cpus, retrained=retrained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", dest="env_name", default='l2t_emto-v1', type=str, help="specifies the environment")
    parser.add_argument("-p", dest="prob_name", default='bbob-v1', type=str,
                        help="specifies the problem set to be sample from in the environment")
    parser.add_argument("-b", dest="base_solver_name", default='de', type=str,
                        help="specifies base solver for self evolution")
    parser.add_argument("-a", dest="alg_name", default='ppo-v0', type=str,
                        help="specifies the rl algorithm for training agent")
    parser.add_argument("--agent", dest="agent_name", default='mtde-l2t', type=str,
                        help="specifies the rl algorithm for training agent")
    parser.add_argument("--n_cpus", dest="n_cpus", default=20, type=int, help="specifies the number of cpus")
    parser.add_argument("--single", action='store_true', help="specifies whether only evaluating single algorithm")
    parser.add_argument("--human", action='store_true', help="specifies whether to evaluate human designed algorithm")
    parser.add_argument("--all_probs", action='store_true', help="specifies whether to evaluate all problems at once")
    parser.add_argument("--retrain", action='store_true', help="specifies whether to evaluate retrained model")
    args = parser.parse_args()

    env_name = args.env_name
    prob_name = args.prob_name
    base_solver_name = args.base_solver_name
    alg_name = args.alg_name
    evaluate_single_policy = args.single
    evaluate_human_policy = args.human
    agent_name = args.agent_name
    n_envs = 100
    all_probs = args.all_probs
    retrained = args.retrain
    prob_names = [prob_name]
    if all_probs:
        prob_names = config.EVAL_PROBLEM_NAMES

    # # manual setting--------------------------------
    # env_name = 'l2t_emto-v8'
    # evaluate_single_policy = True
    # evaluate_human_policy = False
    # alg_name = 'ppo-v0'
    # prob_name = 'bbob-v1'
    # args.n_cpus = 20
    # # ----------------------------------------------

    print('problem sets to be evaluated:', prob_names)
    if alg_name not in config.ALG_NAMES and alg_name.replace('_best', '') not in config.ALG_NAMES:
        raise Exception(alg_name + ' is an invalid algorithm name')
    if agent_name not in (config.DE_LEARN_AGENT_NAMES + config.DE_HUMAN_AGENT_NAMES +
                          config.GA_LEARN_AGENT_NAMES + config.GA_HUMAN_AGENT_NAMES):
        raise Exception(agent_name + ' is an invalid agent name')
    print(alg_name)
    if base_solver_name == 'ga':
        if evaluate_human_policy:
            print('evaluate all human-designed agents:', config.GA_HUMAN_AGENT_NAMES)
            evaluate_on_problem_set(env_name, base_solver_name, config.GA_HUMAN_AGENT_NAMES, alg_name,
                                    prob_names=prob_names, seed=0, n_envs=n_envs, n_cpus=args.n_cpus,
                                    retrained=retrained)
        elif not evaluate_single_policy:
            print('evaluate all learnable agents:', config.GA_LEARN_AGENT_NAMES)
            evaluate_on_problem_set(env_name, base_solver_name, config.GA_LEARN_AGENT_NAMES, alg_name,
                                    prob_names=prob_names, seed=0, n_envs=n_envs, n_cpus=args.n_cpus,
                                    retrained=retrained)
        else:
            if '-ga' not in alg_name:
                raise Exception('base solver', base_solver_name, 'does not match the algorithm', alg_name)
            print('evaluate single algorithm: ' + agent_name + '-' + alg_name)
            if agent_name not in config.GA_LEARN_AGENT_NAMES:
                raise Exception(agent_name + ' is an invalid agent name')
            evaluate_on_problem_set(env_name, base_solver_name, [agent_name], alg_name, prob_names=prob_names, seed=0,
                                    n_envs=n_envs, n_cpus=args.n_cpus, retrained=retrained)
    elif base_solver_name == 'de':
        if evaluate_human_policy:
            print('evaluate all human-designed agents:', config.DE_HUMAN_AGENT_NAMES)
            evaluate_on_problem_set(env_name, base_solver_name, config.DE_HUMAN_AGENT_NAMES, alg_name,
                                    prob_names=prob_names, seed=0, n_envs=n_envs, n_cpus=args.n_cpus,
                                    retrained=retrained)
        elif not evaluate_single_policy:
            print('evaluate all learnable agents:', config.DE_LEARN_AGENT_NAMES)
            evaluate_on_problem_set(env_name, base_solver_name, config.DE_LEARN_AGENT_NAMES, alg_name,
                                    prob_names=prob_names, seed=0, n_envs=n_envs, n_cpus=args.n_cpus,
                                    retrained=retrained)
        else:
            print('evaluate single algorithm: ' + agent_name + '-' + alg_name)
            if agent_name not in config.DE_LEARN_AGENT_NAMES:
                raise Exception(agent_name + ' is an invalid agent name')
            evaluate_on_problem_set(env_name, base_solver_name, [agent_name], alg_name, prob_names=prob_names, seed=0,
                                    n_envs=n_envs, n_cpus=args.n_cpus, retrained=retrained)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
