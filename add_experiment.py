import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import gymnasium as gym
import config
import numpy as np
from emto.agent import get_agent
import time
from emto.envs.task import set_timer, get_recorded_time


def get_agent_alias(agt_name):
    return agt_name.upper()


def compare_running_time():
    plt.rcParams.update({'font.size': 16})
    result = {}
    eval_times = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    base_solver = 'ga'
    agent_set = (config.DE_HUMAN_AGENT_NAMES + config.DE_BASE_AGENT_NAME
                 if base_solver == 'de' else config.GA_HUMAN_AGENT_NAMES + config.GA_BASE_AGENT_NAME)
    for agt_name in agent_set:
        print(agt_name)
        set_timer(1e-7)  # postpone the evaluation result return to increase and modify the evaluation cost
        agt = get_agent(agt_name, alg_name='ppo-v0_best' if base_solver == 'de' else 'ppo-v0-ga_best')
        env = gym.make('l2t_emto-v1', env_mode='test', problem_name='bbob-v1', base_solver_name=base_solver, max_gen=100)
        env_seed = 10086

        t0 = time.time()
        env.reset(seed=env_seed)
        agt.rollout(env, env_seed)
        # print(f"eval_time={get_recorded_time()}s")
        elapsed_time = time.time() - t0
        result[get_agent_alias(agt_name)] = elapsed_time - get_recorded_time()
        print(f"Overhead={result[get_agent_alias(agt_name)]}")
    # colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # agent_names = list(result.keys())
    # times = list(result.values())
    #
    from matplotlib.markers import MarkerStyle

    markers = MarkerStyle.filled_markers
    markers = markers[7:]

    plt.figure(figsize=(6, 6))
    for i, agt_name in enumerate(result.keys()):
        plt.plot(eval_times, result[agt_name] / (eval_times * env.max_gen * env.large_pop_size),
                 marker=markers[i], label=agt_name)
    plt.xlabel('Evaluation time of single solution (s)')
    plt.ylabel('Overhead / Evaluation time (log scale)')
    plt.axhline(y=0.1, color='k', linestyle='--', linewidth=2)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Comparison of Agent Running Time')
    plt.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"add_figs/runtime_{base_solver}.png", dpi=300)


def compare_rl_algos():
    from train_policy import pretrain_akt
    pretrain_options = dict(env_name='l2t_emto-v1', base_solver_name='de', alg_name='sac',
                            prob_name=f'bbob-train', num_proc=20)
    pretrain_akt(**pretrain_options)
    pretrain_options = dict(env_name='l2t_emto-v1', base_solver_name='de', alg_name='td3',
                            prob_name=f'bbob-train', num_proc=20)
    pretrain_akt(**pretrain_options)


def compare_jade():
    from train_policy import finetune_akt
    from evaluate_policy import evaluate_on_problem_set

    for i in range(9, 11):
        finetune_options = dict(env_name='l2t_emto-v1', base_solver_name='jade', alg_name='ppo-v0-jade',
                                prob_name=f'bbob-v{i}-train', num_proc=20)
        finetune_akt(**finetune_options)

    evaluate_options = dict(env_name='l2t_emto-v1', base_solver_name='jade', agent_names=['stjade', 'mtjade-l2t'],
                            alg_name='ppo-v0-jade_best', prob_names=[f"bbob-v{i}" for i in range(9, 11)], seed=0,
                            n_envs=100, n_cpus=20, retrained=True)
    evaluate_on_problem_set(**evaluate_options)


def run_mfea_rl():
    from evaluate_policy import evaluate_on_problem_set
    evaluate_on_problem_set(env_name='l2t_emto-v1', base_solver_name='ga', agent_names=['mfea-rl'], alg_name='ppo-v0',
                            prob_names=config.EVAL_PROBLEM_NAMES, seed=0, n_envs=100, n_cpus=20, retrained=False)


if __name__ == "__main__":
    # compare_running_time()
    run_mfea_rl()