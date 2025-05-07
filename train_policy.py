import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import time
import matplotlib.pyplot as plt
import emto
from emto.agent import get_agent_training_algo, get_agent_training_param
import argparse
import pickle
import os
import pandas as pd
import config
import shutil


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, alg_name: str,verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.alg_name = alg_name
        self.save_path = os.path.join(log_dir, self.alg_name + "_best")
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

        return True


def pretrain_akt(env_name, base_solver_name, alg_name, prob_name, total_timesteps: int = 5e6, num_proc: int = 1):
    train_mode = 'pretrain'
    log_dir = 'model/' + env_name + '/' + train_mode + '/' + alg_name
    mdl_dir = 'model/' + env_name + '/' + train_mode + '/' + alg_name
    os.makedirs(log_dir, exist_ok=True)

    # call back setting

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, alg_name=alg_name)

    t0 = time.time()
    if num_proc == 1:
        env = gym.make(env_name,
                       env_mode='train',
                       problem_name=prob_name,
                       base_solver_name=base_solver_name,
                       max_gen=100,
                       rec_freq=5,
                       env_kwargs=dict(alg_name=alg_name))
        monitored_env = Monitor(env, filename=mdl_dir + '/' + alg_name)
    else:
        vec_env = make_vec_env(env_name,
                               n_envs=num_proc,
                               vec_env_cls=SubprocVecEnv,
                               env_kwargs=dict(env_mode='train',
                                               problem_name=prob_name,
                                               base_solver_name=base_solver_name,
                                               max_gen=100,
                                               rec_freq=5,
                                               render_mode=None,
                                               env_kwargs=dict(alg_name=alg_name)))

        monitored_env = VecMonitor(vec_env, filename=mdl_dir + '/' + alg_name)

    rl_algo_cls = get_agent_training_algo(alg_name)
    algo_params = get_agent_training_param(alg_name)
    model = rl_algo_cls("MlpPolicy", monitored_env, policy_kwargs={}, verbose=1, **algo_params)
    print(model.policy)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print('Consumed time:', time.time() - t0)

    data = pd.read_csv(mdl_dir + '/' + alg_name + '.monitor.csv', header=1)
    episode_rewards = data.loc[:, 'r'].values
    episode_lengths = data.loc[:, 'l'].values
    episode_times = data.loc[:, 't'].values
    time_steps = np.cumsum(episode_lengths)
    algo_training_data = {'episode_rewards': episode_rewards,
                          'episode_lengths': episode_lengths,
                          'episode_times': episode_times,
                          'time_steps': time_steps}
    with open(mdl_dir + '/' + alg_name + '_training_data.pkl', 'wb') as f:
        pickle.dump(algo_training_data, f)
    plt.plot(time_steps[::100], episode_rewards[::100], c='b')
    plt.xlabel('time steps')
    plt.ylabel('episode reward')
    plt.title(alg_name + ' training curve on ' + env_name + ' environment')
    plt.savefig(mdl_dir + '/' + alg_name + '.png')
    model.save(mdl_dir + '/' + alg_name)


def finetune_akt(env_name: str, base_solver_name: str, alg_name: str, prob_name: str, num_proc: int = 1, num_run=1):
    # if alg_name not in ['ppo-v0-wotr', 'ppo-v0-ft', 'ppo-v0-ga-wotr', 'ppo-v0-ga-ft']:
    #     raise Exception('For retrain mode, algorithm name should belong to',
    #                     ['ppo-v0-wotr', 'ppo-v0-ft', 'ppo-v0-ga-wotr', 'ppo-v0-ga-ft'])

    if '-ft' in alg_name:
        pretrained_mdl_path = '/public2/home/wushenghao/project/L2T/model/l2t_emto-v1/pretrain/' + alg_name.replace(
            '-ft', '') + '_best'

    # if base_solver_name == 'de' and alg_name not in ['ppo-v0-wotr', 'ppo-v0-ft', ]:
    #     raise Exception('base solver',base_solver_name,'should match algorithms in',
    #                     ['ppo-v0-wotr', 'ppo-v0-ft'])
    # if base_solver_name == 'ga' and alg_name not in ['ppo-v0-ga-wotr', 'ppo-v0-ga-ft', ]:
    #     raise Exception('base solver',base_solver_name,'should match algorithms in',
    #                     ['ppo-v0-ga-wotr', 'ppo-v0-ga-ft'])

    train_mode = 'retrain'

    t0 = time.time()
    for run in range(num_run):
        log_dir = 'model/' + env_name + '/' + train_mode + '/' + prob_name + '/' + alg_name + '/run' + str(run)
        mdl_dir = 'model/' + env_name + '/' + train_mode + '/' + prob_name + '/' + alg_name + '/run' + str(run)
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, alg_name=alg_name)
        if num_proc == 1:
            env = gym.make(env_name,
                           env_mode='train',
                           problem_name=prob_name,
                           base_solver_name=base_solver_name,
                           max_gen=100,
                           rec_freq=5,
                           env_kwargs=dict(alg_name=alg_name))
            monitored_env = Monitor(env, filename=mdl_dir + '/' + alg_name)
        else:
            vec_env = make_vec_env(env_name, n_envs=num_proc, vec_env_cls=SubprocVecEnv,
                                   env_kwargs=dict(env_mode='train',
                                                   problem_name=prob_name,
                                                   base_solver_name=base_solver_name,
                                                   max_gen=100,
                                                   rec_freq=5,
                                                   render_mode=None,
                                                   env_kwargs=dict(alg_name=alg_name)))
            monitored_env = VecMonitor(vec_env, filename=mdl_dir + '/' + alg_name)
        # if alg_name == 'ppo-v0-wotr' or alg_name == 'ppo-v0-ga-wotr':
        #     model = PPO("MlpPolicy", monitored_env, policy_kwargs={}, verbose=1)

        if alg_name == 'ppo-v0-ft' or alg_name == 'ppo-v0-ga-ft':
            rl_algo_cls = get_agent_training_algo(alg_name)
            model = rl_algo_cls.load(pretrained_mdl_path, monitored_env, policy_kwargs={}, verbose=1)
        else:
            rl_algo_cls = get_agent_training_algo(alg_name)
            model = rl_algo_cls("MlpPolicy", monitored_env, policy_kwargs={}, verbose=1)

        print(model.policy)
        model.learn(total_timesteps=int(2e6), callback=callback)  # CHANGE to 2e6
        print('Consumed time:', time.time() - t0)
        model.save(mdl_dir + '/' + alg_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", dest="env_name", default='l2t_emto-v1', type=str, help="specifies the environment")
    parser.add_argument("-p", dest="prob_name", default='bbob-train', type=str,
                        help="specifies the problem set to be sample from in the environment")
    parser.add_argument("-b", dest="base_solver_name", default='de', type=str,
                        help="specifies base solver for self evolution")
    parser.add_argument("-a", dest="alg_name", default='ppo-v0', type=str,
                        help="specifies the rl algorithm for training agent")
    parser.add_argument("--n_step", dest="n_timesteps", default=5e6, type=int,
                        help="specifies the number of time steps")
    parser.add_argument("--n_env", dest="n_envs", default=20, type=int,
                        help="specifies the number of parallel environments")
    parser.add_argument("--n_run", dest='n_runs', default=1, help="specifies the number of independent training")
    parser.add_argument("--retrain", action='store_true', help="specifies whether to retrain model")
    args = parser.parse_args()

    env_name = args.env_name
    prob_name = args.prob_name
    base_solver_name = args.base_solver_name
    alg_name = args.alg_name
    n_timesteps = args.n_timesteps
    n_envs = args.n_envs
    n_runs = args.n_runs
    do_retrain = args.retrain

    if alg_name not in config.ALG_NAMES:
        raise Exception('Invalid algorithm name')

    if not do_retrain:
        pretrain_akt(env_name=env_name, base_solver_name=base_solver_name, alg_name=alg_name,
                     prob_name=prob_name, total_timesteps=n_timesteps, num_proc=n_envs)
    else:
        finetune_akt(env_name=env_name, base_solver_name=base_solver_name, alg_name=alg_name,
                     prob_name=prob_name, num_proc=n_envs, num_run=n_runs)


if __name__ == "__main__":
    main()