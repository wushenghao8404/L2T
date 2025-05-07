import gymnasium as gym
from copy import deepcopy
import numpy as np
from typing import Any, SupportsFloat, TypeVar, Optional
from gymnasium import spaces
import emto.evol_optimizer as evopt
import emto.modifier
from scipy.stats import multivariate_normal
import config
import utils
import scipy
from problems.multitask_problem import get_multiple_tasks

ObsType = TypeVar("ObsType")
RenderFrame = TypeVar("RenderFrame")
GLOBAL_CONFIG = {'env_name': 'l2t_emto-v1',
                 'n_tasks': 2,
                 'n_dims': 10,}


class MultiTaskOptEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    default_action_dict = {'a1': 0.5,  # controls KT ratio
                           'a2': 1.0,  # controls the truncation ratio of selecting source elite solutions
                           'a3': 0.0,  # controls base vector transfer ratio
                           'a4': 0.5,  # controls trial vector transfer ratio, default value is the scaling factor in DE
                           'a5': 0.0,  # controls differential vector transfer ratio
                           'a6': 0.5,  # controls crossover vector transfer ratio, default value same as cr rate in DE
                           }
    reward_param_dict = {'b1': 1.0,  # controls short-term reward ratio
                         'b2': 10.0, # controls kt reward ratio
                         'b3': 1.0,  # controls long-term reward ratio
                         }

    def __init__(self,
                 env_mode: str,
                 render_mode: Optional[str] = None,
                 problem_name='bbob-v1',
                 base_solver_name='de',
                 max_gen=100,
                 rec_freq=5,
                 env_kwargs=None,
                 ):
        if env_mode not in config.ENV_MODES:
            raise Exception('Invalid environment mode')

        if env_mode == 'train' and 'train' not in problem_name:
            raise Exception('For training mode, a training problem set name should be provided')

        self.env_mode = env_mode
        self.render_mode = render_mode
        self.problem_name = problem_name
        self.env_name = GLOBAL_CONFIG['env_name']
        self.n_tasks = GLOBAL_CONFIG['n_tasks']
        self.n_dims = GLOBAL_CONFIG['n_dims']
        self._seed = 168  # random seed for f sampling is 168, yi lu fa!
        self.np_gen = np.random.RandomState(self._seed)
        self.env_kwargs = env_kwargs or {}

        if 'alg_name' not in self.env_kwargs.keys():
            print('algorithm name is not specified, by default use ppo-v0')
            self.env_kwargs['alg_name'] = 'ppo-v0'

        if base_solver_name not in config.BASE_SOLVER_NAMES:
            raise Exception('Invalid environment mode')

        if base_solver_name != 'de':
            if self.env_kwargs['alg_name'] != f'ppo-v0-{base_solver_name}':
                print(f"Using base solver {base_solver_name}, redirect alg_name={self.env_kwargs['alg_name']} to "
                      f"alg_name=ppo-v0-{base_solver_name}")
                self.env_kwargs['alg_name'] = f'ppo-v0-{base_solver_name}'

        self.base_solver_name = base_solver_name
        self.mod = emto.modifier.get_modifier(self)
        self.reward_param_dict = utils.load_hyper_param(self.env_kwargs['alg_name'], self.reward_param_dict)

        # parameter tuning for hpo !!!!!!!!!!! IMPORTANT !!!!!!!!!!!
        if self.problem_name[:3] == 'hpo' and self.base_solver_name == 'de':
            self.reward_param_dict['b2'] = 0.01  # currently b2=0.01 seems to work on hpo-svm, hpo-xgboost, hpo-fcnet

        # instantiate problem
        self.tasks, self.task_info = get_multiple_tasks(self.problem_name, self.np_gen, self.n_tasks, self.n_dims)
        self.n_dims = np.max([task.dim for task in self.tasks])

        # problem setting
        self.xlb = np.zeros(self.n_dims)
        self.xub = np.ones(self.n_dims)

        # algorithm setting
        self.pop_size_per_task = utils.get_pop_size_default(self.n_dims)
        self.large_pop_size = self.n_tasks * self.pop_size_per_task
        self.lh = scipy.stats.qmc.LatinHypercube(self.n_dims, seed=1)  # random seed for latin hypercube is 1
        # init pop set size may have certain effects, to be investigated !!!!!!!!!!! IMPORTANT !!!!!!!!!!!
        self.init_pops = [self.lh.random(self.pop_size_per_task) for _ in range(config.NUM_PRE_INIT_POPS)]

        # environment observation and action setting
        self.action_names = utils.load_actions(self.env_kwargs['alg_name'])
        self.act_dim_per_task = len(self.action_names)
        self.common_features, self.task_features = utils.load_obs_features(self.env_kwargs['alg_name'])
        self.n_common_features = len(self.common_features)
        self.n_task_features = len(self.task_features)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_tasks, self.act_dim_per_task), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(self.n_common_features + self.n_task_features * self.n_tasks
                                                   + self.act_dim_per_task * self.n_tasks,),
                                            dtype=np.float64)

        # environment rollout episode setting
        self.task_ids = np.arange(self.n_tasks)
        self.max_gen = max_gen
        self.rec_freq = rec_freq
        self.task_max_nfe = self.max_gen * self.pop_size_per_task
        self.rec_nfe = self.rec_freq * self.pop_size_per_task
        self.max_nfe = self.task_max_nfe * self.n_tasks

        # modify environment if necessary
        self.mod.modified_env_init(self)

    def get_obs_common_features(self):
        obs_common_feats = []
        for feat_name in self.common_features:
            if feat_name == 'oc1':
                obs_common_feats.append(self.gen / self.max_gen)
            elif feat_name == 'oc2':
                obs_common_feats.append(np.sqrt(np.sum((self.gbx[0] - self.gbx[1]) ** 2)) / np.sqrt(self.n_dims))
            elif feat_name == 'oc3':
                obs_common_feats.append(np.sqrt(np.sum((self.pop_mu[0] - self.pop_mu[1]) ** 2)) / np.sqrt(self.n_dims))
            elif feat_name == 'oc4':
                obs_common_feats.append(np.sqrt(np.sum((self.pop_sig[0] - self.pop_sig[1]) ** 2)) / np.sqrt(0.5 * self.n_dims))
            elif feat_name == 'oc5':
                dv = [(self.gbx[task_id] - self.pop_mu[task_id]) for task_id in range(self.n_tasks)]
                dv_cos = np.sum(dv[0] * dv[1]) / (np.sqrt(np.sum(dv[0] ** 2)) + 1e-10) / (np.sqrt(np.sum(dv[1] ** 2)) + 1e-10)
                obs_common_feats.append((dv_cos + 1) / 2)
            else:
                raise Exception('Unexpected common feature name')
        return np.array(obs_common_feats)

    def get_obs_task_features(self):
        obs_task_feats = []
        for feat_name in self.task_features:
            if feat_name == 'oi1':
                # number of stagnated generations
                obs_task_feats += list(self.n_stag / self.max_gen)
            elif feat_name == 'oi2':
                # whether the best-found solution of last generation is improved
                obs_task_feats += list(self.gbx_improved)
            elif feat_name == 'oi3':
                # knowledge transfer quality
                obs_task_feats += list(self.tr_quality)
            elif feat_name == 'oi4':
                # average deviation of population
                obs_task_feats += [np.mean(pop_sig) for pop_sig in self.pop_sig]
            elif feat_name == 'oi5':
                # self evolution quality
                obs_task_feats += list(self.se_quality)
            elif feat_name == 'oi6':
                # knowledge transfer diversity
                obs_task_feats += list(self.tr_diversity)
            elif feat_name == 'oi7':
                # self evolution diversity
                obs_task_feats += list(self.se_diversity)
            else:
                raise Exception('Unexpected task feature name')
        return np.array(obs_task_feats)

    def get_kt_param_from_action(self, sub_action):
        assert sub_action.ndim == 1
        assert sub_action.shape[0] == len(self.action_names)
        action_dict = deepcopy(self.default_action_dict)
        for i in range(sub_action.shape[0]):
            action_dict[self.action_names[i]] = sub_action[i]
        a1 = action_dict['a1']
        a2 = action_dict['a2']
        a3 = action_dict['a3']
        a4 = action_dict['a4']
        a5 = action_dict['a5']
        a6 = action_dict['a6']
        return a1, a2, a3, a4, a5, a6

    def get_env_info(self)->dict:
        return {'env_mode': self.env_mode,
                'base_solver_name': self.base_solver_name,
                'prob_name': self.problem_name,
                'task_fids': self.cur_fids,
                'task_iids': self.cur_iids,
                'task_fopts': np.array([task.f.fopt for task in self.tasks]),
                'task_xopts': [task.f.xopt for task in self.tasks],
                'env_seed': self._seed,
                'evol_proc_params': [ev_proc.get_alg_setting() for ev_proc in self.ev_procs],
                'max_gen': self.max_gen,
                'max_nfe': self.max_nfe,
                'task_max_nfe': self.task_max_nfe,
                'rec_nfe': self.rec_nfe}

    def get_obs(self):
        obs, flag = self.mod.modified_get_obs(self)
        if flag:
            return obs
        else:
            # return np.concatenate((self.get_obs_common_features(), self.get_obs_task_features(), self.cur_action.flatten()), axis=0)
            return np.concatenate(
                (self.get_obs_common_features(), self.get_obs_task_features(), self.cur_action.T.flatten()),
                axis=0)  # original version

    def get_evaluation_cpu_time(self):
        import time
        t0 = time.time()
        n_sample = 100
        X = np.random.rand(n_sample, self.n_dims)
        for task_id in range(self.n_tasks):
            Y = self.tasks[task_id](X)
        return (time.time() - t0) / (self.n_tasks * n_sample)

    def reset_tmp_var(self):
        # temporary variables along the search
        self.reward = np.zeros(self.n_tasks)
        self.obs = None
        self.done = np.full(shape=(self.n_tasks,), fill_value=False)
        self.truncate = False
        self.offspring_pops = np.full(shape=(self.n_tasks, self.pop_size_per_task, self.n_dims), fill_value=np.nan)
        self.offspring_fits = np.full(shape=(self.n_tasks, self.pop_size_per_task), fill_value=np.nan)
        self.offspring_scores = np.full(shape=(self.n_tasks, self.pop_size_per_task), fill_value=np.nan)
        self.offspring_disps = np.full(shape=(self.n_tasks, self.pop_size_per_task), fill_value=np.nan) # normalized pdfs
        self.transfer_masks = np.full(shape=(self.n_tasks, self.pop_size_per_task), fill_value=False)  # mark the position with knowledge transferred

    def reset(self, seed: int = None, options: dict = None) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            self.np_gen = np.random.RandomState(self._seed)

        # instantiate problem
        self.tasks, self.task_info = get_multiple_tasks(self.problem_name, self.np_gen, self.n_tasks, self.n_dims)
        self.n_dims = np.max([task.dim for task in self.tasks])
        self.cur_fids = self.task_info['cur_fids']
        self.cur_iids = self.task_info['cur_iids']

        self.mod.modified_init_add_var(self)

        # initialize task-specific information and extracted features
        self.pop_mu = np.full(shape=(self.n_tasks, self.n_dims), fill_value=np.nan)
        self.pop_sig = np.full(shape=(self.n_tasks, self.n_dims), fill_value=np.nan)
        self.mvn_dists = [None] * self.n_tasks
        self.n_stag = np.zeros(self.n_tasks)
        self.gbx_improved = np.zeros(self.n_tasks)
        self.tr_quality = np.zeros(self.n_tasks)
        self.se_quality = np.zeros(self.n_tasks)
        self.tr_diversity = np.zeros(self.n_tasks)
        self.se_diversity = np.zeros(self.n_tasks)

        # observation features (common)
        self.cur_action = np.zeros(self.action_space.shape)
        self.done = [False] * self.n_tasks
        self.gen = 0

        # instantiate algorithm
        self.setup_base_solver_procs()

        if self.env_mode == 'train':
            if not self.mod.modified_pop_init(self):
                self.assigned_init_pop_ids = np.random.permutation(self.n_tasks)  # !!!!!!IMPORTANT!!!!!!!!
                init_pops = [deepcopy(self.init_pops[self.assigned_init_pop_ids[task_id]]) for task_id in range(self.n_tasks)]
                for task_id, ev_proc in enumerate(self.ev_procs):
                    ev_proc.setup(self.tasks[task_id], init_pop=init_pops[task_id])
        elif self.env_mode == 'test':
            if not self.mod.modified_pop_init(self):
                if 'ppo-v0' in self.env_kwargs['alg_name']:
                    self.assigned_init_pop_ids = np.random.permutation(self.n_tasks)  # !!!!!!IMPORTANT!!!!!!!!
                    init_pops = [deepcopy(self.init_pops[self.assigned_init_pop_ids[task_id]]) for task_id in range(self.n_tasks)]
                    for task_id, ev_proc in enumerate(self.ev_procs):
                        ev_proc.setup(self.tasks[task_id], init_pop=init_pops[task_id])
                else:
                    for task_id, ev_proc in enumerate(self.ev_procs):
                        ev_proc.setup(self.tasks[task_id])
        else:
            raise Exception('The environment mode ' + self.env_mode + ' is invalid')
        self.gbx = []
        self.gby = []

        # initialize intra-task stats
        for task_id in range(len(self.tasks)):
            # X = np.random.uniform(self.xlb, self.xub, size=(self.pop_size_per_task, self.n_dims))
            self.gbx_improved[task_id] = 0
            self.n_stag[task_id] = 0
            self.tr_quality[task_id] = 0
            self.se_quality[task_id] = 0
            self.tr_diversity[task_id] = 0
            self.se_diversity[task_id] = 0
            self.pop_mu[task_id] = np.mean(self.ev_procs[task_id].cur_pop, axis=0)
            self.pop_sig[task_id] = np.std(self.ev_procs[task_id].cur_pop, axis=0)
            self.mvn_dists[task_id] = multivariate_normal(mean=self.pop_mu[task_id], cov=self.pop_sig[task_id] ** 2 + 1e-10)
            gbx, gby = self.ev_procs[task_id].memory.get('gbest')
            self.gbx.append(deepcopy(gbx))
            self.gby.append(deepcopy(gby))

        # initialize environment variable
        self.rewarded_for_done = np.array([False] * self.n_tasks)
        self.init_errors = np.array([task.get_error() for task in self.tasks])
        obs = self.get_obs()
        return obs, self.get_env_info()

    def update_task_stats(self, task_id):
        # update intra-task stats
        self.gbx_improved[task_id] = self.ev_procs[task_id].memory.get('gbx_improved')
        self.n_stag[task_id] = self.ev_procs[task_id].memory.get('n_stag')
        tr_mask = self.transfer_masks[task_id]
        se_mask = ~self.transfer_masks[task_id]
        self.tr_quality[task_id] = np.mean(self.offspring_scores[task_id][tr_mask]) if tr_mask.any() else 0
        self.se_quality[task_id] = np.mean(self.offspring_scores[task_id][se_mask]) if se_mask.any() else 0
        self.tr_diversity[task_id] = np.mean(self.offspring_disps[task_id][tr_mask]) if tr_mask.any() else 0
        self.se_diversity[task_id] = np.mean(self.offspring_disps[task_id][se_mask]) if se_mask.any() else 0
        self.pop_mu[task_id] = np.mean(self.ev_procs[task_id].cur_pop, axis=0)
        self.pop_sig[task_id] = np.std(self.ev_procs[task_id].cur_pop, axis=0)
        self.mvn_dists[task_id] = multivariate_normal(mean=self.pop_mu[task_id], cov=self.pop_sig[task_id] ** 2 + 1e-10)
        self.gbx[task_id], self.gby[task_id] = self.ev_procs[task_id].memory.get('gbest')

    def update_reward(self, task_id):
        # update reward
        b1 = self.reward_param_dict['b1']
        b2 = self.reward_param_dict['b2']
        self.reward[task_id] += - b1 * self.tasks[task_id].get_error() / self.init_errors[task_id]  # !!!!!!IMPORTANT!!!!!!!!
        if self.transfer_masks[task_id].any() and (~self.transfer_masks[task_id]).any():
            self.reward[task_id] += b2 * (np.mean(self.offspring_scores[task_id][self.transfer_masks[task_id]]) -
                                          np.mean(self.offspring_scores[task_id][~self.transfer_masks[task_id]]))
        self.done[task_id] = self.tasks[task_id].has_reach_target()
        if self.done[task_id] and not self.rewarded_for_done[task_id]:
            self.reward[task_id] += self.max_gen
            self.rewarded_for_done[task_id] = True

    def update_step_stats(self,):
        # pack next population into obs
        reward = np.sum(self.reward)
        obs = self.get_obs()
        done = np.sum(self.done) == self.n_tasks
        self.gen += 1
        truncated = self.gen >= self.max_gen
        return obs, reward, done, truncated, {}

    def setup_base_solver_procs(self):
        # instantiate algorithm
        if self.base_solver_name == 'de':
            self.ev_procs = [evopt.DiffEvol(self.n_dims, self.pop_size_per_task) for _ in range(self.n_tasks)]
        elif self.base_solver_name == 'jade':
            self.ev_procs = [evopt.SelfAdaptiveDiffEVol(self.n_dims, self.pop_size_per_task)
                             for _ in range(self.n_tasks)]
        elif self.base_solver_name == 'ga':
            # use GA as base solver
            self.ev_procs = [evopt.GeneAlgo(self.n_dims, self.pop_size_per_task) for _ in range(self.n_tasks)]
        else:
            raise ValueError

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        flag, obs, reward, done, truncated, info = self.mod.modified_step(self, action)
        # print('gen', self.gen)
        if flag:
            return obs, reward, done, truncated, info
        else:
            self.cur_action = action
            self.reset_tmp_var()
            for task_id in range(self.n_tasks):
                # sequentially evolve tasks in a generation
                self.evolve(task_id, self.cur_action[task_id, :])
                self.update_task_stats(task_id)
                if self.env_mode == 'train':
                    self.update_reward(task_id)
            obs, reward, done, truncated, info = self.update_step_stats()
            return obs, reward, done, truncated, info

    def evolve(self, task_id, sub_action):
        other_task_ids = np.r_[self.task_ids[:task_id], self.task_ids[task_id + 1:]]
        s_task_id = np.random.choice(other_task_ids)
        # retrieve the action values of knowledge transfer decision
        a1, a2, a3, a4, a5, a6 = self.get_kt_param_from_action(sub_action)

        pop_t = self.ev_procs[task_id].cur_pop
        pop_s = self.ev_procs[s_task_id].cur_pop
        sorted_id_s = np.argsort(self.ev_procs[s_task_id].cur_fit)

        # pseudo-random for deciding whether to transfer
        n_tr_sol = int(np.ceil(a1 * self.pop_size_per_task * 0.5))
        r_tr_ = np.random.permutation(self.pop_size_per_task)
        self.transfer_masks[task_id, r_tr_[:n_tr_sol]] = True

        # sample by assigned solver
        self.offspring_pops[task_id] = self.ev_procs[task_id].sample()

        for i in range(self.pop_size_per_task):
            if self.transfer_masks[task_id, i]:
                # random selection of the transferred base vector
                rs_b = sorted_id_s[np.random.choice(np.arange(max(int(self.pop_size_per_task * a2), 1)))]

                # random permutation for sampling indices of the transferred differential vector
                rs_d = np.random.permutation(self.pop_size_per_task)
                rs_d = rs_d[rs_d != rs_b]  # delete base vector index to keep source index distinct
                rt_ = np.random.permutation(self.pop_size_per_task)

                # replace base population with transferred individuals to obtain offspring population
                self.offspring_pops[task_id][i] = (
                        (1 - a3) * pop_t[rt_[0]] + a3 * pop_s[rs_b] +
                        a4 * ((1 - a5) * (pop_t[rt_[1]] - pop_t[rt_[2]]) + a5 * (pop_s[rs_d[0]] - pop_s[rs_d[1]]))
                                                   )

                # sample random numbers required for binomial crossover-based knowledge transfer
                mask_binocr = np.random.binomial(1, a6, self.n_dims)  # for binomial crossover
                mask_binocr[np.random.randint(self.n_dims)] = 1
                self.offspring_pops[task_id][i] = mask_binocr * self.offspring_pops[task_id][i] + \
                                                  (1 - mask_binocr) * pop_t[i]

        # clip the population to satisfy box constraint
        self.offspring_pops[task_id] = np.clip(self.offspring_pops[task_id], self.xlb, self.xub)

        # fitness evaluation
        self.offspring_fits[task_id] = self.tasks[task_id](self.offspring_pops[task_id])

        # calculate the scores of offspring relative to the parental population
        self.offspring_scores[task_id] = np.array([np.sum(sol_fit < self.ev_procs[task_id].cur_fit)
                                                   / self.pop_size_per_task for sol_fit in self.offspring_fits[task_id]])

        # calculate the dispersion of offspring relative to the parental population
        self.offspring_disps[task_id] = 1 - self.mvn_dists[task_id].pdf(self.offspring_pops[task_id]) / self.mvn_dists[
                                        task_id].pdf(self.pop_mu[task_id])

        # update population by selection
        self.ev_procs[task_id].update(self.offspring_pops[task_id], self.offspring_fits[task_id])

    def close(self):
        return


if __name__ == '__main__':
    seed = 1000
    # env = gym.make('emto:l2t_toy-v0', render_mode="human")
    env = gym.make('l2t_emto-v1',
                   env_mode='train',
                   problem_name='hpo-xgboost-train',
                   base_solver_name='de',
                   env_kwargs=dict(alg_name='ppo-v0'),
                   )
    obs, _ = env.reset(seed=seed)
    print(env.action_space)
    print(env.observation_space)
    print(obs)
    import time

    t0 = time.time()
    episode_errors = []
    n_episodes = 0
    while n_episodes < 100:
        action = env.action_space.sample()
        # print(action)
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        # print((obs>1).any(),(obs<0).any())

        # env resets automatically
        if done or truncated:
            n_episodes += 1
            print('ep', n_episodes)
            print('done',done)
            episode_errors.append(np.array([task.get_error() for task in env.tasks]))
            print(np.array(env.gby) - np.array([f.f.fopt for f in env.tasks]))
            ep_reward = 0.0
            obs, _ = env.reset(seed=seed + n_episodes)

    env.close()
    episode_errors = np.array(episode_errors)
    print(f"Random policy: error:{np.mean(episode_errors):.2e} +/- {np.std(episode_errors):.2e}")
