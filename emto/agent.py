import numpy as np
from copy import deepcopy
from stable_baselines3 import PPO, SAC, TD3
from emto.cmp_algs.de import MKTDE, MFDE, MTDE, EMTEA, AEMTO, MTDE_AD
from emto.cmp_algs.ga import ATMFEA, GMFEA, MFEA, MFEA2, MFEA_AKT, MTEA_AD, MFEA_RL
import config


def get_agent_training_algo(alg_name):
    if 'ppo' in alg_name:
        return PPO
    elif 'sac' in alg_name:
        return SAC
    elif 'td3' in alg_name:
        return TD3
    else:
        raise NotImplementedError


def get_agent_training_param(alg_name):
    if 'ppo' in alg_name:
        return {}
    elif 'sac' in alg_name:
        return {}
    elif 'td3' in alg_name:
        return dict(train_freq=5)
    else:
        raise NotImplementedError


def get_agent(agent_name, alg_name=None, retrained=False):
    if agent_name in config.DE_HUMAN_AGENT_NAMES or agent_name in config.GA_HUMAN_AGENT_NAMES:
        return HandcraftedAgent(agent_name)
    elif agent_name in config.DE_LEARN_AGENT_NAMES or agent_name in config.GA_LEARN_AGENT_NAMES:
        return LearnableAgent(agent_name, alg_name, retrained)
    else:
        raise Exception('Unexpected agent name.')


class BaseAgent:
    def __init__(self, agent_name,):
        self.agent_name = agent_name
        self.model = None

    def load_model(self, env):
        raise NotImplementedError

    def predict(self, obs):
        raise NotImplementedError

    def rollout(self, env, env_seed):
        raise NotImplementedError


class HandcraftedAgent(BaseAgent):
    def __init__(self, agent_name,):
        super().__init__(agent_name)

    def load_model(self, env):
        # check env and agent consistency
        if env.env_name in config.GA_ENV_NAMES and self.agent_name not in config.GA_HUMAN_AGENT_NAMES:
            raise ValueError
        if env.env_name in config.DE_ENV_NAMES and self.agent_name not in config.DE_HUMAN_AGENT_NAMES:
            raise ValueError
        if env.env_name in config.BASE_ENV_NAME and self.agent_name not in (config.DE_HUMAN_AGENT_NAMES + config.GA_HUMAN_AGENT_NAMES):
            raise ValueError
        if env.base_solver_name =='ga' and self.agent_name not in config.GA_HUMAN_AGENT_NAMES:
            raise ValueError
        if env.base_solver_name =='de' and self.agent_name not in config.DE_HUMAN_AGENT_NAMES:
            raise ValueError
        if self.agent_name == 'atmfea':
            self.model = ATMFEA.ATMFEA(env)
        elif self.agent_name == 'gmfea':
            self.model = GMFEA.GMFEA(env)
        elif self.agent_name == 'mfea':
            self.model = MFEA.MFEA(env)
        elif self.agent_name == 'mfea2':
            self.model = MFEA2.MFEA2(env)
        elif self.agent_name == 'mfea-akt':
            self.model = MFEA_AKT.MFEA_AKT(env)
        elif self.agent_name == 'mtea-ad':
            self.model = MTEA_AD.MTEA_AD(env)
        elif self.agent_name == 'mfea-rl':
            self.model = MFEA_RL.MFEA_RL(env)
        elif self.agent_name == 'mktde':
            self.model = MKTDE.MKTDE(env)
        elif self.agent_name == 'mfde':
            self.model = MFDE.MFDE(env)
        elif self.agent_name == 'mtde-b':
            self.model = MTDE.MTDE_Base(env)
        elif self.agent_name == 'mtde-ea':
            self.model = EMTEA.EMTEA(env)
        elif self.agent_name == 'aemto':
            self.model = AEMTO.AEMTO(env)
        elif self.agent_name == 'mtde-ad':
            self.model = MTDE_AD.MTDE_AD(env)
        else:
            raise Exception('Unexpected agent name')

    def rollout(self, env, env_seed):
        print('running on',env.env_name,'problem',env.problem_name,'with env_seed',env_seed,'by',self.agent_name)
        self.load_model(env)
        return self.model.run(env_seed)


class LearnableAgent(BaseAgent):
    def __init__(self, agent_name, alg_name, retrained):
        super().__init__(agent_name)
        self.alg_name = alg_name  # algorithm for learning the policy, e.g., PPO
        self.retrained = retrained
        self.run_id = 0

    def predict(self, obs):
        return self.model(obs)

    def rollout(self, env, env_seed):
        print('running on', env.env_name, 'problem', env.problem_name,
              'with env_seed', env_seed, 'by', self.agent_name, '-', self.alg_name)
        self.load_model(env)
        obs, env_info = env.reset(seed=env_seed)
        actions = []
        states = []
        for _ in range(env.max_gen):
            action = self.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action.T.flatten())
            states.append(obs)
        y_trajectory = []
        y_final = np.array([np.min(task._y_trajectory[:env.task_max_nfe]) - task.f.fopt for task in env.tasks])
        for task_id in range(env.n_tasks):
            task_y_trajectory = np.array(env.tasks[task_id]._y_trajectory) - env.tasks[task_id].f.fopt
            for i in range(1, len(task_y_trajectory)):
                task_y_trajectory[i] = np.min([task_y_trajectory[i], task_y_trajectory[i - 1]])
            y_trajectory.append(task_y_trajectory[:env.task_max_nfe:env.rec_nfe])
        y_trajectory = np.array(y_trajectory).T
        actions = np.array(actions)
        states = np.array(states)
        state_stat = {'mean': np.mean(states, axis=0), 'std': np.std(states, axis=0)}
        action_stat = {'mean': np.mean(actions, axis=0), 'std': np.std(actions, axis=0)}
        return y_final, y_trajectory, env_info, state_stat, action_stat

    def load_model(self, env):
        # check env and agent consistency
        if env.env_name in config.GA_ENV_NAMES and self.agent_name not in config.GA_LEARN_AGENT_NAMES:
            raise ValueError

        if env.env_name in config.DE_ENV_NAMES and self.agent_name not in config.DE_LEARN_AGENT_NAMES:
            raise ValueError

        if env.env_name in config.BASE_ENV_NAME and self.agent_name not in (
                config.DE_LEARN_AGENT_NAMES + config.GA_LEARN_AGENT_NAMES):
            raise ValueError

        if env.base_solver_name == 'ga' and self.agent_name not in config.GA_LEARN_AGENT_NAMES:
            raise ValueError

        if env.base_solver_name == 'de' and self.agent_name not in config.DE_LEARN_AGENT_NAMES:
            raise ValueError

        if env.base_solver_name == 'jade' and self.agent_name not in config.DE_LEARN_AGENT_NAMES:
            raise ValueError

        if env.base_solver_name in ['ga', 'jade']:
            assert env.base_solver_name in self.alg_name

        if self.agent_name == 'mtde-l2t' or self.agent_name == 'mtga-l2t' or self.agent_name == 'mtjade-l2t':
            rl_algo_cls = get_agent_training_algo(self.alg_name)
            rl_algo_dir_name = self.alg_name.replace('_best', '')
            if self.retrained:
                # NOTE: should mind the change between using one dataset to transfer to another and the direct training
                # on the targeted problem set
                model = rl_algo_cls.load(
                    config.PROJECT_PATH + '/model/' + env.env_name + '/retrain/' + env.problem_name + '-train/'
                    + rl_algo_dir_name + f'/run{self.run_id}/' + self.alg_name, device='cpu')
            else:
                model = rl_algo_cls.load(config.PROJECT_PATH + '/model/' + env.env_name + '/pretrain/' +
                                         rl_algo_dir_name + '/' + self.alg_name, device='cpu')
            self.model = lambda x: model.predict(x, deterministic=True)[0]
        elif self.agent_name == 'mtde-r' or self.agent_name == 'mtga-r':
            self.model = lambda x: env.action_space.sample()
        elif self.agent_name == 'stde' or self.agent_name == 'stga' or self.agent_name == 'stjade':
            self.model = lambda x: np.zeros(env.action_space.shape)
        default_action = np.ones(env.action_space.shape)
        if self.agent_name == 'mtde-f(1,1,1)' or self.agent_name == 'mtga-f(1,1,1)':
            # transfer both base vectors and differential vectors with probability 0.5
            self.model = lambda x: default_action
        elif self.agent_name == 'mtde-f(1,0,1)' or self.agent_name == 'mtga-f(1,0,1)':
            # transfer differential vectors with probability 0.5
            default_action[:, 1] = 0
            self.model = lambda x: default_action
        elif self.agent_name == 'mtde-f(1,1,0)' or self.agent_name == 'mtga-f(1,1,0)':
            # transfer base vectors with probability 0.5
            default_action[:, 2] = 0
            self.model = lambda x: default_action
        elif self.agent_name == 'mtde-f(.5,1,1)' or self.agent_name == 'mtga-f(.5,1,1)':
            # transfer both base vectors and differential vectors with probability 0.25
            default_action[:, 0] = .5
            self.model = lambda x: default_action
        elif self.agent_name == 'mtde-f(.5,0,1)' or self.agent_name == 'mtga-f(.5,0,1)':
            # transfer differential vectors with probability 0.25
            default_action[:, 0] = .5
            default_action[:, 1] = 0
            self.model = lambda x: default_action
        elif self.agent_name == 'mtde-f(.5,1,0)' or self.agent_name == 'mtga-f(.5,1,0)':
            # transfer base vectors with probability 0.25
            default_action[:, 0] = .5
            default_action[:, 2] = 0
            self.model = lambda x: default_action


if __name__ == "__main__":
    import emto
    import gymnasium as gym

    for agt_name in config.GA_HUMAN_AGENT_NAMES:
        agt = HandcraftedAgent(agt_name)
        env = gym.make('l2t_emto-v1', env_mode='test',problem_name='cec17mtop6',base_solver_name='ga',max_gen=500)
        for env_seed in np.arange(1):
            for i in range(1):
                res = agt.rollout(env, env_seed)
                print(res[0])
                # print(res[1])

    # agt = LearnableAgent('mtde-l2t','ppo-v0')
    # env = gym.make('l2t_emto-v6', problem_name='bbob-v1')
    # for env_seed in range(3):
    #     print(agt.rollout(env, env_seed))