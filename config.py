import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_MODES = ['train',
             'test']
BASE_SOLVER_NAMES = ['de',
                     'jade',
                     'ga']
NUM_PRE_INIT_POPS = 10
EVAL_PROBLEM_NAMES = [
                      'bbob-v1',
                      'bbob-v2',
                      'bbob-v3',
                      'bbob-v4',
                      'bbob-v5',
                      'bbob-v6',
                      'bbob-v7',
                      'bbob-v8',
                      'bbob-v9',
                      'bbob-v10',
                      'bbob-v11',
                      'bbob-v12',
                      'bbob-v13',
                      'bbob-v14',
                      'bbob-v15',
                      # 'cec17mtop',
                      # 'cec19mtop',
                      # 'hpo-svm',
                      # 'hpo-xgboost',
                      # 'hpo-fcnet',
                      # 'cust-v1',
                      # 'cust-v2',
                      # 'cust-v3',
                      # 'cust-v4',
                      # 'cust-v5',
                      # 'cust-v6',
                      # 'cust-v7',
                      # 'cust-v8',
                      # 'cust-v9',
                      # 'cust-v10',
                      ]
BASE_ENV_NAME = ['l2t_emto-v1']
PROBLEM_NAMES = ['bbob-train',
                 'bbob-v1',
                 'bbob-v2',
                 'bbob-v3',
                 'bbob-v4',
                 'bbob-v5',
                 'bbob-v6',
                 'bbob-v7',
                 'bbob-v8',
                 'bbob-v9',
                 'hpo-svm',
                 'hpo-xgboost',
                 'hpo-fcnet']
ALG_NAMES = ['ppo-v0',
             'ppo-v0_best',
             'ppo-v1',
             'ppo-v1_best',
             'ppo-v0-a1',  # remove action 1
             'ppo-v0-a2',  # remove action 2
             'ppo-v0-a3',  # remove action 3
             'ppo-v0-s1',  # remove common feature
             'ppo-v0-s2',  # remove task feature
             'ppo-v0-spop',  # replace pop as input state
             'ppo-v0-b(1,0)',
             'ppo-v0-b(1,.01)',
             'ppo-v0-b(1,.05)',
             'ppo-v0-b(1,.1)',
             'ppo-v0-b(1,.5)',
             'ppo-v0-b(1,1)',
             'ppo-v0-b(1,5)',
             'ppo-v0-b(1,10)',
             'ppo-v0-b(1,50)',
             'ppo-v0-b(1,100)',
             'ppo-v0-b(0,1)',
             'ppo-v0-ga',
             'ppo-v0-ft',  # finetune
             'ppo-v0-wotr', # train from scratch
             'ppo-v0-ga-ft',
             'ppo-v0-ga-wotr',
             'sac',
             'td3',
             'ppo-v0-jade',
             ]
DE_ENV_NAMES = ['l2t_emto-v6']
DE_BASE_AGENT_NAME = ['mtde-l2t']
DE_LEARN_AGENT_NAMES = ['mtde-l2t',
                        'mtde-r',
                        'stde',
                        'mtde-f(1,1,1)',
                        'mtde-f(1,0,1)',
                        'mtde-f(1,1,0)',
                        'mtde-f(.5,1,1)',
                        'mtde-f(.5,0,1)',
                        'mtde-f(.5,1,0)',
                        'stjade',
                        'mtjade-l2t',
                        ]
DE_HUMAN_AGENT_NAMES = ['aemto',
                        'mktde',
                        'mfde',
                        'mtde-b',
                        'mtde-ea',
                        'mtde-ad']
GA_ENV_NAMES = ['l2t_emto-v5']
GA_BASE_AGENT_NAME = ['mtga-l2t']
GA_LEARN_AGENT_NAMES = ['mtga-l2t',
                        'mtga-r',
                        'stga',
                        'mtga-f(1,1,1)',
                        'mtga-f(1,0,1)',
                        'mtga-f(1,1,0)',
                        'mtga-f(.5,1,1)',
                        'mtga-f(.5,0,1)',
                        'mtga-f(.5,1,0)',
                        ]
GA_HUMAN_AGENT_NAMES = ['atmfea',
                        'gmfea',
                        'mfea',
                        'mfea2',
                        'mfea-akt',
                        'mtea-ad',
                        'mfea-rl']
