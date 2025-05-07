import numpy as np
from emto.envs.task import TaskBBOB, TaskHPO, TaskCEC19, TaskCEC17, TaskCustom
import config
from problems.base_func import rotate_gen
import scipy


class CEC17MTOP:
    def __init__(self, benchmark_id):
        assert benchmark_id in list(np.arange(9) + 1)
        self.benchmark_id = benchmark_id

    def create_tasks(self):
        tasks = None
        if self.benchmark_id == 1:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/CI_H.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=2, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-100,
                                       ub=100),
                     TaskCEC17().build(f_id=3, i_id=2, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task2'][0], rotate=task_data['Rotation_Task2'], lb=-50,
                                       ub=50)]
        elif self.benchmark_id == 2:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/CI_M.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=1, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=3, i_id=2, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task2'][0], rotate=task_data['Rotation_Task2'], lb=-50,
                                       ub=50)]
        elif self.benchmark_id == 3:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/CI_L.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=1, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=5, i_id=2, benchmark_id=self.benchmark_id, dim=50, shift=np.zeros(50),
                                       rotate=np.eye(50), lb=-500, ub=500)]
        elif self.benchmark_id == 4:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/PI_H.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=3, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=6, i_id=2, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task2'][0], rotate=np.eye(50), lb=-100, ub=100)]
        elif self.benchmark_id == 5:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/PI_M.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=1, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=4, i_id=2, benchmark_id=self.benchmark_id, dim=50, shift=np.zeros(50),
                                       rotate=np.eye(50), lb=-50, ub=50)]
        elif self.benchmark_id == 6:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/PI_L.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=1, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=7, i_id=2, benchmark_id=self.benchmark_id, dim=25,
                                       shift=task_data['GO_Task2'][0], rotate=task_data['Rotation_Task2'], lb=-0.5,
                                       ub=0.5)]
        elif self.benchmark_id == 7:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/NI_H.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=4, i_id=1, benchmark_id=self.benchmark_id, dim=50, shift=np.zeros(50),
                                       rotate=np.eye(50), lb=-50, ub=50),
                     TaskCEC17().build(f_id=3, i_id=2, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task2'][0], rotate=task_data['Rotation_Task2'], lb=-50,
                                       ub=50)]
        elif self.benchmark_id == 8:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/NI_M.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=2, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-100,
                                       ub=100),
                     TaskCEC17().build(f_id=7, i_id=2, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task2'][0], rotate=task_data['Rotation_Task2'], lb=-0.5,
                                       ub=0.5)]
        elif self.benchmark_id == 9:
            task_data = np.load(config.PROJECT_PATH + '/problems/CEC17MultiTasks/NI_L.npy',
                                allow_pickle=True).tolist()
            tasks = [TaskCEC17().build(f_id=3, i_id=1, benchmark_id=self.benchmark_id, dim=50,
                                       shift=task_data['GO_Task1'][0], rotate=task_data['Rotation_Task1'], lb=-50,
                                       ub=50),
                     TaskCEC17().build(f_id=5, i_id=2, benchmark_id=self.benchmark_id, dim=50, shift=np.zeros(50),
                                       rotate=np.eye(50), lb=-500, ub=500)]
        return tasks


class CEC19MTOP:
    def __init__(self, benchmark_id):
        assert benchmark_id in list(np.arange(10) + 1)
        self.benchmark_id = benchmark_id
    
    def create_tasks(self):
        tasks = None
        if self.benchmark_id == 1:
            tasks = [TaskCEC19().build(f_id=6, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 2:
            tasks = [TaskCEC19().build(f_id=7, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 3:
            tasks = [TaskCEC19().build(f_id=17, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 4:
            tasks = [TaskCEC19().build(f_id=13, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 5:
            tasks = [TaskCEC19().build(f_id=15, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 6:
            tasks = [TaskCEC19().build(f_id=21, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 7:
            tasks = [TaskCEC19().build(f_id=22, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 8:
            tasks = [TaskCEC19().build(f_id=5, i_id=i_id, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)
                     for i_id in [1,2]]
        elif self.benchmark_id == 9:
            tasks = [TaskCEC19().build(f_id=11, i_id=1, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100),
                     TaskCEC19().build(f_id=16, i_id=2, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)]
        elif self.benchmark_id == 10:
            tasks = [TaskCEC19().build(f_id=20, i_id=1, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100),
                     TaskCEC19().build(f_id=21, i_id=2, benchmark_id=self.benchmark_id, dim=50, lb=-100, ub=100)]
        return tasks


class DiverseMTOP:
    def __init__(self, benchmark_id):
        assert benchmark_id in list(np.arange(10) + 1)
        self.benchmark_id = benchmark_id
        self.dim = 10
        self.ctr = np.ones(self.dim) * 0.5
        self.lh = scipy.stats.qmc.LatinHypercube(self.dim, seed=1997)  # random seed for latin hypercube is 1997
        self.n_tasks = 2
        self.rads = [0.025, 0.05, 0.1, 0.2, 0.4]  # define the radius of each sub-distribution
        if benchmark_id in [1, 2, 3, 4, 5]:
            self.sub_ctrs = self.lh.random(1) # generate the center for each sub-distribution
        elif benchmark_id in [6, 7]:
            self.sub_ctrs = self.lh.random(2)
        elif benchmark_id in [8]:
            self.sub_ctrs = self.lh.random(3)
        elif benchmark_id in [9]:
            self.sub_ctrs = self.lh.random(4)
        elif benchmark_id in [10]:
            self.sub_ctrs = self.lh.random(5)
        self.f_ids = [1,2,3,6,7] # define the function ID for constructing task, see 'task' file for detailed definition of each function
        self.i_ids = [1,2]

    def sample_subdist_id(self, np_gen: np.random.RandomState):
        return np_gen.choice(np.arange(self.sub_ctrs.shape[0]),p=np.ones(self.sub_ctrs.shape[0])/self.sub_ctrs.shape[0])

    def create_tasks(self, np_gen: np.random.RandomState):
        tasks = None
        n_tasks = self.n_tasks
        if self.benchmark_id in [1,2,3,4,5]:
            task_sfts = np.array([np_gen.uniform(self.ctr - self.rads[int(self.benchmark_id - 1)],
                                                 self.ctr + self.rads[int(self.benchmark_id - 1)]) for i in range(n_tasks)])
            task_rots = [rotate_gen(self.dim, np_gen) for _ in range(n_tasks)]
            task_fids = [np_gen.choice(self.f_ids) for _ in range(n_tasks)]
            tasks = [TaskCustom().build(f_id=task_fids[i], i_id=self.i_ids[i], dim=self.dim, shift=task_sfts[i],
                                        rotate=task_rots[i]) for i in range(n_tasks)]
        elif self.benchmark_id in [6]:
            # interpolation
            task_sfts = np.array([np_gen.rand() * (self.sub_ctrs[1] - self.sub_ctrs[0]) + self.sub_ctrs[0] for i in range(n_tasks)])
            task_sfts = np.clip(task_sfts, 0.0, 1.0)
            task_rots = [rotate_gen(self.dim, np_gen) for _ in range(n_tasks)]
            task_fids = [np_gen.choice(self.f_ids) for _ in range(n_tasks)]
            tasks = [TaskCustom().build(f_id=task_fids[i], i_id=self.i_ids[i], dim=self.dim, shift=task_sfts[i],
                                        rotate=task_rots[i]) for i in range(n_tasks)]
        elif self.benchmark_id in [7,8,9,10]:
            task_subdist_ids = [self.sample_subdist_id(np_gen) for _ in range(n_tasks)]
            task_sfts = np.array([np_gen.uniform(self.sub_ctrs[task_subdist_ids[i]] - 0.1,
                                                 self.sub_ctrs[task_subdist_ids[i]] + 0.1) for i in range(n_tasks)])
            task_sfts = np.clip(task_sfts, 0.0, 1.0)
            task_rots = [rotate_gen(self.dim, np_gen) for _ in range(n_tasks)]
            task_fids = [np_gen.choice(self.f_ids) for _ in range(n_tasks)]
            tasks = [TaskCustom().build(f_id=task_fids[i], i_id=self.i_ids[i], dim=self.dim, shift=task_sfts[i],
                                        rotate=task_rots[i]) for i in range(n_tasks)]
        return tasks


def get_multiple_tasks(problem_name, np_gen: np.random.RandomState, n_tasks, dim):
    # for training or fine-tuning
    if problem_name == 'bbob-train':
        f_fids = [1, 3, 8, 10, 16, 20]  # value should lie in [1,24]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v3-train':
        f_fids = [1, ]  # value should lie in [1,24]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v4-train':
        f_fids = [3, ]  # value should lie in [1,24]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v9-train':
        f_fids = list(set(np.arange(1, 25)) - {1, 3, 8, 10, 16, 20})  # value should lie in [1,24]
        i_ids = list(np.arange(500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v10-train':
        f_fids = [2,6,12,15,21]  # value should lie in [1,24]
        i_ids = list(np.arange(500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'hpo-svm-train':
        permute = np.random.RandomState(1)
        f_fids = [1, ]
        i_ids = list(np.arange(100)[permute.permutation(100)][:50])
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-2) for task_id in
                 range(n_tasks)]
    elif problem_name == 'hpo-xgboost-train':
        permute = np.random.RandomState(2)
        f_fids = [2, ]
        i_ids = list(np.arange(100)[permute.permutation(100)][:50])
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-1) for task_id in
                 range(n_tasks)]
    elif problem_name == 'hpo-fcnet-train':
        permute = np.random.RandomState(3)
        f_fids = [3, ]
        i_ids = list(np.arange(100)[permute.permutation(100)][:50])
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-1) for task_id in
                 range(n_tasks)]
    elif problem_name in ['cust-v1-train','cust-v2-train', 'cust-v3-train', 'cust-v4-train','cust-v5-train',
                          'cust-v6-train','cust-v7-train', 'cust-v8-train', 'cust-v9-train','cust-v10-train']:
        mtop_gen = DiverseMTOP(int(problem_name.replace('cust-v','').replace('-train','')))
        f_fids = mtop_gen.f_ids
        i_ids = np.arange(mtop_gen.n_tasks)
        tasks = mtop_gen.create_tasks(np_gen)
        cur_fids = [task.f_id for task in tasks]
        cur_iids = [task.i_id for task in tasks]

    # testing the i.i.d. (with a moderate difference) generalization
    elif problem_name == 'bbob-v1':
        f_fids = [1, 3, 8, 10, 16, 20]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v2':
        f_fids = [1, 3, 8, 10, 16, 20]  # value should lie in [1,24]
        i_ids = list(np.arange(1000, 1005))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v3':
        f_fids = [1, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v4':
        f_fids = [3, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v5':
        f_fids = [8, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v6':
        f_fids = [10, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v7':
        f_fids = [16, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v8':
        f_fids = [20, ]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]

    # testing the o.o.d. (possibly) generalization
    elif problem_name == 'bbob-v9':
        f_fids = list(set(np.arange(1,25)) - {1, 3, 8, 10, 16, 20})  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v10':
        f_fids = [2,6,12,15,21]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v11':
        f_fids = [2]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v12':
        f_fids = [6]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v13':
        f_fids = [12]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v14':
        f_fids = [15]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'bbob-v15':
        f_fids = [21]  # value should lie in [1,24]
        i_ids = list(np.arange(500, 1500))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskBBOB().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], dim=dim,
                                  targets_amount=10, targets_precision=-8) for task_id in range(n_tasks)]
    elif problem_name == 'cec17mtop':
        f_fids = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # value should lie in [1,9]
        i_ids = [1, 2]
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        while (cur_fids[0] == cur_fids[1]) and (cur_iids[0] == cur_iids[1]):
            cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
            cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = []
        for cur_fid, cur_iid in zip(cur_fids, cur_iids):
            benchmark_id = int(cur_fid)
            benchmark = CEC17MTOP(benchmark_id)
            tasks.append(benchmark.create_tasks()[cur_iid - 1])
    elif problem_name == 'cec19mtop':
        f_fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # value should lie in [1,10]
        i_ids = [1, 2]
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        while (cur_fids[0] == cur_fids[1]) and (cur_iids[0] == cur_iids[1]):
            cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
            cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = []
        for cur_fid, cur_iid in zip(cur_fids, cur_iids):
            benchmark_id = cur_fid
            benchmark = CEC19MTOP(int(benchmark_id))
            tasks.append(benchmark.create_tasks()[cur_iid - 1])
    elif problem_name == 'hpo-svm':
        f_fids = [1, ]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-3) for task_id in
                 range(n_tasks)]
    elif problem_name == 'hpo-xgboost':
        f_fids = [2, ]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-3) for task_id in
                 range(n_tasks)]
    elif problem_name == 'hpo-fcnet':
        f_fids = [3, ]
        i_ids = list(np.arange(100))
        cur_fids = [np_gen.choice(f_fids) for _ in range(n_tasks)]
        cur_iids = [np_gen.choice(i_ids) for _ in range(n_tasks)]
        tasks = [TaskHPO().build(f_id=cur_fids[task_id], i_id=cur_iids[task_id], targets_precision=-3) for task_id in
                 range(n_tasks)]
    elif problem_name in ['cust-v1','cust-v2','cust-v3','cust-v4','cust-v5','cust-v6','cust-v7','cust-v8','cust-v9','cust-v10']:
        mtop_gen = DiverseMTOP(int(problem_name.replace('cust-v','')))
        f_fids = mtop_gen.f_ids
        i_ids = np.arange(mtop_gen.n_tasks)
        tasks = mtop_gen.create_tasks(np_gen)
        cur_fids = [task.f_id for task in tasks]
        cur_iids = [task.i_id for task in tasks]
    else:
        raise Exception('Invalid problem name')
    info = {'f_fids'  : f_fids,
            'i_ids'   : i_ids,
            'cur_fids': cur_fids,
            'cur_iids': cur_iids}
    return tasks, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # tasks, info = get_multiple_tasks('cec19mtop', np.random.RandomState(1), 2, None)
    # print(tasks,info)
    # print()
    # a=np.random.rand(100,50)
    # print([task(a) for task in tasks])
    # a=TaskCustom()

    # np_gen = np.random.RandomState(8)
    # mtop_gen = DiverseMTOP(10)
    # x = []
    # for i in range(100):
    #     tasks = mtop_gen.create_tasks(np_gen)
    #     x += [task.f.xopt for task in tasks]
    #     print([task(task.f.xopt.reshape(1,-1))[0] for task in tasks])
    # x = np.array(x)
    # plt.scatter(x[:,0], x[:,1])
    # plt.axis([0,1,0,1])
    # plt.show()

