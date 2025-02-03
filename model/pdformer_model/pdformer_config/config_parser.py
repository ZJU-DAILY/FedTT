import os
import json
import torch


class ConfigParser(object):
    def __init__(self, task, model, dataset, data_config, config_file=None,
                 saved_model=True, train=True, other_args=None, hyper_config_dict=None, initial_ckpt=None):
        self.config = {}
        self.config['file_path'] = data_config["file_path"]
        self.config['data_file'] = data_config["data_file"]
        self.config['geo_file'] = data_config["geo_file"]
        self.config['rel_file'] = data_config["rel_file"]
        self.config['dyna_file'] = data_config["dyna_file"]
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict,
                                    initial_ckpt)
        self._parse_config_file(config_file)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None, hyper_config_dict=None,
                               initial_ckpt=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = False if task == 'map_matching' else train
        if other_args is not None:
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]
        self.config['initial_ckpt'] = initial_ckpt

    def _parse_config_file(self, config_file):
        print("此时的config_file为：", config_file)
        print("os.getcwd：", os.getcwd())
        if config_file is not None:
            if os.path.exists('traffic_model/pdformer_model/pdformer_config/{}.json'.format(config_file)):
                print('{}.json'.format(config_file),
                      "！！！！！！config_file文件存在")
                with open('traffic_model/pdformer_model/pdformer_config/{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                print('{}.json'.format(config_file), "不存在！！！")
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        with open('traffic_model/pdformer_model/pdformer_config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = task_config[model]['dataset_class']
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                self.config['traj_encoder'] = task_config[model]['traj_encoder']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
        default_file_list = []
        default_file_list.append('traffic_model/pdformer_model/pdformer_config/model/{}/{}.json'.format(self.config['task'], self.config['model']))
        default_file_list.append('traffic_model/pdformer_model/pdformer_config/data/{}.json'.format(self.config['dataset_class']))
        default_file_list.append('traffic_model/pdformer_model/pdformer_config/executor/{}.json'.format(self.config['executor']))
        default_file_list.append('traffic_model/pdformer_model/pdformer_config/evaluator/{}.json'.format(self.config['evaluator']))
        print("正在读取model/PDFormer配置文件")
        print("正在读取data/PDFormerDataset配置文件")
        print("正在读取executor/PDFormerExecutor配置文件")
        print("正在读取evaluator/TrafficStateEvaluator配置文件")
        for file_name in default_file_list:
            with open('{}'.format(file_name), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        with open('Dataset/new_dataset/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        distributed = False
        if 'WORLD_SIZE' in os.environ:
            distributed = int(os.environ['WORLD_SIZE']) > 1
        self.config['distributed'] = distributed
        if use_gpu and distributed:
            local_rank = self.config['local_rank']
            assert local_rank >= 0
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            rank = torch.distributed.get_rank()
            self.config["rank"] = rank
            assert rank >= 0
            self.config['world_size'] = torch.distributed.get_world_size()
            # wql add: modify the device
            # self.config['device'] = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
            self.config['device'] = "cuda:%d" % local_rank if torch.cuda.is_available() else "cpu"
        else:
            if use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                torch.cuda.set_device(0)
            # wql add: modify the device
            # self.config['device'] = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
            self.config['device'] = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return self.config.__iter__()
