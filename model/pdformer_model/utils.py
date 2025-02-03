import importlib
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy

from traffic_model.pdformer_model.list_dataset import ListDataset
from traffic_model.pdformer_model.batch import Batch


def get_dataset(config):
    try:
        return getattr(importlib.import_module('federatedscope.contrib.model.pdformer_model.dataset'),
                       config['dataset_class'])(config)
    except AttributeError:
        raise AttributeError('dataset_class is not found')


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, pad_item=None,
                        pad_max_len=None, shuffle=True,
                        pad_with_last_sample=False, distributed=False):
    # print("train_data.shape", len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][0][0][0]))
    if pad_with_last_sample:
        # num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        # data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        # train_data = np.concatenate([train_data, data_padding], axis=0)
        # num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        # data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        # eval_data = np.concatenate([eval_data, data_padding], axis=0)
        # num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        # data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        # test_data = np.concatenate([test_data, data_padding], axis=0)
        train_data = train_data[:-(len(train_data) % batch_size)]
        eval_data = eval_data[:-(len(eval_data) % batch_size)]
        test_data = test_data[:-(len(test_data) % batch_size)]
        
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)
    train_sampler = None
    eval_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    def collator(indices):
        # print("indices.shape: ", len(indices), len(indices[0]), len(indices[0][0]), len(indices[0][0][0]), len(indices[0][0][0][0]))
        batch = Batch(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    # print("train_dataset.shape", len(train_dataset), len(train_dataset[0]), len(train_dataset[0][0]), len(train_dataset[0][0][0]), len(train_dataset[0][0][0][0]))
    # print("num_workers", num_workers)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle and train_sampler is None, sampler=train_sampler)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle and eval_sampler is None, sampler=eval_sampler)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader
