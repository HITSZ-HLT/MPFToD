import copy
import importlib
import os
import pickle
import random

from torch.utils.data import DataLoader

from utils.multi_task_dataset import QiDataset, HiDataset, KbiDataset, MultiTaskDataset, MultiTaskBatchSampler, Collater


class Args(object):
    def __init__(self, contain=None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name[-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if not (value is None) or (name[:2] == "__" and name[-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        return list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)


class Vocab(object):
    def __init__(self, words, add_pad=False):
        if add_pad:
            self.wordlist = ["<unk>"] + ["<sos>"] + ["<eos>"] + words
        else:
            self.wordlist = words
        self.worddict = {}
        for idx, word in enumerate(self.wordlist):
            self.worddict[word] = idx

    def __len__(self):
        return len(self.wordlist)

    def __iter__(self):
        return self.wordlist.__iter__()

    def word2idx(self, word):
        if word not in self.wordlist:
            return self.worddict["<unk>"]
        return self.worddict[word]

    def idx2word(self, idx):
        return self.wordlist[idx]


class Batch(object):
    @staticmethod
    def to_list(source, batch_size):
        """
        Change the list to list of lists, which each list contains a batch size number of items.
        :param source: list
        :param batch_size: batch size
        :return: list of lists
        """
        batch_list = []
        idx = 0
        while idx < len(source):
            next_idx = idx + batch_size
            if next_idx > len(source):
                next_idx = len(source)
            batch_list.append(source[idx: next_idx])
            idx = next_idx
        return batch_list

    @staticmethod
    def get_batch(source, batch_size, idx):
        """
        get the idx-th batch
        :param source:
        :param batch_size:
        :param idx:
        :return:
        """
        bgn = min(idx * batch_size, len(source))
        end = min((idx + 1) * batch_size, len(source))
        return source[bgn: end]


def idx_extender(source, max_len=None, pad=None, bias=0):
    """
    [(1,3),(2,2)] ---> [1,1,1,2,2]
    if bias==1, then ---> [2,2,2,3,3]
    useful for the type token ids
    :param source: list of tuples
    :param max_len: max length we want to pad to
    :param pad: pad token, "<pad>" e.g.
    :param bias: add bias to all idx
    :return: the extended idx
    """
    out = []
    for idx, num in source:
        for _ in range(num):
            out.append(idx + bias)
    cur_len = len(out)
    while cur_len < max_len:
        out.append(pad)
        cur_len += 1
    return out


def in_each(source, method):
    """
    In each is a iterator function which you can employ the method
    in every item in source.
    :param source: a list of items
    :param method: the method you want to employ to the items
    :return: the new items
    """
    return [method(x) for x in source]


def pad(inputs, pad):
    """
    Pad function for a list of lists(each list is a sequence of word.)
    :param inputs: list of lists
    :param pad: pad symbol
    :return: all_padded(padded list od lists), all_idx(type idx)
    """
    max_len = 0
    for input in zip(*inputs):
        cur_len = 0
        for x in input:
            cur_len += len(x)
        if cur_len > max_len:
            max_len = cur_len
    all_padded = []
    all_idx = []
    for input in zip(*inputs):
        line_padded = []
        line_idx = []
        for idx, x in enumerate(input):
            line_idx.append((idx, len(x)))
            line_padded += x
        cur_len = len(line_padded)
        while cur_len < max_len:
            line_padded.append(pad)
            cur_len += 1
        all_padded.append(line_padded)
        all_idx.append(line_idx)
    return all_padded, all_idx


def get_model(model):
    Model = importlib.import_module('models.{}'.format(model)).Model
    return Model


def get_loader(dataset_tool):
    DatasetTool = importlib.import_module('utils.process.{}'.format(dataset_tool)).DatasetTool
    return DatasetTool


def get_multi_task_loader(args, tokenizer):
    qi_dataset_name = args.dataset.qi_dataset_names.split(',')
    hi_dataset_name = args.dataset.hi_dataset_names.split(',')
    kbi_dataset_name = args.dataset.kbi_dataset_names.split(',')
    qi_dataset = QiDataset(qi_dataset_name, tokenizer, args.dataset.cached, args.dataset.cache_dir, args.train.toy)
    hi_dataset = HiDataset(hi_dataset_name, tokenizer, args.dataset.cached, args.dataset.cache_dir, args.train.toy)
    kbi_dataset = KbiDataset(kbi_dataset_name, tokenizer, args.dataset.cached, args.dataset.cache_dir, args.train.toy)

    if args.train.task_id == 0:
        mt_dataset = MultiTaskDataset([qi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([qi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)
    elif args.train.task_id == 1:
        mt_dataset = MultiTaskDataset([hi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([hi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)
    elif args.train.task_id == 2:
        mt_dataset = MultiTaskDataset([kbi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([kbi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)
    elif args.train.task_id == 3:  # remove KBI
        mt_dataset = MultiTaskDataset([qi_dataset, hi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([qi_dataset, hi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)
    elif args.train.task_id == 4:  # remove HI
        mt_dataset = MultiTaskDataset([qi_dataset, kbi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([qi_dataset, kbi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)
    elif args.train.task_id == 5:  # remove QI
        mt_dataset = MultiTaskDataset([hi_dataset, kbi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([hi_dataset, kbi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)

    else:
        mt_dataset = MultiTaskDataset([qi_dataset, hi_dataset, kbi_dataset])
        multi_task_batch_sampler = MultiTaskBatchSampler([qi_dataset, hi_dataset, kbi_dataset], args.train.batch,
                                                         args.dataset.mix_opt,
                                                         args.dataset.ratio)

    collector = Collater(tokenizer, args.dataset.mask_rate, args.dataset.qi_rep_rate, args.dataset.hi_rep_rate)
    train_data = DataLoader(mt_dataset, batch_sampler=multi_task_batch_sampler, collate_fn=collector.collate_fn)
    return train_data


def resample(dataset, length):
    data = copy.deepcopy(dataset.data)
    while len(dataset.data) < length:
        rand_idx = random.randint(0, len(data) - 1)
        dataset.data.append(data[rand_idx])
    return dataset


def get_evaluator():
    EvaluateTool = importlib.import_module('utils.evaluate').EvaluateTool
    return EvaluateTool


def get_entities_set(args):
    entities = []
    if os.path.exists(os.path.join(args.dataset.cache_dir, 'kbi_dataset')):
        fr = open(os.path.join(args.dataset.cache_dir, 'kbi_dataset'), 'rb')
        data = pickle.load(fr)
        for dic in data:
            entities = ['entities']
            entities.extend(dic)
        fr.close()
    return list(set(entities))


def str_filter(inp):
    return inp.startswith('goodbye') or inp.startswith('thank') or inp.startswith('enjoy') or inp.startswith(
        'have a nice day') or inp.startswith('glad') or inp.startswith('you\'re') or inp.startswith(
        'no problem') or inp.startswith('good bye') or inp.startswith('you are') or inp.startswith(
        'is that all') or inp.startswith('anytime') or inp.startswith('have a good day') or inp.startswith(
        'is there anything else') or inp.startswith('of course.') or inp.startswith('i\'m') or inp.startswith(
        'i am') or inp.startswith('ok thank') or inp.startswith('you too')


def qi_filter(inp):
    return inp.startswith('goodbye') or inp.startswith('thank') or inp.startswith('enjoy') or inp.startswith(
        'have a nice day') or inp.startswith('glad') or inp.startswith('you\'re') or inp.startswith(
        'no problem') or inp.startswith('good bye') or inp.startswith('you are') or inp.startswith(
        'is that all') or inp.startswith('anytime') or inp.startswith('have a good day') or inp.startswith(
        'is there anything else') or inp.startswith('of course.') or inp.startswith('i\'m') or inp.startswith(
        'i am') or inp.startswith('ok thank') or inp.startswith('you too') or inp.startswith('have a great day')


def hi_filter(inp):
    return inp == 'done' or inp == 'sure' or inp == 'get it' or inp.startswith(
        'have a') or inp == 'ok all set' or inp == 'anything else' or inp == 'great, all set'
