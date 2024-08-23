import os
import torch
import importlib
import random
import numpy as np
import copy
from torch.utils.data import Dataset, BatchSampler

import pickle
import utils.tool


def search_bin(bins, size):
    idx = len(bins) - 1
    for i, bin in enumerate(bins):
        if size <= bin:
            idx = i
            break
    return idx


def create_bins(bin_size, maxlen):
    return [min(i + bin_size, maxlen) for i in range(0, maxlen, bin_size)]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, mix_opt, extra_task_ratio):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio

        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i + batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio)
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class QiDataset(Dataset):
    def __init__(self, dataset_names, tokenizer, cached=False, cache_dir=None, toy=False):
        self.task_id = 0
        self.cached = cached
        self.cache_dir = cache_dir
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        if cached and os.path.exists(os.path.join(cache_dir, 'qi_dataset')):
            fr = open(os.path.join(cache_dir, 'qi_dataset'), 'rb')
            if not toy:
                self.data = pickle.load(fr)

            else:
                data = pickle.load(fr)
                toy_len = len(data) // 30
                self.data = data[:toy_len]
            fr.close()
        else:
            self.data = self.read_data()

    def create_qi_sample(self, dialogs):
        samples = []
        for dialog in dialogs:
            for idx, sentence in enumerate(dialog):
                if idx % 2 == 1 and not utils.tool.qi_filter(dialog[idx]) and not utils.tool.qi_filter(dialog[idx - 1]) and len(
                        dialog[idx - 1]) > 10:
                    samples.append({"task_id": 0, "query": dialog[idx - 1], 'response': dialog[idx]})
        return samples

    def read_data(self):
        data = []
        for dataset_name in self.dataset_names:
            parser = importlib.import_module('utils.process.{}.parser'.format(dataset_name)).Parser
            data_list = parser.load()
            data.extend(data_list)
        results = self.create_qi_sample(data)
        if self.cached:
            fw = open(os.path.join(self.cache_dir, 'qi_dataset'), 'wb')
            pickle.dump(results, fw)
            fw.close()
        return results

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_task_id(self):
        return self.task_id


class HiDataset(Dataset):
    def __init__(self, dataset_names, tokenizer, cached=False, cache_dir=None, toy=False):
        self.task_id = 1
        self.cached = cached
        self.cache_dir = cache_dir
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        if cached and os.path.exists(os.path.join(cache_dir, 'hi_dataset')):
            fr = open(os.path.join(cache_dir, 'hi_dataset'), 'rb')
            if not toy:
                self.data = pickle.load(fr)
            else:
                data = pickle.load(fr)
                toy_len = len(data) // 30
                self.data = data[:toy_len]
            fr.close()
        else:
            self.data = self.read_data()

    def create_hi_sample(self, dialogs):
        samples = []
        for dialog in dialogs:
            if len(dialog) >= 4 and len(dialog) % 2 == 0 and not utils.tool.hi_filter(dialog[-1]):
                samples.append({"task_id": 1, "history": dialog[:-2], 'response': dialog[-1]})
            elif len(dialog) > 3 and not utils.tool.hi_filter(dialog[-1]):
                samples.append({"task_id": 1, "history": dialog[:-3], 'response': dialog[-2]})
            else:
                continue
        return samples

    def read_data(self):
        data = []
        for dataset_name in self.dataset_names:
            parser = importlib.import_module('utils.process.{}.parser'.format(dataset_name)).Parser
            data_list = parser.load()
            data.extend(data_list)
        results = self.create_hi_sample(data)
        if self.cached:
            fw = open(os.path.join(self.cache_dir, 'hi_dataset'), 'wb')
            pickle.dump(results, fw)
            fw.close()
        return results

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_task_id(self):
        return self.task_id


class KbiDataset(Dataset):
    def __init__(self, dataset_names, tokenizer, cached=False, cache_dir=None, toy=False):
        self.task_id = 2
        self.cached = cached
        self.cache_dir = cache_dir
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        if cached and os.path.exists(os.path.join(cache_dir, 'kbi_dataset')):
            fr = open(os.path.join(cache_dir, 'kbi_dataset'), 'rb')
            if not toy:
                self.data = pickle.load(fr)
            else:
                data = pickle.load(fr)
                toy_len = len(data) // 30
                self.data = data[:toy_len]
            fr.close()
        else:
            self.data = self.read_data()

    def create_kbi_sample(self, datas):
        pass

    def read_data(self):
        data = []
        for dataset_name in self.dataset_names:
            parser = importlib.import_module('utils.process.{}.parser'.format(dataset_name)).Parser
            data_list = parser.load()
            data.extend(data_list)
        if self.cached:
            fw = open(os.path.join(self.cache_dir, 'kbi_dataset'), 'wb')
            pickle.dump(data, fw)
            fw.close()
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_task_id(self):
        return self.task_id


class Collater(object):
    def __init__(self, tokenizer, mask_rate=0.15, qi_rep_rate=0.5, hi_rep_rate=0.5):
        self.tokenizer = tokenizer
        self.mask_rate = mask_rate
        self.qi_rep_rate = qi_rep_rate
        self.hi_rep_rate = hi_rep_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = self.tokenizer.max_model_input_sizes['bert-base-cased']
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.sot_token = "[SOT]"
        self.sot_token_id = self.tokenizer(self.sot_token)
        self.special_token_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens)

    def pad_sequence(self, datas, id=None):
        pad_id = self.pad_id if id is None else id
        max_len = min(max(len(response) for response in datas), self.max_len)
        attention_mask = []
        for idx, data in enumerate(datas):
            data_len = len(data)
            if data_len > self.max_len:
                datas[idx] = data[:self.max_len - 1] + [self.sep_id]
                data_len = len(datas[idx])
                assert data_len == self.max_len
            pad_len = max_len - data_len
            attention_mask.append([1] * data_len + [0] * pad_len)
            datas[idx] = datas[idx] + [pad_id] * pad_len
        return datas, attention_mask

    def replace_op_for_hi(self, histories, labels):
        if len(histories) > 1:
            for idx, history in enumerate(histories):
                if random.random() < self.hi_rep_rate:
                    rand_history_idx = random.randint(0, len(histories) - 1)
                    # Two different histories of guaranteed exchange
                    while rand_history_idx == idx:
                        rand_history_idx = random.randint(0, len(histories) - 1)
                    new_history = histories[rand_history_idx]
                    cnt = 0
                    rep_num = min(len(history), len(new_history)) / 2
                    while cnt < rep_num:
                        rand_utt_idx = random.randint(0, len(history) - 1)
                        new_rand_utt_idx = random.randint(0, len(new_history) - 1)
                        # Ensure that the two sentences exchanged are SYS/USR
                        if len(new_history) >= 2:
                            while rand_utt_idx % 2 != new_rand_utt_idx % 2:
                                new_rand_utt_idx = random.randint(0, len(new_history) - 1)
                        # Replace words and labels
                        histories[idx][rand_utt_idx] = new_history[new_rand_utt_idx]
                        cnt += 1

                    labels[idx] = 1

        return histories, labels

    def replace_op_for_qi(self, histories):
        labels = []
        for idx, history in enumerate(histories):
            length = int((len(history) - 2) / 2)
            label = [0] * length
            rand_idx = random.randint(0, length)
            histories[idx].insert(rand_idx, history[-1])
            histories[idx] = histories[idx][:-1]
            label.insert(rand_idx, 1)
            pad_len = len(histories[idx]) - len(label)
            label += [-100] * pad_len
            labels.append(label)
        return histories, labels

    def pad_histories(self, histories, labels, task_id):
        if task_id == 1:
            new_histories, labels = self.replace_op_for_hi(copy.deepcopy(histories), labels)
        else:
            new_histories, labels = self.replace_op_for_qi(copy.deepcopy(histories))
        return new_histories, labels

    def mask_one_part(self, _text, _table, entity, part='text'):
        # 0:mask， 1:not mask
        text = self.tokenizer.tokenize(_text)
        text_label = copy.deepcopy(text)
        table = self.tokenizer.tokenize(_table)
        table_label = copy.deepcopy(table)
        text_infos = [0] * len(text)
        table_infos = [0] * len(table)
        mask_num = self.mask_rate * (len(text))
        cnt = 0
        if part == 'text':
            for idx, token in enumerate(text_label):
                for ent in entity:
                    ent_list = self.tokenizer.tokenize(ent)
                    if token in ent_list:
                        if idx == 0:
                            text[idx] = self.tokenizer.mask_token
                            text_infos[idx] = 1
                            cnt += 1
            while cnt < mask_num:
                rand_idx = random.randint(0, len(text) - 1)
                while text_infos[rand_idx] == 1 or text[rand_idx] in self.special_token_list:
                    rand_idx = random.randint(0, len(text) - 1)
                if text_infos[rand_idx] == 0:
                    text[rand_idx] = self.tokenizer.mask_token
                    text_infos[rand_idx] = 1
                    cnt += 1

        elif part == 'table':
            for idx, token in enumerate(table_label):
                for ent in entity:
                    ent_list = self.tokenizer.tokenize(ent)
                    if token in ent_list:
                        if idx == 0:
                            table[idx] = self.tokenizer.mask_token
                            table_infos[idx] = 1
                            cnt += 1
            while cnt < mask_num:
                rand_idx = random.randint(0, len(table) - 1)
                while table_infos[rand_idx] == 1 or table[rand_idx] in self.special_token_list:
                    rand_idx = random.randint(0, len(table) - 1)
                if table_infos[rand_idx] == 0:
                    table[rand_idx] = self.tokenizer.mask_token
                    table_infos[rand_idx] = 1
                    cnt += 1

        text_label_ids = self.tokenizer.convert_tokens_to_ids(text_label)
        table_label_ids = self.tokenizer.convert_tokens_to_ids(table_label)

        for idx, (_text, _text_info) in enumerate(zip(text, text_infos)):
            if _text_info == 0:
                text_label_ids[idx] = -100

        for idx, (_table, _table_info) in enumerate(zip(table, table_infos)):
            if _table_info == 0:
                table_label_ids[idx] = -100

        label = [-100] + text_label_ids + [-100] + table_label_ids + [-100]

        return text, table, label

    def mask_tokens(self, texts, tables, entities):
        _texts = []
        _tables = []
        labels = []
        for text, table, entity in zip(texts, tables, entities):
            part = 'text' if random.random() < 0.5 else 'table'
            text, table, label = self.mask_one_part(text, table, entity, part=part)
            _texts.append(text)
            _tables.append(table)
            labels.append(label)

        return _texts, _tables, labels

    def replace_response(self, responses, labels):
        # 0:repalcement， 1: not replacement
        tmp_responses = copy.deepcopy(responses)
        if len(responses) > 1:
            for idx, res in enumerate(responses):
                if random.random() < self.qi_rep_rate:
                    rand_idx = random.randint(0, len(responses) - 1)
                    while rand_idx == idx:
                        rand_idx = random.randint(0, len(responses) - 1)
                    responses[idx] = tmp_responses[rand_idx]
                    labels[idx] = 1
        return responses, labels

    def replace_query(self, queries, labels):
        # 0:repalcement， 1: not replacement
        tmp_queries = copy.deepcopy(queries)
        if len(queries) > 1:
            for idx, res in enumerate(queries):
                if random.random() < self.qi_rep_rate:
                    rand_idx = random.randint(0, len(queries) - 1)
                    while rand_idx == idx:
                        rand_idx = random.randint(0, len(queries) - 1)
                    queries[idx] = tmp_queries[rand_idx]
                    labels[idx] = 1
        return queries, labels

    def collate_fn(self, batch):
        task_id = batch[0]['task_id']
        # QI
        if task_id == 0:
            queries = [item['query'] for item in batch]
            responses = [item['response'] for item in batch]
            labels = [0] * len(queries)
            if random.random() < 0.5:
                queries, labels = self.replace_query(queries, labels)
            else:
                responses, labels = self.replace_response(responses, labels)
            query_infos = []
            response_infos = []
            for idx, (query, response) in enumerate(zip(queries, responses)):
                query_info = " [USR] " + query
                response_info = " [SYS] " + response
                query_infos.append(query_info)
                response_infos.append(response_info)
            tokenized = self.tokenizer(query_infos, response_infos, truncation='only_first', padding=True,
                                       return_tensors='pt',
                                       max_length=self.tokenizer.max_model_input_sizes['bert-base-uncased'],
                                       return_token_type_ids=True)
            label_ids = torch.tensor(labels)
            return task_id, {'input_ids': tokenized.data['input_ids'].to(self.device),
                             'token_type_ids': tokenized.data['token_type_ids'].to(self.device),
                             'attention_mask': tokenized.data['attention_mask'].to(self.device)}, label_ids.to(
                self.device)

        # HI
        elif task_id == 1:
            histories = [item['history'] for item in batch]
            responses = [item['response'] for item in batch]
            labels = [0] * len(responses)
            new_histories, labels = self.pad_histories(histories, labels, task_id)
            history_infos = []
            response_infos = []
            for idx, (history, response) in enumerate(zip(new_histories, responses)):
                history_info = ''
                for i, sentence in enumerate(history):
                    if i % 2 == 0:
                        history_info += " [USR] " + sentence
                    else:
                        history_info += " [SYS] " + sentence
                response_info = " [SYS] " + response
                response_infos.append(response_info)
                history_infos.append(history_info)

            tokenized = self.tokenizer(history_infos, response_infos, truncation='only_first', padding=True,
                                       return_tensors='pt',
                                       max_length=self.tokenizer.max_model_input_sizes['bert-base-uncased'],
                                       return_token_type_ids=True)

            label_ids = torch.tensor(labels)

            return task_id, {'input_ids': tokenized.data['input_ids'].to(self.device),
                             'token_type_ids': tokenized.data['token_type_ids'].to(self.device),
                             'attention_mask': tokenized.data['attention_mask'].to(self.device)}, label_ids.to(
                self.device)

        # KBI
        elif task_id == 2:
            texts = [item['text'] for item in batch]
            table_strs = [item['linear_table'] for item in batch]
            entities = [item['entities'] for item in batch]
            text, table, labels = self.mask_tokens(texts, table_strs, entities)

            tokenized = self.tokenizer(text, table, truncation='only_second', padding=True,
                                       return_tensors='pt',
                                       max_length=self.tokenizer.max_model_input_sizes['bert-base-uncased'],
                                       return_token_type_ids=True)

            label_ids, _ = self.pad_sequence(labels, id=-100)
            label_ids = torch.tensor(label_ids)

            return task_id, {'input_ids': tokenized.data['input_ids'].to(self.device),
                             'token_type_ids': tokenized.data['token_type_ids'].to(self.device),
                             'attention_mask': tokenized.data['attention_mask'].to(self.device)}, label_ids.to(
                self.device)
        else:
            raise Exception()
