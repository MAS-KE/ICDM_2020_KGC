import bisect
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *
from torch.nn.utils.rnn import *
import os
import torch
import json
import numpy as np

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class GroupBatchRandomSampler(object):
    def __init__(self, data, batch_size, drop_last=False, shuffle=True, data_group=True, breakpoints=None):
        self.batch_indices = []
        if data_group:
            groups = group(data, breakpoints=breakpoints)
            total =0
            for data_group in groups:
                total += len(data_group.indices)
            for data_group in groups:
                self.batch_indices.extend(list(BatchSampler(SubsetRandomSampler(data_group.indices),
                                                            batch_size, drop_last=drop_last)))
        else:
            total_num = len(data)
            data_idx = list(range(total_num))
            if shuffle:
                np.random.shuffle(data_idx)
            dataGroup = Subset(data, data_idx)
            self.batch_indices = []
            self.batch_indices.extend(list(BatchSampler(SubsetSampler(dataGroup.indices),
                                                        batch_size, drop_last=drop_last)))

    def __iter__(self):
        return (self.batch_indices[i] for i in range(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)

def get_batch(batch_indices, data, device):
    batch = [data[idx] for idx in batch_indices]
    token_ids, \
    text_id, \
    total_text, \
    context, \
    query_context, \
    eve_type = zip(*batch)

    lengths = [len(s) for s in token_ids]
    max_length = max(lengths)
    query_lengths = [len(q) + 2 for q in query_context]
    context_lengths = [len(t) for t in context]
    padded_sentence_ids = torch.tensor(pad_sequence(token_ids, max_length, pad_value=0))
    bert_att_mask = getAttentionMask(lengths, max_length)
    segment_mask = getSegmentMask(query_lengths, context_lengths, max_length)

    return padded_sentence_ids.to(device), \
           text_id, \
           total_text, \
           query_context, \
           context, \
           eve_type, \
           bert_att_mask.to(device), \
           segment_mask.to(device)

def pad_sequence(token_ids, max_length, pad_value=0):
    padded_res = [seq+(max_length-len(seq))*[pad_value] for seq in token_ids]
    return padded_res

def get_position_ids(batch_start_pos, batch_end_pos, max_length):
    batch_size = len(batch_start_pos)
    start_pos_ids = torch.zeros(batch_size, max_length)
    end_pos_ids = torch.zeros(batch_size, max_length)
    for batch_id, (sent_ss, sent_ee) in enumerate(zip(batch_start_pos, batch_end_pos)):
        for ss, ee in zip(sent_ss, sent_ee):
            start_pos_ids[batch_id][ss] = 1.
            end_pos_ids[batch_id][ee] = 1.
    return start_pos_ids, end_pos_ids

def getAttentionMask(lengths, max_length):
    mask = [[1]*s_len + [0]*(max_length-s_len) for s_len in lengths]
    return torch.tensor(mask)

def getSegmentMask(query_lengths, text_lengths, max_length):
    mask = [[0]*query_len + [1]*text_len + [0]*(max_length-query_len-text_len) for query_len, text_len in zip(query_lengths, text_lengths)]
    mask = torch.tensor(mask)
    return mask

def group(data, breakpoints=None):
    if not breakpoints:
        breakpoints = [10, 20, 30, 40, 50, 60, 70]
    groups = [[] for _ in range(len(breakpoints)+1)]
    for idx, item in enumerate(data):
        i = bisect.bisect_left(breakpoints, len(item[0]))
        groups[i].append(idx)
    data_groups = [Subset(data, g) for g in groups]
    return data_groups

