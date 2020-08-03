import os
import pickle
import torch
import numpy as np
import re
import string

def time_display(s):
    d = s // (3600*24)
    s -= d * (3600*24)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    str_time = "{:1d}d ".format(int(d)) if d else "   "
    return str_time + "{:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s))

def print_detail_info(metric_reason, metric_cls):
    reason_precision = metric_reason['precision']
    reason_recall = metric_reason['recall']
    reason_f1 = metric_reason['f1']
    cls_precision = metric_cls['precision']
    cls_recall = metric_cls['recall']
    cls_f1 = metric_cls['f1']
    print('Eval Result: ')
    print('| reason_precision: {} | reason_recall: {} | reason_f1: {} |'.format(reason_precision, reason_recall, reason_f1))
    print('| cls_precision: {} | cls_recall: {} | cls_f1: {} |'.format(cls_precision, cls_recall, cls_f1))


def decode_reason(start_seq, end_seq):
    result = []
    for s_id, st in enumerate(start_seq):
        if st > 0:
            for e_id in range(s_id, len(end_seq)):
                if end_seq[e_id] > 0:
                    result.append([s_id, e_id + 1])
                    break
    return result

def get_triple(start_labels, end_labels, event_label, mask, query_length, cur_length):
    span_triple_lst = set()
    start_labels = torch.nonzero(start_labels)
    end_labels = torch.nonzero(end_labels)
    if len(start_labels) == 0 or len(end_labels) == 0:
        return span_triple_lst
    start_labels = torch.clamp(start_labels, min=query_length, max=cur_length)

    for tmp_start in start_labels:
        if mask[tmp_start]:
            tmp_end = end_labels[end_labels > tmp_start.item()]
            tmp_end = torch.clamp(tmp_end, min=query_length, max=cur_length)

            if len(tmp_end) == 0:
                continue
            for candidate_end in tmp_end:
                if mask[candidate_end]:
                        span_triple_lst.add(
                            (event_label, tmp_start.item(), candidate_end.item()))
                        break
    return span_triple_lst
