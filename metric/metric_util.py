import json
import os
import string
from collections import Counter
import re
import nltk
from argparse import ArgumentParser

def clean_sent(sent: list, lower=True):
    new_sent = []
    for word in sent:
        if lower:
            word = word.lower()
        new_word = clean_word(word)
        new_sent.append(new_word)
    return new_sent

def clean_word(word):
    chars = []
    for ch in word:
        ch_ord = ord(ch)
        if (ch_ord >= 65 and ch_ord<=90) or (ch_ord>=97 and ch_ord<=122) or (ch_ord>=48 and ch_ord<=57):
            chars.append(ch)
    if len(chars) > 0:
        return ''.join(chars)

    exclude = set(string.punctuation)
    if word in exclude:
        return word
    else:
        return ' '

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(nltk.word_tokenize(text))
        # return ' '.join(text.split())

    return white_space_fix(remove_punc(lower(s)))

def normlize2(s: list):
    new_list = []
    for word in s:
        n_word = remove_punc(lower(word))
        if len(n_word) > 0:
            new_list.append(n_word)
    return new_list


def f1_score(prediction, ground_truth):
    if not prediction or ground_truth == '':
        return 0, 0, 0
    prediction_tokens = normlize2(prediction)
    ground_truth_tokens = clean_sent(nltk.word_tokenize(ground_truth))

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def transform(origin_data: list):
    data = {}
    for item in origin_data:
        item_id = item['id']
        context = item['context']
        product_brand = item['product/brand']
        reason = item['reason']
        new_reason = {}
        for rea in reason:
            rea_text = rea['text']
            eve_type = rea['type']
            if eve_type not in new_reason:
                new_reason[eve_type] = rea_text
            else:
                new_reason[eve_type] = new_reason[eve_type] + ' ' + rea_text
        if item_id not in data:
            data[item_id] = {}
        data[item_id]['context'] = context
        data[item_id]['product/brand'] = product_brand
        data[item_id]['reason'] = new_reason
    return data

def calculate_metric(pred_data, gold_data):
    total_f1, total_num, total_precision, total_recall = 0., 0., 0., 0.
    for text_id in gold_data:
        # {"eve_type1": "xxx xxx xx", "eve_type2": "xx xx"}
        pred_reasons = pred_data[text_id]['reason']
        gold_reasons = gold_data[text_id]['reason']

        all_eve_types = pred_reasons.keys() | gold_reasons.keys()
        for k in all_eve_types:
            total_num += 1
            pred = pred_reasons.get(k, [])
            gold = gold_reasons.get(k, '')
            _f1, precision, recall = f1_score(prediction=pred, ground_truth=gold)
            total_f1 += _f1
            total_precision += precision
            total_recall += recall

    return total_f1 / total_num, total_precision / total_num, total_recall / total_num


def get_metric_f1(pred_data, gold_file_path):
    '''
    :param file_path: submit file path
    :return: pre, rec, f1
    '''
    gold_data = json.load(open(gold_file_path, 'r', encoding='utf-8'))
    gold_data = transform(gold_data)
    f1, precision, recall = calculate_metric(pred_data=pred_data, gold_data=gold_data)

    return f1, precision, recall
