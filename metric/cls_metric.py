import json
import os

class CLSMetrics(object):
    def __init__(self):
        self.reset()

    def __call__(self, pred_result, dataPath):
        gold_labels, predictions = self.getResult(pred_result, dataPath)
        for text_id, gold in gold_labels.items():
            pred = predictions[text_id]
            self.tp_fp += len(pred)
            self.tp_fn += len(gold)
            self.tp += len(pred & gold)

    def get_metric(self, reset=False):
        print('tp_fp: ', self.tp_fp, 'tp_fn: ',self.tp_fn, self.tp)
        precision = 0. if self.tp_fp == 0 else (self.tp / self.tp_fp)
        recall = 0. if self.tp_fn == 0 else (self.tp / self.tp_fn)
        f1 = 0. if self.tp_fp + self.tp_fn == 0 else 2 * self.tp / (self.tp_fp + self.tp_fn)

        if reset:
            self.reset()
        return {"precision": precision,
                "recall": recall,
                "f1": f1}

    def reset(self):
        self.tp_fp = 0
        self.tp_fn = 0
        self.tp = 0

    def getResult(self, pred_result, dataPath):
        gold_data = json.load(open(dataPath, 'r', encoding='utf-8'))
        gold_data, pred_data = self.data_transform(gold_data, pred_result)
        return gold_data, pred_data

    def data_transform(self, gold_data, pred_result):
        ret_data_pred, ret_data_gold = {}, {}
        for item in gold_data:
            id = item['id']
            reason = item['reason']
            if id not in ret_data_gold:
                ret_data_gold[id] = []
            if id not in ret_data_pred:
                ret_data_pred[id] = set(pred_result[id]['reason'].keys())

            for rea in reason:
                ret_data_gold[id].append(rea['type'])
            ret_data_gold[id] = set(ret_data_gold[id])

        return ret_data_gold, ret_data_pred
