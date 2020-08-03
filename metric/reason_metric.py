import torch
from metric.metric_util import get_metric_f1
import json

class ReasonMetrics(object):
    def __init__(self):
        self.reset()

    def __call__(self, predictions, dataPath):
        f1, precision, recall = get_metric_f1(predictions, dataPath)
        self.result['precision'] = precision
        self.result['recall'] = recall
        self.result['f1'] = f1

    def get_metric(self, reset=False):
        result = self.result
        if reset:
            self.reset()
        return result

    def reset(self):
        self.result = {}
