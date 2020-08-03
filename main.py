import os
import json
import time
import torch
import argparse
import torch.nn as nn
from dataloaders import group, GroupBatchRandomSampler, get_batch
from model.model import EventModel
from utils.util import time_display, print_detail_info, get_triple
from pytorch_transformers import *
import numpy as np
from metric.reason_metric import ReasonMetrics
from metric.cls_metric import CLSMetrics
from tqdm import tqdm
from preprocessings.process import Process
import pdb

class Runner(object):
    def __init__(self, config_path):
        self.hyper = json.load(open(config_path, 'r'))
        self.model_dir = self.hyper['saved_model_dir']
        self.device = torch.device('cuda:{}'.format(self.hyper['gpu_id']) if torch.cuda.is_available() else 'cpu')
        self.preprocess = Process(self.hyper)
        self.event_vocab = json.load(open(self.hyper['event_vocab'], 'r'))
        self.id2event = {id: event for event, id in self.event_vocab.items()}
        self._reason_metrics = ReasonMetrics()
        self._cls_metrics = CLSMetrics()
        self.read_data()

    def read_data(self):
        self.train_data = self.preprocess.process(filePath=self.hyper['train_data_path'], mode='train')
        self.dev_data = self.preprocess.process(filePath=self.hyper['dev_data_path'], mode='dev')

    def _optimizer(self):
        bert_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if "bert" in name:
                bert_params += [param]
            else:
                other_params += [param]
        optimizer_grouped_parameters = [
            {'params': other_params, "lr": self.hyper['lr'], 'weight_decay': 0.0},
            {'params': bert_params, "lr": self.hyper['bert_lr']}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyper['lr'],eps=1e-8)
        sampler = GroupBatchRandomSampler(self.train_data, self.hyper['train_batch_size'],
                                          data_group=self.hyper['train_data_group'],
                                          breakpoints=self.hyper['groups'])
        t_total = len(sampler) * self.hyper['epoch_num']
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=self.hyper['warmup_ratio']*t_total, t_total=t_total)
        del sampler

    def _init_model(self):
        self.model = EventModel(self.hyper).to(self.device)

    def run(self, mode: str):
        if mode == 'train':
            self._init_model()
            self._optimizer()
            self.train()
        if mode == 'eval_dev':
            self._init_model()
            self.load_model()
            metric_reason, metric_cls = self.evaluation(self.dev_data)
            print_detail_info(metric_reason, metric_cls)
        if mode not in ['train', 'eval_test', 'eval_dev']:
            raise ValueError('invalid mode')

    def load_model(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'model.pt'), map_location=self.device))
        
    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'model.pt'))

    def evaluation(self, data):
        self.model.eval()
        result = {}
        sampler = GroupBatchRandomSampler(data, self.hyper['eval_batch_size'],
                                          data_group=self.hyper['dev_data_group'],
                                          breakpoints=self.hyper['groups'])
        with torch.no_grad():
            for batch_idx, batch_indices in tqdm(enumerate(sampler)):
                padded_sentence_ids, \
                batch_text_id, \
                all_text, \
                query, \
                context, \
                event_type, \
                start_pos_ids, \
                end_pos_ids, \
                bert_att_mask, \
                segment_mask, \
                note_flags = get_batch(batch_indices, data, self.device)

                output = self.model(input_ids=padded_sentence_ids,
                                    attention_mask=bert_att_mask,
                                    token_type_ids=segment_mask,
                                    start_pos=start_pos_ids,
                                    end_pos=end_pos_ids,
                                    note_flag=note_flags)
                loss, start_logits, end_logits, event_logits = output['total_loss'], output['start_logits'], output[
                    'end_logits'], output['event_logits']
                event_note = torch.argmax(event_logits, dim=-1)

                for text_id, start, end, eve_label, mask, text, s_query, _all_text, flag in zip(batch_text_id,
                                                                                        start_logits,
                                                                                        end_logits,
                                                                                        event_type,
                                                                                        segment_mask.eq(1),
                                                                                        context,
                                                                                        query,
                                                                                        all_text,
                                                                                        event_note):
                    if text_id not in result:
                        result[text_id] = {}
                        result[text_id]["context"] = text
                        result[text_id]['product/brand'] = ""
                        result[text_id]['reason'] = {}
                    query_length = len(s_query)+2
                    cur_length = query_length + len(text)
                    single_result = get_triple(start, end, eve_label, mask, query_length, cur_length)
                    
                    if flag.item() == 0:
                        continue
                    result[text_id]['reason'][eve_label] = []
                    for event_label, reason_start, reason_end in single_result:
                        reason_text = _all_text[reason_start:reason_end+1]
                        result[text_id]['reason'][eve_label].extend(reason_text)
                        
            self._reason_metrics(result, self.hyper['dev_data_path'])
            self._cls_metrics(result, self.hyper['dev_data_path'])
            
            metric_reason = self._reason_metrics.get_metric(reset=True)
            metric_cls = self._cls_metrics.get_metric(reset=True)
            return metric_reason, metric_cls


    def train(self):
        best_fscore = None
        start_time = time.time()
        for epoch in range(self.hyper['epoch_num']):
            self.model.train()
            total_loss = 0.
            count = 0
            sampler = GroupBatchRandomSampler(self.train_data, self.hyper['train_batch_size'],
                                              data_group=self.hyper['train_data_group'],
                                              breakpoints=self.hyper['groups'])
            for batch_idx, batch_indices in enumerate(sampler):
                padded_sentence_ids, \
                text_id, \
                all_text, \
                query, \
                context, \
                event_type, \
                start_pos_ids, \
                end_pos_ids, \
                bert_att_mask, \
                segment_mask, note_flag = get_batch(batch_indices, self.train_data, self.device)

                self.optimizer.zero_grad()

                output = self.model(input_ids=padded_sentence_ids,
                                    attention_mask=bert_att_mask,
                                    token_type_ids=segment_mask,
                                    start_pos=start_pos_ids,
                                    end_pos = end_pos_ids,
                                    note_flag=note_flag)

                loss = output['total_loss']
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper['grad_clip_norm'])
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                count += 1

                if (batch_idx + 1) % self.hyper['print_per_batch'] == 0:
                    elapsed = time.time() - start_time
                    print("| Epoch {:2d}/{:2d}| Batch {:3d}/{:3d}| Time {:s}| "
                          "Loss {:3.3f} | lr {:2.7f}|".format(epoch, self.hyper['epoch_num'], batch_idx + 1, len(sampler),
                                                                                        time_display(elapsed),
                                                                                        total_loss / count,self.scheduler.get_lr()[0]
                                                                                        ))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("| End of Epoch {:2d} : ".format(epoch))

            metric_reason, metric_cls = self.evaluation(self.dev_data)
            reason_f1 = metric_reason['f1']
            print_detail_info(metric_reason, metric_cls)
            
            if not best_fscore or reason_f1 > best_fscore:
                self.save_model()
                best_fscore = reason_f1
            else:
                pass
            
        print("------------------------- training end -------------------------")
        print('| -----------------  BEST  Result ------------------|')
        print('DevData Result: ')
        self._init_model()
        self.load_model()
        metric_reason, metric_cls = self.evaluation(self.dev_data)
        print_detail_info(metric_reason, metric_cls)   
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process mode of run.')
    parser.add_argument('mode',metavar='MODE',type=str,choices=['train','eval_test','eval_dev'], help='train for train and produce a model ,eval for evaluate trained model for prediciton')
    args = parser.parse_args()
    main = Runner('./conf/config.json')
    main.run(mode=args.mode)
