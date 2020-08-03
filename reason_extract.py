import torch
import os
import json
from ReaExtract.data_loader import GroupBatchRandomSampler, get_batch
from ReaExtract.process import Process
from model.model import EventModel
from utils.util import get_triple

class Reason_Extract(object):
    def __init__(self, config_path, dataPath):
        self.hyper = json.load(open(config_path, 'r', encoding='utf-8'))
        self.model_dir = self.hyper['saved_model_dir']
        self.device = torch.device('cuda:{}'.format(self.hyper['gpu_id']) if torch.cuda.is_available() else 'cpu')
        self.event_vocab = json.load(open(self.hyper['event_vocab'], 'r', encoding='utf-8'))
        self.id2event = {id: event for event, id in self.event_vocab.items()}
        self.process = Process(self.hyper)

        self._init_model()
        self.load_model()
        self.model.eval()

    def _init_model(self):
        self.model = EventModel(self.hyper).to(self.device)

    def load_model(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'model.pt'), map_location=self.device))

    def predict(self):
        '''
        :param
        :return:
        '''
        data = self.process.process(filePath=self.hyper['valid_data_path'], mode='dev')
        sampler = GroupBatchRandomSampler(data, self.hyper['eval_batch_size'], data_group=False, shuffle=False)
        result = {}
        with torch.no_grad():
            for batch_idx, batch_indices in enumerate(sampler):
                padded_sentence_ids, \
                batch_text_id, \
                all_text, \
                query, \
                context, \
                event_type, \
                bert_att_mask, \
                segment_mask = get_batch(batch_indices, data, self.device)

                output = self.model(input_ids=padded_sentence_ids,
                                    attention_mask=bert_att_mask,
                                    token_type_ids=segment_mask)

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

                    query_length = len(s_query) + 2
                    cur_length = query_length + len(text)
                    single_result = get_triple(start, end, eve_label, mask, query_length, cur_length)
                    if not flag:
                        continue
                    result[text_id]['reason'][eve_label] = []
                    for event_label, reason_start, reason_end in single_result:
                        reason_text = _all_text[reason_start:reason_end + 1]
                        reason_text = ' '.join(reason_text)
                        result[text_id]['reason'][eve_label].append(reason_text)
        return result

    def transform_format(self, pred_result):
        new_result = []
        for text_id, item in pred_result.items():
            context = item['context']
            reasons = item['reason']
            product = item['product/brand']
            new_reasons = []
            for eve_type, rea_list in reasons.items():
                for rea in rea_list:
                    new_reasons.append({"text": rea, "type": eve_type})
            new_result.append({"context": ' '.join(context),
                                "reason": new_reasons,
                                "product/brand": product,
                                "id": text_id})

        return new_result

    def submit(self):
        pred_result = self.predict()
        new_result = self.transform_format(pred_result)

        with open('pred_result.json', 'w', encoding='utf-8') as f:
            json.dump(new_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pred = Reason_Extract('conf/config.json', 'data/')
    pred.submit()

