import torch
import torch.nn as nn
from pytorch_transformers import BertModel

class EventModel(nn.Module):
    def __init__(self, config):
        super(EventModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model_path'])
        self.dropout = nn.Dropout(config['bert_dropout'])

        self.span_cls = nn.Linear(config['bert_embedding_dim'], 2)
        self.has_ans_cls = nn.Linear(config['bert_embedding_dim'], 2)
        self.weight_start = config['weight']['start']
        self.weight_end = config['weight']['end']
        self.weight_event = config['weight']['event_note']

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='sum')
        self.has_ans_loss = nn.CrossEntropyLoss(reduction='sum', weight=torch.tensor([1, 10]).float())


    def forward(self, input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_pos=None,
                end_pos=None,
                note_flag=None):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        batch_size, seq_len, hid_size = sequence_output.size()

        span_logits = self.span_cls(sequence_output)
        start_logits, end_logits = torch.split(span_logits, split_size_or_sections=1, dim=-1)
        event_logits = self.has_ans_cls(pooled_output)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        output = {"span_loss": 0.,
                  "note_loss": 0.,
                  "total_loss": 0.,
                  "start_logits": [],
                  "end_logits": [],
                  "event_logits": []}

        mask = token_type_ids.eq(1)
        if start_pos is not None and note_flag is not None:
            start_loss = self.loss_fct(start_logits[mask], start_pos[mask])
            end_loss = self.loss_fct(end_logits[mask], end_pos[mask])
            event_note_loss = self.has_ans_loss(event_logits, note_flag)

            total_loss = self.weight_start * start_loss + self.weight_end * end_loss + \
                self.weight_event * event_note_loss

            output['start_loss'] = start_loss
            output['end_loss'] = end_loss
            output['event_note_loss'] = event_note_loss
            output['total_loss'] = total_loss / batch_size

        '''
        start_logits = (torch.sigmoid(start_logits) > self.config['start_threshold']).long()
        end_logits = (torch.sigmoid())
        
        '''
        start_logits = (torch.sigmoid(start_logits) > self.config['start_threshold']).long()
        end_logits = (torch.sigmoid(end_logits) > self.config['end_threshold']).long()
        # event_logits = torch.sigmoid(event_logits)

        output['start_logits'] = start_logits
        output['end_logits'] = end_logits
        output['event_logits'] = event_logits

        return output
