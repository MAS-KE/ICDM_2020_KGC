import os
import json
import pickle
from tqdm import tqdm
import nltk
from pytorch_transformers import *
import string

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

class Process():
    def __init__(self, config):
        self.config = config
        self.data_root = 'data/'
        self.event_vocab = {'O': 0}
        self.event_query_map = json.load(open(config['event_query_path'], 'r', encoding='utf-8'))
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.config['bert_model_path'])

    def locate_sequence(self, reason_text, context):
        def match_score(reason_list, text_list):
            match_num = 0
            for i in range(len(text_list)):
                if text_list[i] == reason_list[i]:
                    match_num += 1
            return match_num / len(text_list)

        pos = -1
        reason_len, context_len = len(reason_text), len(context)
        for i in range(0, context_len-reason_len+1):
            if reason_text == context[i:i+reason_len]:
                pos = i
                break

        if pos == -1:
            max_score, idx = 0, -1
            for i in range(0, context_len - reason_len + 1):
                m_score = match_score(reason_text, context[i:i + reason_len])
                if m_score > max_score:
                    idx = i
                    max_score = m_score

            return idx, max_score
        return pos, None

    def data_check(self, reason_list, context):
        new_reason_info = []
        for reason in reason_list:
            start = reason['start']
            reason_type = reason['type']
            reason_text = reason['text']
            reason_text = nltk.word_tokenize(reason_text)
            reason_text = clean_sent(reason_text, lower=True)
            seq_pos, score = self.locate_sequence(reason_text, context)
            reason_update_text = None
            if score:
                reason_update_text = context[seq_pos:seq_pos + len(reason_text)]

            new_reason_info.append({"reason_start": seq_pos,
                                    "reason_end": seq_pos + len(reason_text),
                                    "reason_text": reason_text,
                                    "reason_update_text": reason_update_text,
                                    "event_type": reason_type})

        return new_reason_info

    def process(self, mode='train', filePath='train_data.json'):
        all_data = []
        for instance in tqdm(json.load(open(os.path.join(filePath), 'r', encoding='utf-8'))):
            context = instance['context']
            _, context = context.split(',', maxsplit=1)
            text_id = instance["id"]
            context = nltk.word_tokenize(context)
            context = clean_sent(context, lower=True)
            product_brand = instance['product/brand']

            for eve_type in self.event_query_map:
                query = self.event_query_map[eve_type]
                query_context = query.format(product_brand).lower()
                query_context = query_context.split()
                query_len = len(query_context)

                total_text = ['[CLS]'] + query_context + ['[SEP]'] + context
                # bert: word->id
                token_ids = self.text2ids(total_text)

                assert len(total_text) == len(context) + len(query_context) + 2

                all_data.append([token_ids, # bert: word->id
                                 text_id,   # context_id
                                 total_text, # query + context
                                 context,
                                 query_context,
                                 eve_type])

        print('loading finished !, total: ', len(all_data))
        return all_data


    def text2ids(self, text):
        # TODO: tokenizer
        padded_list = list(map(lambda x: self.bert_tokenizer.convert_tokens_to_ids(x), list(text)))
        assert len(padded_list) == len(text),'not equal !'
        return padded_list
