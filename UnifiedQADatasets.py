import torch
import json
import re
import os
import random
from torch.utils.data import Dataset
from tokenization_t5 import EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0
from CPM2Datasets import CPM2Dataset


class UnifiedQADataset(CPM2Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(UnifiedQADataset, self).__init__(args, tokenizer, path, split, ratio, num, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []
        full_enc_sizes = []

        with open(self.path, "r") as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                d = json.loads(line)
                passage = d['passage']
                question = d['question']
                answer_set = set(d['answers'])
                if '?' not in question:
                    question += '?'
                idx = d['idx']
                sentence = ' '.join(['question:', question, 'context:', passage])
                sid = self.tokenizer.encode(sentence)
                aid = self.tokenizer.encode('answer:') + [self.tokenizer.get_sentinel_id(0)] + [self.tokenizer.eod_id]
                if self.prompt_config:
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context = [-(i + 1) for i in range(prompt_len)] + sid
                else:
                    context = sid
                full_enc_sizes.append(len(context) + len(aid))
                if len(context) > self.args.enc_seq_length - len(aid):
                    context = context[: self.args.enc_seq_length - len(aid)]
                context += aid
                for answer in answer_set:
                    target = [0, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(answer) + [self.tokenizer.eod_id]
                    current_data = {
                        "idx": self.idx,  
                        "enc_input_ids": context,
                        "dec_input_ids": target[:-1],
                        "label_ids": target[1:],
                        "origin_idx": d['idx'],
                        "answers": answer_set,
                        "origin_dataset": d['idx'].split('.')[0]
                    }
                    if 'question_idx' in d:
                        current_data['question_idx'] = d['question_idx']
                    data.append(current_data)

                    enc_sizes.append(len(context))
                    dec_sizes.append(len(target) - 1)
                    self.idx += 1
                    break

        max_full_enc_len = max(full_enc_sizes)
        if max_full_enc_len > self.args.enc_seq_length:
            print_rank_0("Full encoder input lenth: {:d}".format(max_full_enc_len))
            save_rank_0(self.args, "Full encoder input lenth: {:d}".format(max_full_enc_len))
        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)
        if max_dec_len < self.args.dec_seq_length:
            max_dec_len = self.args.dec_seq_length  # set max decoder length

        return data, max_enc_len, max_dec_len
