USE_TORCH_DDP = False
import os
import re
import random
import torch
import torch.nn.functional as F
import json
import shutil
from sklearn.metrics import f1_score
from collections import OrderedDict, Counter
import string
import requests
import time

from arguments import get_args
from tokenization_t5 import EncDecTokenizer

import mpu
from utils import save_checkpoint
from utils import print_args
from utils import print_rank_0, save_rank_0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed

from samplers import DistributedBatchSampler, RandomSampler


tokenizer = None
enc_seq_length = 512
dec_length = 256
question = "What are US and Iran relations tensioned about?"
docs = [{
    'title': 'test',
    'text': "TEHRAN, Iran (CNN) -- Iran's parliament speaker has criticized U.S. President-elect Barack Obama for saying that Iran's development of a nuclear weapon is unacceptable. Iranian President Mahmoud Ahmadinejad has outlined where he thinks U.S. policy needs to change. Ali Larijani said Saturday that Obama should apply his campaign message of change to U.S. dealings with Iran. \"Obama must know that the change that he talks about is not simply a superficial changing of colors or tactics,\" Larijani said in comments carried by the semi-official Mehr News Agency. \"What is expected is a change in strategy, not the repetition of objections to Iran's nuclear program, which will be taking a step in the wrong direction.\" In his first post-election news conference Friday afternoon, Obama reiterated that he believes a nuclear-armed Iran would be \"unacceptable.\" He also said he would help mount an international effort to prevent it from happening. Larijani said that U.S. behavior toward Iran \"will not change so simply\" but that Obama's election showed internal conditions in the United States have shifted. He added that Iran does not mind if the United States provides other Persian Gulf countries with nuclear technology, but \"you should know that you cannot prevent the Islamic Republic [from reaching its goals in the nuclear field],\" according to the news agency. Obama cautioned Friday that it had only been a few days since the election and that he was not in office. \"Obviously, how we approach and deal with a country like Iran is not something that we should simply do in a knee-jerk fashion. I think we've got to think it through,\" Obama said. \"But I have to reiterate once again that we only have one president at a time. And I want to be very careful that we are sending the right signals to the world as a whole that I am not the president, and I won't be until January 20th.\" Larijani was speaking two days after Iranian President Mahmoud Ahmadinejad congratulated Obama, the first time an Iranian leader has offered such wishes to a U.S. president-elect since the 1979 Islamic Revolution. One analyst said the welcome was a gesture from the hard-line president that he is open to a more conciliatory relationship with the United States. Ahmadinejad said Tehran \"welcomes basic and fair changes in U.S. policies and conducts,\" according to the state-run Islamic Republic News Agency on Thursday. Relations between the United States and Iran have historically been chilly and have been further strained in recent years over Iran's nuclear program. Tehran insists that the program exists for peaceful purposes, but the United States and other Western nations are concerned by Iran's refusal to halt uranium enrichment activities. CNN's Shirzad Bozorgmehr contributed to this report."
}]
doc_scores = [1.0]
# question = "highest mountain in Shandong Province"
host="http://166.111.5.239:21212/search"

data_name_info = {
    "enc_input_ids": (2, torch.long),
    "enc_attention_mask": (4, torch.float),
    "dec_attention_mask": (4, torch.float),
    "cross_attention_mask": (4, torch.float),
    "dec_input_ids": (2, torch.long),
}

def construct_input(question, passages, tokenizer, args):
    if '?' not in question:
        question += '?'
    bs = len(passages)
    model_data = {
        "enc_input_ids": torch.ones(bs, enc_seq_length, dtype=torch.long) * tokenizer.pad_id,
        "enc_attention_mask": torch.zeros(bs, 1, enc_seq_length, enc_seq_length),
        "dec_attention_mask": torch.zeros(bs, 1, dec_length, dec_length),
        "cross_attention_mask": torch.zeros(bs, 1, dec_length, enc_seq_length),
        "dec_input_ids": torch.ones(bs, dec_length, dtype=torch.long) * tokenizer.pad_id
    }
    for i, passage in enumerate(passages):
        sentence = ' '.join(['question:', question, 'context:', passage])
        sid = tokenizer.encode(sentence)
        aid = tokenizer.encode('answer:') + [tokenizer.get_sentinel_id(0)] + [tokenizer.eod_id]
        context = sid
        if len(context) > enc_seq_length - len(aid):
            context = context[: enc_seq_length - len(aid)]
        context += aid
        enc_len, dec_len = len(context), dec_length
        model_data["enc_input_ids"][i][:enc_len] = torch.tensor(context, dtype=torch.long)
        model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
        model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
        model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
    return model_data

def generate(args, model, model_batch, tokenizer, device):
    all_preds = []
    all_logits = []
    with torch.no_grad():
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        bs = enc_hidden_states.size(0)
        unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
        output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
        # for generating responses
        # we only use the <go> token, so truncate other tokens
        dec_prompt_len = 0
        dec_input_ids = model_batch['dec_input_ids'][..., :1 + dec_prompt_len]
        dec_attention_mask = model_batch['dec_attention_mask'][..., :1 + dec_prompt_len, :1 + dec_prompt_len]
        # # we use past_key_values, so only the current token mask is needed
        cross_attention_mask = model_batch['cross_attention_mask'][..., :1 + dec_prompt_len, :]
        unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
        output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
        past_key_values = None
        scores = []
        gen_len = 0
        while gen_len < dec_length:
            if unfinished_sents.max() == 0:
                break
            else:
                dec_outputs = model(
                    dec_input_ids=dec_input_ids,
                    dec_attention_mask=dec_attention_mask,
                    cross_attention_mask=cross_attention_mask,
                    enc_hidden_states=enc_hidden_states,
                    past_key_values=past_key_values,
                )
                lm_logits = dec_outputs["lm_logits"]
                past_key_values = dec_outputs['past_key_values']

                gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())

                all_lm_logits = torch.cat(gathered_lm_logits, dim=-1)
                next_token_logits = all_lm_logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                loss = mpu.vocab_parallel_cross_entropy(lm_logits.contiguous().float().squeeze(), next_token)
                # current_logits, _ = torch.max(next_token_logits, dim=-1)  # get logits score for sort on OpenQA
                tokens_to_add = next_token * unfinished_sents + tokenizer.pad_id * (1 - unfinished_sents)
                dec_input_ids = tokens_to_add.unsqueeze(-1)
                output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                scores.append(loss * unfinished_sents)
                dec_attention_mask = torch.cat([dec_attention_mask[:, :, -1:, :], dec_attention_mask[:, :, -1:, -1:]], dim=-1)
                cross_attention_mask = cross_attention_mask[:, :, -1:, :]
            gen_len += 1
            # unfinished_sents.mul_(tokens_to_add.ne(tokenizer.get_sentinel_id(1)).long())
            unfinished_sents.mul_(tokens_to_add.ne(tokenizer.eod_id).long())
        scores = torch.stack(scores).transpose(1, 0) # batch size, decoder length
        # scores = scores.sum(dim=-1) / torch.count_nonzero(scores, dim=-1) # batch size
        scores_list = []
        scores = scores.sum(dim=-1) / (scores != 0).sum(dim=-1)
        scores = scores.exp()
        gathered_scores = [torch.zeros_like(scores) for _ in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather(gathered_scores, scores, mpu.get_data_parallel_group())  # gather all scores
        all_logits.extend(gathered_scores)
        gathered_preds = [torch.zeros_like(output_ids) for _ in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
        all_preds.extend(gathered_preds)
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_logits = torch.cat(all_logits, dim=0).cpu().tolist()
    predictions = []
    for i, (p, s) in enumerate(zip(all_preds, all_logits)):
        prediction = tokenizer.decode(p[1:-1])
        predictions.append({
            'prediction': prediction,
            'score': s,
        })
    return predictions        

def request_for_doc(question):
    get_result = requests.get(host, params={'query': question, 'num': 100})
    docs_scores = json.loads(get_result.text)
    docs = docs_scores['results']
    scores = docs_scores['score']
    return docs, scores

def broadcast_data(data, device, shape_len, dtype, group):
    shape_tensor = torch.zeros(shape_len, device=device).long()
    if mpu.get_model_parallel_rank() == 0:
        shape = data.shape
        shape_tensor = torch.tensor(shape, dtype=torch.long, device=device)
        torch.distributed.broadcast(shape_tensor, 0, group)
        assert data is not None
        torch.distributed.broadcast(data, 0, group)
    else:
        torch.distributed.broadcast(shape_tensor, 0, group)
        data = torch.zeros(shape_tensor.tolist(), dtype=dtype, device=device)
        torch.distributed.broadcast(data, 0, group)
    return data

if __name__ == "__main__":
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()
    # setup tokenizer
    # global tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))
    prompt_config = None
    if args.prompt_tune:
        print_rank_0("use prompt. load prompt config ...")
        with open(args.prompt_config, "r") as f:
            prompt_config = json.load(f)
            init_from_vocab = prompt_config.get("init_from_vocab", False)
            init_from_label = prompt_config.get("init_from_label", False)
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                if init_from_vocab:
                    raise NotImplementedError    
                    if torch.distributed.get_rank() == 0:
                        print(extra_id_list)
                        print(tokenizer.decode(extra_id_list))
                    prompt_config[t]["init_ids"].extend(extra_id_list)
                elif init_from_label:
                    all_label_ids = prompt_config["all_label_ids"]
                    repeat_num = 100 // len(all_label_ids) + 1
                    extra_id_list = (all_label_ids * repeat_num)[:100]
                    if torch.distributed.get_rank() == 0:
                        print(extra_id_list)
                        print(tokenizer.decode(extra_id_list))
                    prompt_config[t]["init_ids"].extend(extra_id_list)
                else:
                    prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)
    # Model, optimizer, and learning rate.
    print_rank_0("Load model ...")
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, None, prompt_config)
    model.eval()
    print_rank_0("Load model over.")
    batch_size = 16
    while True:
        ans_predictions = []
        no_ans_predictions = []
        if mpu.get_model_parallel_rank() == 0:
            question = input(">>Please input question:")
            # question = "capital of Shanxi"
            question = question.strip()
            if question == '':
                continue
            start_time = time.time()
            print_rank_0("Question: {}".format(question))
            print_rank_0("request for relevent docs ...")
            # docs, doc_scores = request_for_doc(question)
            qa_start_time = time.time()
            print_rank_0(f"get {len(docs)} docs. generate answers from docs ...")
            print_rank_0(f"for question \"{question}\"")
            # for i, (doc, doc_score) in enumerate(zip(docs, doc_scores)):
            max_doc = min(10, len(docs))
            max_doc_tensor = torch.tensor(max_doc, dtype=torch.long, device=device)
            if mpu.model_parallel_is_initialized():
                torch.distributed.barrier()
                torch.distributed.broadcast(max_doc_tensor, 0, mpu.get_model_parallel_group())
        else:  # other model rank
            torch.distributed.barrier()
            max_doc_tensor = torch.tensor(0, dtype=torch.long, device=device)
            # print(max_doc_tensor)
            torch.distributed.broadcast(max_doc_tensor, 0, mpu.get_model_parallel_group())
            max_doc = max_doc_tensor.item()
        for i in range(0, max_doc, batch_size):
            if mpu.get_model_parallel_rank() == 0:
                # construct data
                end = min(i + batch_size, max_doc)
                batch_docs = [docs[j]['text'] for j in range(i, end)]
                batch_doc_scores = [doc_scores[j] for j in range(i, end)]
                model_batch = construct_input(question, batch_docs, tokenizer, args)
                for key, (shape_len, dtype)in data_name_info.items():
                    data = model_batch[key]
                    current_data = data.to(device)
                    if mpu.model_parallel_is_initialized():
                        model_batch[key] = broadcast_data(current_data, device, shape_len, dtype, mpu.get_model_parallel_group())
            else:
                model_batch = {}
                for key, (shape_len, dtype)in data_name_info.items():
                    model_batch[key] = broadcast_data(None, device, shape_len, dtype, mpu.get_model_parallel_group())
            predictions = generate(args, model, model_batch, tokenizer, device)
            if mpu.get_model_parallel_rank() == 0:
                for j, p in enumerate(predictions):
                    doc = docs[i + j]
                    doc_score = batch_doc_scores[j]
                    final_score = p['score']
                    if p['prediction'] == 'no answer':
                        no_ans_predictions.append({
                            'prediction': p['prediction'],
                            'doc': doc['text'],
                            'p_score': p['score'],
                            'd_score': doc_score,
                            'f_score': final_score,
                            'title': doc['title']
                        })
                    else:
                        ans_predictions.append({
                            'prediction': p['prediction'],
                            'doc': doc['text'],
                            'p_score': p['score'],
                            'd_score': doc_score,
                            'f_score': final_score,
                            'title': doc['title']
                        })
                    # print_rank_0(f"doc {i}: \nprediction: {p['prediction']}\ndoc: {doc['text']}")
        ans_predictions.sort(key=lambda p: p['f_score'], reverse=True)
        no_ans_predictions.sort(key=lambda p: p['f_score'], reverse=True)
        cnt = 0
        for p in no_ans_predictions[::-1]:
            print_rank_0("prediction {}: {}".format(max_doc - cnt, p['prediction']))
            print_rank_0("final score: {:.3f}, doc score: {:.3f}, qa score: {:.3f}".format(p['f_score'], p['d_score'], p['p_score']))
            print_rank_0("title: {}".format(p['title']))
            print_rank_0("doc: {}".format(p['doc']))
            print_rank_0('')
            cnt += 1
        for p in ans_predictions[::-1]:
            print_rank_0("prediction {}: {}".format(max_doc - cnt, p['prediction']))
            print_rank_0("final score: {:.3f}, doc score: {:.3f}, qa score: {:.3f}".format(p['f_score'], p['d_score'], p['p_score']))
            print_rank_0("title: {}".format(p['title']))
            print_rank_0("doc: {}".format(p['doc']))
            print_rank_0('')
            cnt += 1
        print_rank_0('Question: {}'.format(question))
        if mpu.get_model_parallel_rank() == 0:
            print_rank_0('time for getting doc: {:.3} s. time for qa: {} s'.format(qa_start_time - start_time, time.time() - qa_start_time))
        if mpu.model_parallel_is_initialized():
            torch.distributed.barrier() # sychronize

