import torch
import argparse
from arguments import get_args
import sys
import json
import os
from tokenization_t5 import EncDecTokenizer
from utils import print_rank_0, save_rank_0
from utils import setup_model_and_optimizer, set_random_seed, initialize_distributed
import time
from test import request_for_doc, get_answer_from_model
import mpu

args_file = 'configs/model_laucher/xl.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
def set_distributed_environment(args):
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '8889')
    os.environ['LOCAL_RANK'] = str(args.rank)

def load_model(args):
    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()
    # setup tokenizer
    # global tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'spiece.model'))
    prompt_config = None
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, None, prompt_config)
    model.eval()
    print_rank_0("Load model over.")
    return model, device, tokenizer

def distributed_main(i, args):
    args.rank = i + 1
    set_distributed_environment(args)
    # Pytorch distributed.
    initialize_distributed(args)
    print(f"Rank {args.rank}: run distributed")
    model, device, tokenizer = load_model(args)
    batch_size = 16
    while True:
        ans, no_ans = get_answer_from_model(None, model, batch_size, tokenizer, device)



class ModelLauncher():
    def __init__(self, args_file):
        # Disable CuDNN.
        torch.backends.cudnn.enabled = False

        # Arguments.
        args = get_args()
        with open(args_file) as f:
            default_args = json.load(f)
        for key in default_args:
            setattr(args, key, default_args[key])
        if args.model_parallel_size > 1:
            cxt = torch.multiprocessing.spawn(
                fn = distributed_main,
                args = (args,),
                nprocs = args.model_parallel_size - 1,
                join = False,
            )
        set_distributed_environment(args)
        # Pytorch distributed.
        print("Rank 0: run distributed")
        initialize_distributed(args)
        self.model, self.device, self.tokenizer = load_model(args)
        self.batch_size = 16
    
    def generate_answer(self, question):
        if mpu.model_parallel_is_initialized() and mpu.get_model_parallel_rank() == 0:
            question = question.strip()
            if question == '':
                return None, None
        ans, no_ans = get_answer_from_model(question, self.model, self.batch_size, self.tokenizer, self.device)
        return ans, no_ans


if __name__ == "__main__":
    launcher = ModelLauncher(args_file)
    while True:
        question = input(">> Please input question: ")
        ans_predictions, no_ans_predictions = launcher.generate_answer(question)
        if ans_predictions is None:
            continue
        cnt = 0
        max_doc = len(ans_predictions) + len(no_ans_predictions)
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
