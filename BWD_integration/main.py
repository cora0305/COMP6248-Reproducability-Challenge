import json
import random
import argparse
import numpy as np
from config import Config
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import random
import torch.nn.functional as F
import math
from Model import build_model
from utils import EnvHndler, bool_flag
from trainer import Trainer
from evaluator import Evaluator

## Raise an Error for every floating-point operation error: devition by zero, overfelow, underflow, and invalid operation
np.seterr(all='raise')

def set_seed(config):
    """Set seed"""
    if config.env_seed == -1 :
        config.env_seed = np.random.randint(1_000_000_000)
    seed = config.env_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f'set seed to {config.env_seed}')

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=int, default=0,
                        help="Experiment ID (if 0, generate a new one)")


    # model parameters
    parser.add_argument("--model_type", type=str, default="Transformers",
                        help="Set the model type {Transformers, Performers}")
    parser.add_argument("--model_dim", type=int, default=512,
                        help="Embedding and other layers size")
    parser.add_argument("--num_enc_layer", type=int, default=6,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--num_dec_layer", type=int, default=6,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--forward_expansion", type=int, default=4,
                        help="The ratio of the hidden size of the Feed Forward net to model_dim")
    parser.add_argument("--max_position", type=int, default=4096,
                        help="The maximum number of positions we have in the data")
    parser.add_argument("--num_head", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")

    # training parameters
    parser.add_argument("--env_seed", type=int, default=0,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Number of epochs")

    # reload data
    parser.add_argument("--train_reload_size", type=int, default=10000,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--test_reload_size", type=int, default=500,
                        help="Reloaded training set size (-1 for everything)")

    # reload pretrained model / checkpoint
    parser.add_argument("--load_model", type=bool_flag, default=False,
                        help="Load a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")


    return parser



def main(args):
    """
    Main function contains the main procedure of training and evaluation.
    It take args as its argument
        args: a parser object that contains main configurations of the current experiment
    """
    config = Config(args)
    set_seed(config)
    logger = config.get_logger()
    env = EnvHndler(config)
    # Clear the cash of CUDA
    torch.cuda.empty_cache()
    model = build_model(config)
    trainer = Trainer(config, env, model)
    evaluator = Evaluator(trainer)
    # evaluation
    if config.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # training
    for epoch in range(config.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0
        torch.cuda.empty_cache()
        while trainer.n_equations < trainer.epoch_size:
            # training steps
            torch.cuda.empty_cache()
            trainer.enc_dec_step()
            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))

        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        if epoch%10 == 0:
            while True:
                t = input("Continue training? [y/n]")
                if t not in ['y', 'n']:
                    print('Invalid input')
                    continue
                elif t == 'y':
                    break
                else:
                    exit()

if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()
    args = args.parse_args()
    print('Hey')
    main(args)
