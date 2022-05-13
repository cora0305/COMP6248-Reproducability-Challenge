from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datahandler import create_data_loader
import os
import torch
import sympy as sp
from utils import to_cuda, timeout, TimeoutError

logger = getLogger()

BUCKET_LENGTH_SIZE = 5


def idx_to_sp(env, idx, return_infix=False):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    prefix = env.unclean_prefix(prefix)
    infix = env.prefix_to_infix(prefix)
    eq = sp.S(infix, locals=env.local_dict)
    return (eq, infix) if return_infix else eq


@timeout(5)
def check_valid_solution(env, src, tgt, hyp):
    """
    Check that a solution is valid.
    """
    f = env.local_dict['f']
    x = env.local_dict['x']

    valid = simplify(hyp - tgt, seconds=1) == 0
    if not valid:
        diff = src.subs(f(x), hyp).doit()
        diff = simplify(diff, seconds=1)
        valid = diff == 0

    return valid


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    src = idx_to_sp(env, eq['src'])
    tgt = idx_to_sp(env, eq['tgt'])
    hyp = eq['hyp']

    hyp_infix = [env.id2word[wid] for wid in hyp]

    try:
        hyp, hyp_infix = idx_to_sp(env, hyp, return_infix=True)
        is_valid = check_valid_solution(env, src, tgt, hyp)
        if is_valid_expr(hyp_infix):
            hyp_infix = str(hyp)

    except (TimeoutError, Exception) as e:
        e_name = type(e).__name__
        if not isinstance(e, InvalidPrefixExpression):
            logger.error(f"Exception {e_name} when checking hypothesis: {hyp_infix}")
        hyp = f"ERROR {e_name}"
        is_valid = False

    # update hypothesis
    f = env.local_dict['f']
    x = env.local_dict['x']
    eq['src'] = src.subs(f(x), 'f')  # hack to avoid pickling issues with lambdify
    eq['tgt'] = tgt
    eq['hyp'] = hyp_infix
    eq['is_valid'] = is_valid

    return eq

class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.model = trainer.model
        self.config = trainer.config
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                for task in [self.config.task]:
                    self.enc_dec_step(data_type, task, scores)

        return scores

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        config = self.config
        env = self.env
        transformer = self.model
        transformer.eval()
        assert config.eval_verbose in [0, 1]
        assert config.eval_verbose_print is False or config.eval_verbose > 0
        assert task in ['prim_bwd']

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # evaluation details
        if config.eval_verbose:
            eval_path = os.path.join(config.exp_dir, f"eval.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = create_data_loader(config=config,env = env, dtype = data_type)
        eval_size = len(iterator.dataset)

        for (x, len_x), (y, len_y), nb_ops in iterator:

            # print status
            if n_total.sum().item() % 100 < config.batch_size:
                logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len_y.max(), dtype=torch.long, device=len_y.device)
            pred_mask = alen[:, None] < len_y[None] - 1  # do not predict anything given the last target word
            t = y[1:].masked_select(pred_mask[:-1])
            assert len(t) == (len_y - 1).sum().item()

            # cuda
            x, len_x, y, len_y, t = to_cuda(self.config, x, len_x, y, len_y, t)

            # forward / loss
            encoded = transformer(mode = 'encode', x = x, len_x = len_x)
            decoded = transformer(mode = 'decode', y = y, len_y = len_y, encoded = encoded.transpose(0,1), len_enc = len_x)
            word_scores, loss = transformer(mode = 'predict', tensor = decoded, pred_mask = pred_mask, y = t, get_scores = True)

            # correct outputs per sequence / valid top-1 predictions
            o = torch.zeros_like(pred_mask, device=t.device)
            o[pred_mask] += word_scores.max(1)[1] == t
            valid = (o.sum(0) == len_y - 1).cpu().long()

            # export evaluation details
            if config.eval_verbose:
                for i in range(len(len_x)):
                    src = idx_to_sp(env, x[1:len_x[i] - 1, i].tolist())
                    tgt = idx_to_sp(env, y[1:len_y[i] - 1, i].tolist())
                    s = f"Equation {n_total.sum().item() + i} ({'Valid' if valid[i] else 'Invalid'})\nsrc={src}\ntgt={tgt}\n"
                    if config.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(t)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if config.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size or self.trainer.data_path
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total
        scores[f'{data_type}_{task}_acc'] = 100. * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            # logger.info(f"{i}: {n_valid[i].item()} / {n_total[i].item()} ({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_acc_{i}'] = 100. * n_valid[i].item() / max(n_total[i].item(), 1)
        # Deletes data on CUDA to free its memory
        del x, len_x, y, len_y, t, alen, o
