import os
import io
import re
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from datahandler import create_data_loader
import torch.optim as optim
from utils import to_cuda

logger = getLogger()

class Trainer(object):
    """
    This class is responsible for training the model
    """
    EQUATIONS = {}

    def __init__(self, config, env, model):
        """
        Initialize trainer.
        """
        # modules / params
        self.config = config
        self.model = model
        self.env = env
        self.n_equations = 0

        # epoch / iteration size
        self.epoch_size = config.epoch_size

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # stopping criterion used for early stopping
        if config.stopping_criterion != '':
            split = configs.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in config.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [('processed_e', 0)] +
            [('processed_w', 0)] +
            sum([
                [(x, []), (f'{x}-AVG-STOP-PROBS', [])]
                for x in [config.task]
            ], [])
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        self.data_path = {config.task: (config.train_dir, config.valid_dir, config.test_dir)}
        # create data loaders
        if not config.eval_only:
            self.dataloader = iter(create_data_loader(config, env, dtype = 'train'))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.clip_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), max_norm = self.config.clip_grad_norm)

        self.optimizer.step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k.upper().replace('_', '-'), np.mean(v))
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = ""
        s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats['processed_e'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_e'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """

        path = os.path.join(self.config.exp_dir, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
            'config': {k: v for k, v in self.config.__dict__.items()},
        }

        logger.warning("Saving Model parameters ...")
        data['model'] = self.model.state_dict()

        if include_optimizers:
            logger.warning("Saving Model optimizer ...")
            data['model_optimizer'] = self.optimizer.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.config.exp_dir, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.config.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.config.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        print(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        self.model.load_state_dict(data['model'])

        # reload optimizers
        logger.warning("Reloading checkpoint optimizer model ...")
        self.optimizer.load_state_dict(data['model_optimizer'])


        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.config.save_periodic > 0 and self.epoch % self.config.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None :
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()
        self.save_checkpoint('checkpoint')
        self.epoch += 1

    def enc_dec_step(self):
        """
        Encoding / decoding step.
        """
        transformer = self.model
        transformer.train()
        task = self.config.task
        # batch
        (x, len_x), (y, len_y), _ = next(self.dataloader)

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
        _, loss = transformer(mode = 'predict', tensor = decoded, pred_mask = pred_mask, y = t, get_scores = False)
        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += self.config.batch_size
        self.stats['processed_e'] += len_x.size(0)
        self.stats['processed_w'] += (len_x + len_y - 2).sum().item()
        # Deletes data on CUDA to free its memory
        del x, len_x, y, len_y, t, alen, pred_mask
