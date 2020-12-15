from core.engine import BaseEngine
from pyhocon import ConfigTree
from core.dataloaders import DataLoaderFactory
from core.models import ModelFactory
from torchpie.environment import experiment_path
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from torchpie.utils.checkpoint import save_checkpoint
from torchpie.meters import AverageMeter
from torchpie.logging import logger
import time
from core.dataloaders.youtube_dataset import YoutubeDataset
from core.criterion import SmoothCrossEntropyLoss
from core.optimizer import CustomSchedule
from core.metrics import compute_epiano_accuracy
from pprint import pprint


class Engine(BaseEngine):

    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg
        print(cfg)
        self.summary_writer = SummaryWriter(log_dir=experiment_path)
        self.model_builder = ModelFactory(cfg)
        self.dataset_builder = DataLoaderFactory(cfg)

        self.train_ds = self.dataset_builder.build(split='train')
        self.test_ds = self.dataset_builder.build(split='val')
        self.ds: YoutubeDataset = self.train_ds.dataset

        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.val_criterion = nn.CrossEntropyLoss(
            ignore_index=self.ds.PAD_IDX
        )
        self.model: nn.Module = self.model_builder.build(device=torch.device('cuda'), wrapper=nn.DataParallel)
        optimizer = optim.Adam(self.model.parameters(), lr=0., betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = CustomSchedule(
            self.cfg.get_int('model.emb_dim'),
            optimizer=optimizer,
        )

        self.num_epochs = cfg.get_int('num_epochs')

        logger.info(f'Use control: {self.ds.use_control}')

    def train(self, epoch=0):
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        num_iters = len(self.train_ds)
        self.model.train()
        for i, data in enumerate(self.train_ds):
            midi_x, midi_y = data['midi_x'], data['midi_y']

            if self.ds.use_pose:
                feat = data['pose']
            elif self.ds.use_rgb:
                feat = data['rgb']
            elif self.ds.use_flow:
                feat = data['flow']
            else:
                raise Exception('No feature!')

            feat, midi_x, midi_y = (
                feat.cuda(non_blocking=True),
                midi_x.cuda(non_blocking=True),
                midi_y.cuda(non_blocking=True)
            )

            if self.ds.use_control:
                control = data['control']
                control = control.cuda(non_blocking=True)
            else:
                control = None

            output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

            loss = self.train_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            acc = compute_epiano_accuracy(output, midi_y, pad_idx=self.ds.PAD_IDX)

            batch_size = len(midi_x)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)

            logger.info(
                f'Train [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
                f'{loss_meter}\t{acc_meter}'
            )
        self.summary_writer.add_scalar('train/loss', loss_meter.avg, epoch)
        self.summary_writer.add_scalar('train/acc', acc_meter.avg, epoch)
        return loss_meter.avg

    def test(self, epoch=0):
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')
        num_iters = len(self.test_ds)
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.test_ds):
                midi_x, midi_y = data['midi_x'], data['midi_y']

                if self.ds.use_pose:
                    feat = data['pose']
                elif self.ds.use_rgb:
                    feat = data['rgb']
                elif self.ds.use_flow:
                    feat = data['flow']
                else:
                    raise Exception('No feature!')

                feat, midi_x, midi_y = (
                    feat.cuda(non_blocking=True),
                    midi_x.cuda(non_blocking=True),
                    midi_y.cuda(non_blocking=True)
                )

                if self.ds.use_control:
                    control = data['control']
                    control = control.cuda(non_blocking=True)
                else:
                    control = None

                output = self.model(feat, midi_x, pad_idx=self.ds.PAD_IDX, control=control)

                """
                For CrossEntropy
                output: [B, T, D] -> [BT, D]
                target: [B, T] -> [BT]
                """
                loss = self.val_criterion(output.view(-1, output.shape[-1]), midi_y.flatten())

                acc = compute_epiano_accuracy(output, midi_y)

                batch_size = len(midi_x)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc.item(), batch_size)
                logger.info(
                    f'Val [{epoch}]/{self.num_epochs}][{i}/{num_iters}]\t'
                    f'{loss_meter}\t{acc_meter}'
                )
            self.summary_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('val/acc', acc_meter.avg, epoch)

        return loss_meter.avg

    @staticmethod
    def epoch_time(start_time: float, end_time: float):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            start_time = time.time()
            _train_loss = self.train(epoch)
            loss = self.test(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint(
                {
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                is_best=is_best,
                folder=experiment_path
            )

    def close(self):
        self.summary_writer.close()


def main():
    from torchpie.config import config as cfg
    print('=' * 100)
    pprint(cfg)
    print('=' * 100)
    engine = Engine(cfg)
    engine.run()
    engine.close()


if __name__ == '__main__':
    main()
