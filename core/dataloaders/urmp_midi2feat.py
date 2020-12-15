from torch.utils.data import Dataset
import random
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import numpy as np
from core import utils
import torch
from pyhocon import ConfigTree


@dataclass
class Sample:
    midi_path: str
    pose_path: str
    start_time: float
    duration: float


class URMPMIDI2FeatDataset(Dataset):
    # PAD_IDX = 240
    SOS_IDX = 10896
    # EOS_IDX = 242

    BODY_PARTS = {
        'body25': 25
    }

    def __init__(
            self,
            split_csv_dir: str,
            duration: float,
            fps=29.97,
            pose_layout='body25',
            split='train',
            duplication=100
    ):
        self.duration = duration
        self.fps = fps
        self.pose_layout = pose_layout
        self.split_csv_dir = Path(split_csv_dir)
        self.duplication = duplication

        assert split in ['train', 'val', 'test'], split
        self.split = split
        self.csv_path = self.split_csv_dir / f'{split}.csv'
        self.df = pd.read_csv(str(self.csv_path))

        self.samples = self.build_samples_from_dataframe(self.df)

        if split == 'train':
            self.samples *= duplication
        else:
            self.samples = self.split_val_samples_into_small_pieces(self.samples, duration)

        self.num_frames = int(duration * fps)
        self.num_events = int(duration * 100)
        self.body_part = self.BODY_PARTS[pose_layout]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.split == 'train':
            start_time = random.random() * (sample.duration - 1.5 * self.duration)
        else:
            start_time = 0.

        start_time += sample.start_time

        start_frame = int(start_time * self.fps)

        if self.split == 'test':
            # midi = [0] * self.num_events
            # midi[0] = self.SOS_IDX
            # midi[-1] = self.EOS_IDX
            midi = [self.SOS_IDX] * (self.num_events + 1)
            # midi[0] = self.SOS_IDX
        else:
            # midi = utils.io.midi_to_list(sample.midi_path, start_time, self.duration)
            midi = utils.io.read_midi_from_npy(sample.midi_path, int(start_time * 100), self.num_events)
            # midi = self.pad_midi_events(midi)
            midi = np.hstack(([self.SOS_IDX], midi))

        pose = utils.io.read_pose_from_npy(sample.pose_path, start_frame, self.num_frames, part=self.body_part)

        midi = torch.tensor(midi)
        pose = torch.from_numpy(pose)

        return pose, midi

    @staticmethod
    def build_samples_from_dataframe(df: pd.DataFrame):
        samples = []
        for _i, row in df.iterrows():
            sample = Sample(
                row.midi_path,
                row.pose_path,
                row.start_time,
                row.duration
            )
            samples.append(sample)
        return samples

    @staticmethod
    def split_val_samples_into_small_pieces(samples, duration: float):
        new_samples = []

        for sample in samples:
            stop = sample.duration
            pieces = np.arange(0., stop, duration)[:-1]
            for new_start in pieces:
                new_samples.append(Sample(
                    midi_path=sample.midi_path,
                    pose_path=sample.pose_path,
                    start_time=new_start,
                    duration=duration,
                ))

        return new_samples

    @classmethod
    def from_cfg(cls, cfg: ConfigTree, split: str = 'train'):
        split_csv_dir = cfg.get_string('dataset.split_csv_dir')
        duration = cfg.get_float('dataset.duration')
        fps = cfg.get_float('dataset.fps')
        pose_layout = cfg.get_string('dataset.pose_layout')
        duplication = cfg.get_int('dataset.duplication')
        return cls(split_csv_dir, duration, fps=fps, pose_layout=pose_layout, split=split, duplication=duplication)
