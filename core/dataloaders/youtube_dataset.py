from torch.utils.data import Dataset
from typing import List, Optional, Dict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import random
from core import utils
import copy
import torch
from pretty_midi import PrettyMIDI
from core.utils.urmp import URMPSepInfo
from pyhocon import ConfigTree
import copy


@dataclass
class Sample:
    """
    可以只加载midi，pose，audio其中一个，或全都加载
    """
    vid: str
    start_time: float
    duration: float
    row: dict
    midi_path: Optional[str] = None
    pose_path: Optional[str] = None
    audio_path: Optional[str] = None
    rgb_path: Optional[str] = None
    flow_path: Optional[str] = None


class YoutubeDataset(Dataset):
    SOS_IDX = 240
    EOS_IDX = 241
    PAD_IDX = 242

    BODY_PARTS = {
        'body25': 25,
    }

    def __init__(
            self,
            split_csv_dir: str,
            streams: Dict[str, str],
            duration=6.0,
            duplication=100,
            fps=29.97,
            events_per_sec=20,
            audio_rate=22050,
            random_shift_rate=0.2,
            pose_layout='body25',
            split='train',
            max_seq=512,
    ):
        self.split_csv_dir = Path(split_csv_dir)
        self.streams = streams
        self.duration = duration
        self.duplication = duplication
        self.fps = fps
        self.events_per_sec = events_per_sec
        self.audio_rate = audio_rate
        self.pose_layout = pose_layout
        self.random_shift_rate = random_shift_rate

        self.audio_duration = self.duration + 0.5  # sound is slightly longer than action

        assert split in ['train', 'val', 'test'], split
        self.split = split

        self.csv_path = self.split_csv_dir / f'{split}.csv'
        self.df = pd.read_csv(str(self.csv_path))

        self.samples = self.build_samples_from_dataframe(
            self.df,
            self.streams,
        )

        if split == 'train':
            self.samples *= duplication
        else:
            self.samples = self.split_val_samples_into_small_pieces(self.samples, duration)

        self.num_frames = int(duration * fps)
        self.num_events = max_seq
        self.num_audio_frames = int(self.audio_duration * audio_rate)  
        self.body_part = self.BODY_PARTS.get(pose_layout, -1) 

        self.use_pose = 'pose' in self.streams
        self.use_midi = 'midi' in self.streams
        self.use_audio = 'audio' in self.streams
        self.use_control = 'control' in self.streams
        self.use_rgb = 'rgb' in self.streams
        self.use_flow = 'flow' in self.streams

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.split == 'train':
            start_time = random.random() * (sample.duration - 1.5 * self.duration)
        else:
            start_time = 0.

        start_time += sample.start_time

        result = {}

        start_frame = int(start_time * self.fps)

        if self.use_pose:

            pose = utils.io.read_pose_from_npy(
                sample.pose_path, start_frame, self.num_frames, part=self.body_part
            )

            if self.split == 'train':
                pose = utils.pose.random_move(pose)

            result['pose'] = torch.from_numpy(pose)

        if self.use_rgb:
            rgb = utils.io.read_feature_from_npy(
                sample.rgb_path, start_frame, self.num_frames
            )
            result['rgb'] = torch.from_numpy(rgb.astype(np.float32))

        if self.use_flow:
            flow = utils.io.read_feature_from_npy(
                sample.flow_path, start_frame, self.num_frames
            )
            result['flow'] = torch.from_numpy(flow.astype(np.float32))

        if self.use_midi:
            pm = utils.io.read_midi(
                sample.midi_path, start_time, self.audio_duration
            )
            tgt = copy.deepcopy(pm)

            # if self.split == 'train':
            #     tgt = self.midi_transform(tgt)

            """
            input:  1, 2, 3, [-1]
            output: 2, 3, 4
            """
            midi_x, control = utils.io.pm_to_list(tgt, use_control=self.use_control)  # Input midi
            # midi_x = midi_x[:-1]

            midi_y, _ = utils.io.pm_to_list(pm, use_control=False)  # Target midi, no predict control
            # midi_y = midi_y[1:]

            midi_x, control = self.pad_midi_events(midi_x, control=control)
            midi_y, _ = self.pad_midi_events(midi_y, control=None)

            # padding first
            midi_x = midi_x[:-1]
            midi_y = midi_y[1:]

            result['midi_x'] = torch.LongTensor(midi_x)
            result['midi_y'] = torch.LongTensor(midi_y)

            if self.use_control:
                # print('=' * 100)
                # print(control.shape)
                control = control[:-1]  # keep the same as midi_x
                result['control'] = torch.from_numpy(control)

        if self.use_audio:
            start_index = int(start_time * self.audio_rate)
            audio = utils.io.read_wav(
                sample.audio_path, start_index, self.num_audio_frames
            )
            result['audio'] = torch.from_numpy(audio)

        if self.split != 'train':
            result['start_time'] = start_time
            result['index'] = index

        return result

    def read_midi(self, start_time: float, duraiton: float):
        pass

    def get_samples_by_indices(self, indices) -> List[Sample]:
        result = []
        for index in indices:
            sample = self.samples[index]
            result.append(sample)
        return result

    def pad_midi_events(
            self,
            midi: List[int],
            control: Optional[np.ndarray] = None
    ) -> (List[int], Optional[np.ndarray]):
        new_midi = [self.SOS_IDX] + midi + [self.EOS_IDX]
        # new_midi = [self.SOS_IDX] + midi
        if control is not None:
            control = np.pad(control, ((1, 1), (0, 0)), 'constant')

        num_events = self.num_events + 1

        if len(new_midi) > num_events:
            new_midi = new_midi[:num_events]
            new_midi[-1] = self.EOS_IDX

            if control is not None:
                control = control[:num_events]
                control[-1, :] = 0

        elif len(new_midi) < num_events:
            pad = num_events - len(new_midi)
            new_midi = new_midi + [self.PAD_IDX] * pad

            if control is not None:
                control = np.pad(control, ((0, pad), (0, 0)), 'constant')

        return new_midi, control

    def midi_transform(self, pm: PrettyMIDI):
        notes = pm.instruments[0].notes  # Ref
        num_notes = len(notes)
        indices = random.sample(range(num_notes), int(self.random_shift_rate * num_notes))

        def get_random_number():
            return (random.random() - 0.5) * 0.2

        for index in indices:
            notes[index].start += get_random_number()
            notes[index].end += get_random_number()

        return pm

    @staticmethod
    def build_samples_from_dataframe(
            df: pd.DataFrame,
            streams: Dict[str, str],
    ):
        new_streams = {k: Path(v) for k, v in streams.items()}
        samples = []
        for _i, row in df.iterrows():
            sample = Sample(
                row.vid,
                row.start_time,
                row.duration,
                row.to_dict()
            )

            vid = row.vid
            if 'midi' in new_streams:
                midi_path = new_streams['midi'] / f'{vid}.midi'
                if not midi_path.is_file():
                    midi_path = new_streams['midi'] / f'{vid}.mid'
                sample.midi_path = str(midi_path)

            if 'pose' in streams:
                pose_path = new_streams['pose'] / f'{vid}.npy'
                sample.pose_path = str(pose_path)

            if 'audio' in streams:
                audio_path = new_streams['audio'] / f'{vid}.wav'
                sample.audio_path = str(audio_path)

            if 'rgb' in streams:
                rgb_path = new_streams['rgb'] / f'{vid}.npy'
                sample.rgb_path = rgb_path

            if 'flow' in streams:
                flow_path = new_streams['flow'] / f'{vid}.npy'
                sample.flow_path = flow_path

            samples.append(sample)
        return samples

    @staticmethod
    def split_val_samples_into_small_pieces(samples, duration: float):
        new_samples = []

        for sample in samples:
            stop = sample.duration
            pieces = np.arange(0., stop, duration)[:-1]
            for new_start in pieces:
                new_sample = copy.deepcopy(sample)
                new_sample.start_time = new_start
                new_sample.duration = duration
                # new_samples.append(Sample(
                #     vid=sample.vid,
                #     audio_path=sample.audio_path,
                #     midi_path=sample.midi_path,
                #     pose_path=sample.pose_path,
                #     start_time=new_start,
                #     duration=duration,
                # ))
                new_samples.append(new_sample)

        return new_samples

    @classmethod
    def from_cfg(cls, cfg: ConfigTree, split='train'):
        return cls(
            cfg.get_string('dataset.split_csv_dir'),
            cfg.get_config('dataset.streams'),
            split=split,
            duration=cfg.get_float('dataset.duration'),
            duplication=cfg.get_int('dataset.duplication'),
            fps=cfg.get_float('dataset.fps'),
            events_per_sec=cfg.get_int('dataset.events_per_sec'),
            random_shift_rate=cfg.get_float('dataset.random_shift_rate', 0.2),
            pose_layout=cfg.get_string('dataset.pose_layout')
        )

    def __len__(self):
        return len(self.samples)


class YoutubeSegmentDataset(YoutubeDataset):
    """
    针对特殊情况，{vid}_{start frame}_{end frame}.mp4
    """

    @staticmethod
    def build_samples_from_dataframe(
            df: pd.DataFrame,
            streams: Dict[str, str],
    ):
        new_streams = {k: Path(v) for k, v in streams.items()}
        samples = []
        for _i, row in df.iterrows():
            sample = Sample(
                row.vid,
                row.start_time,
                row.duration
            )

            vid = row.vid
            if 'midi' in new_streams:
                midi_path = new_streams['midi'] / f'{vid}.mid'
                sample.midi_path = str(midi_path)

            if 'pose' in streams:
                parts = vid.split('_')
                real_vid = '_'.join(parts[:-2])
                pose_path = new_streams['pose'] / f'{real_vid}.npy'
                sample.pose_path = str(pose_path)

            if 'audio' in streams:
                audio_path = new_streams['audio'] / f'{vid}.wav'
                sample.audio_path = str(audio_path)

            samples.append(sample)
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.split == 'train':
            start_time = random.random() * (sample.duration - 1.5 * self.duration)
        else:
            start_time = 0.

        start_time += sample.start_time

        result = {}

        if self.use_pose:
            # start_frame = int(start_time * self.fps)
            parts = sample.vid.split('_')
            start_frame = int(parts[-2])
            pose = utils.io.read_pose_from_npy(
                sample.pose_path, start_frame, self.num_frames, part=self.body_part
            )

            if self.split == 'train':
                pose = utils.pose.random_move(pose)

            result['pose'] = torch.from_numpy(pose)

        if self.use_midi:
            pm = utils.io.read_midi(
                sample.midi_path, start_time, self.duration + 1.  # make sound longer
            )
            tgt = copy.deepcopy(pm)

            # if self.split == 'train':
            #     tgt = self.midi_transform(tgt)

            """
            input:  1, 2, 3, [-1]
            output: 2, 3, 4
            """
            midi_x, control = utils.io.pm_to_list(tgt, use_control=self.use_control)  # Input midi
            midi_x = midi_x[:-1]
            if self.use_control:
                control = control[:-1]  # [T, D], D=24
                # print('control', control.shape)

            midi_y, _ = utils.io.pm_to_list(pm, use_control=False)  # Target midi, no predict control
            midi_y = midi_y[1:]

            midi_x, control = self.pad_midi_events(midi_x, control=control)
            midi_y, _ = self.pad_midi_events(midi_y, control=None)

            result['midi_x'] = torch.LongTensor(midi_x)
            result['midi_y'] = torch.LongTensor(midi_y)

            if self.use_control:
                # print('=' * 100)
                # print(control.shape)
                result['control'] = torch.from_numpy(control)

        if self.use_audio:
            start_index = start_time * self.audio_rate
            audio = utils.io.read_wav_for_sound_net(
                sample.audio_path, start_index, self.num_audio_frames
            )
            result['audio'] = torch.from_numpy(audio)

        if self.split != 'train':
            result['start_time'] = start_time
            result['index'] = index

        return result


class YoutubeURMPDataset(YoutubeDataset):

    @staticmethod
    def build_samples_from_dataframe(
            df: pd.DataFrame,
            streams: Dict[str, str],
    ):
        new_streams = {k: Path(v) for k, v in streams.items()}
        samples = []
        for _i, row in df.iterrows():
            urmp_sep_info = URMPSepInfo.from_row(row)

            sample = Sample(
                urmp_sep_info.vid,
                row.start_time,
                row.duration,
                row.to_dict()
            )

            if 'midi' in new_streams:
                midi_path = new_streams['midi'] / urmp_sep_info.midi_filename
                sample.midi_path = str(midi_path)

            if 'pose' in streams:
                pose_path = new_streams['pose'] / urmp_sep_info.pose_filename
                sample.pose_path = str(pose_path)

            if 'audio' in streams:
                audio_path = new_streams['audio'] / urmp_sep_info.audio_filename
                sample.audio_path = str(audio_path)

            if 'rgb' in streams:
                rgb_path = new_streams['rgb'] / urmp_sep_info.feature_filename
                sample.rgb_path = rgb_path

            if 'flow' in streams:
                flow_path = new_streams['flow'] / urmp_sep_info.feature_filename
                sample.flow_path = flow_path

            samples.append(sample)
        return samples
