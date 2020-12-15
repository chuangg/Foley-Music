from pyhocon import ConfigTree
from torch.utils.data import DataLoader

import torch


class DataLoaderFactory:

    # change from other project
    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg
        self.num_gpus = max(1, torch.cuda.device_count())

    def build(self, split='train'):
        dset = self.cfg.get_string('dataset.dset')

        if dset == 'urmp':
            from .urmp import URMPDataset
            ds = URMPDataset.from_cfg(self.cfg, split=split)
        elif dset == 'urmp_midi2feat':
            from .urmp_midi2feat import URMPMIDI2FeatDataset
            ds = URMPMIDI2FeatDataset.from_cfg(self.cfg, split=split)
        elif dset == 'atinpiano':
            from .urmp_music_transformer import URMPDataset
            ds = URMPDataset.from_cfg(self.cfg, split=split)
        elif dset == 'youtube_atinpiano':
            from .youtube_dataset import YoutubeDataset
            ds = YoutubeDataset.from_cfg(self.cfg, split=split)
        elif dset == 'music21_segment':
            from .youtube_dataset import YoutubeSegmentDataset
            ds = YoutubeSegmentDataset.from_cfg(self.cfg, split=split)
        elif dset == 'youtube_urmp':
            from .youtube_dataset import YoutubeURMPDataset
            ds = YoutubeURMPDataset.from_cfg(self.cfg, split=split)
        else:
            raise Exception

        loader = DataLoader(
            ds,
            batch_size=self.cfg.get_int('batch_size') * self.num_gpus,
            num_workers=self.cfg.get_int('num_workers') * self.num_gpus,
            shuffle=(split == 'train')
        )

        print('Real batch size:', loader.batch_size)

        return loader
