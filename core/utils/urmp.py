from dataclasses import dataclass
from typing import List
from pathlib import Path
import os


@dataclass
class URMPFolderInfo:
    vid: str
    piece: str
    instruments: List[str]

    @classmethod
    def from_folder(cls, filename: str):
        filename = os.path.basename(filename)
        data = filename.split('_')
        vid = data[0]
        piece = data[1]
        instruments = data[2:]
        return cls(vid, piece, instruments)

    @classmethod
    def from_midi(cls, filename: str):
        filename = os.path.basename(filename)
        name, ext = filename.split('.')
        assert ext == 'mid'
        data = name.split('_')

        # 第0位是Sco
        vid = data[1]
        piece = data[2]
        instruments = data[3:]
        return cls(vid, piece, instruments)

    @property
    def folder_name(self):
        return f'{self.vid}_{self.piece}_{self.instruments_str}'

    @property
    def video_name(self):
        return f'Vid_{self.vid}_{self.piece}_{self.instruments_str}.mp4'

    def get_video_seperation_name(self, index: int):
        return f'VidSep_{index + 1}_{self.instruments[index]}_{self.vid}_{self.piece}.mp4'

    @property
    def instruments_str(self):
        return '_'.join(self.instruments)


@dataclass
class URMPSepInfo:
    track_index: int  
    instrument: str
    vid: str
    piece: str

    @classmethod
    def from_sep(cls, filename: str):
        filename = os.path.basename(filename)
        parts = filename.split('.')[0].split('_') 

        # xxxSep -> 0
        index = int(parts[1])
        instrument = parts[2]
        vid = parts[3]
        piece = parts[4]
        return cls(
            index,
            instrument,
            vid,
            piece
        )

    @classmethod
    def from_row(cls, row: dict):
        return cls(
            row['track_index'],
            row['instrument'],
            '{:02d}'.format(row['vid']),
            row['piece']
        )

    def to_dict(self):
        return self.__dict__

    @property
    def midi_filename(self):
        return f'ScoSep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}.mid'

    @property
    def audio_filename(self):
        return f'AuSep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}.wav'

    @property
    def video_filename(self):
        return f'VidSep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}.mp4'

    @property
    def pose_filename(self):
        return f'PoseSep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}.npy'

    @property
    def feature_filename(self):
        return f'VidSep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}.npy'

    @property
    def folder_name(self):
        return f'Sep_{self.track_index}_{self.instrument}_{self.vid}_{self.piece}'