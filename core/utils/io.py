from core.performance_rnn.sequence import NoteSeq, EventSeq, ControlSeq
from pretty_midi import PrettyMIDI
import numpy as np
import torch
from typing import List, Optional
from scipy.io import wavfile


def midi_to_array(filename: str, start_time: float, duration: float) -> np.ndarray:
    pm = PrettyMIDI(filename)

    pm.adjust_times(np.array(start_time, start_time + duration), np.array([0., duration]))

    # ensure one midi contain only on instrument
    note_seq = NoteSeq(notes=pm.instruments[0].notes)
    event_seq = EventSeq.from_note_seq(note_seq)
    return event_seq.to_array()


def midi_to_list(filename: str, start_time: float, duration: float) -> List[int]:
    pm = PrettyMIDI(filename)

    pm.adjust_times(np.array([start_time, start_time + duration]), np.array([0., duration]))

    # ensure one midi contain only on instrument
    note_seq = NoteSeq(notes=pm.instruments[0].notes)
    event_seq = EventSeq.from_note_seq(note_seq)
    return event_seq.to_list()


def pm_to_list(pm: PrettyMIDI, use_control=False) -> (List[int], Optional[np.ndarray]):
    """
    :return:
    """
    # ensure one midi contain only on instrument
    note_seq = NoteSeq(notes=pm.instruments[0].notes)
    event_seq = EventSeq.from_note_seq(note_seq)
    if use_control:
        control_seq = ControlSeq.from_event_seq(event_seq)
        # control = control_seq.to_array()
        control = control_seq.to_pitch_histogram_array()
    else:
        control = None

    return event_seq.to_list(), control


def read_pose_from_tensor(filename: str, start: int, length: int) -> torch.Tensor:
    data: torch.Tensor = torch.load(filename)  # [T, C, V, M]
    data = data[start: start + length]
    data = data.transpose(0, 1)
    return data


def read_pose_from_npy(filename: str, start: int, length: int, part=25) -> np.ndarray:
    data = np.load(filename, mmap_mode='c')  # [T, C, V, M]

    if len(data) < start + length:
        start = 0

    if part > 0:
        data = data[start: start + length, :, :part]
    else:
        data = data[start: start + length]

    data = data.astype(np.float32)
    data = np.transpose(data, (1, 0, 2, 3))

    return data


def read_pose_from_npy_by_time(filename: str, start_time: float, duration: float, fps: float, part=25):
    start_frame = int(start_time * fps)
    length = int(duration * fps)


def read_midi_from_npy(filename: str, start: int, length: int) -> np.ndarray:
    data = np.load(filename, mmap_mode='c')
    # print(data.shape)
    data = data[0, start: start + length]
    return data


def read_midi(filename: str, start_time: float, duration: float) -> PrettyMIDI:
    """

    :param filename:
    :param start_time: in seconds
    :param duration: in seconds
    :return:
    """
    pm = PrettyMIDI(filename)

    pm.adjust_times(
        [start_time, start_time + duration],
        [0., duration]
    )

    return pm


WAV_RANGE = 2 ** -15  # [-1.0, 1.0]


def read_wav(filename: str, start_index: int, length: int):
    rate, audio = wavfile.read(filename, mmap=True)
    audio = audio[start_index: start_index + length]
    audio = audio.astype(np.float32)
    audio = audio * WAV_RANGE
    # audio = audio[None, :, None]  # [T] -> [1, T, 1]
    return audio


def read_feature_from_npy(filename: str, start_frame: int, length: int) -> np.ndarray:
    data = np.load(filename, mmap_mode='c')
    data = data[start_frame: start_frame + length]
    T, C = data.shape

    res = np.zeros((length, C), dtype=np.float32)
    new_length = min(T, length)
    res[:new_length] = data # Padding

    res = np.transpose(res, (1, 0))
    return res
