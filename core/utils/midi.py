import pretty_midi
from pretty_midi import PrettyMIDI
from scipy.io import wavfile
from core.performance_rnn.sequence import EventSeq, ControlSeq, Control
from torch import Tensor
import numpy as np
import torch
import joblib
import copy
from typing import List


def midi_to_wav(pm: PrettyMIDI, filename: str, rate=16000):
    waveform = pm.fluidsynth(fs=rate)
    wavfile.write(filename, rate, waveform)


def pm_to_wav(pm: PrettyMIDI, filename: str, rate=16000):
    waveform = pm.fluidsynth(fs=rate)
    wavfile.write(filename, rate, waveform)


def event_seq_to_wav(event_seq: EventSeq, filename: str, instrument='Violin', rate=16000):
    pm = event_seq_to_pm(event_seq, instrument)
    midi_to_wav(pm, filename, rate=rate)


def event_seq_to_pm(event_seq: EventSeq, instrument: str) -> PrettyMIDI:
    program = pretty_midi.instrument_name_to_program(instrument)
    pm = event_seq.to_note_seq().to_midi(program=program)
    return pm


def event_seq_to_midi(event_seq: EventSeq, filename: str, instrument='Violin'):
    pm = event_seq_to_pm(event_seq, instrument)
    pm.write(filename)


def tensor_to_pm(tensor: Tensor, instrument='Violin') -> PrettyMIDI:
    event_seq = tensor_to_event_seq(tensor)
    pm = event_seq_to_pm(event_seq, instrument)
    return pm


def tensor_to_event_seq(tensor: Tensor) -> EventSeq:
    event_seq = EventSeq.from_array(tensor.cpu().numpy())
    return event_seq


def tensor_to_waveform(tensor: Tensor, instrument='Acoustic Grand Piano', fs=22050) -> np.ndarray:
    pm = tensor_to_pm(tensor, instrument=instrument)
    waveform = pm.fluidsynth(fs=fs)
    return waveform


def ndarray_to_pm(array: np.ndarray, instrument='Acoustic Grand Piano') -> PrettyMIDI:
    event_seq = EventSeq.from_array(array)
    program = pretty_midi.instrument_name_to_program(instrument)
    pm = event_seq.to_note_seq().to_midi(program=program)
    return pm


def ndarray_to_waveform(array: np.ndarray, instrument='Acoustic Grand Piano', fs=22050) -> np.ndarray:
    pm = ndarray_to_pm(array, instrument=instrument)
    waveform = pm.fluidsynth(fs=fs)
    return waveform


def batch_tensor_to_batch_waveform(
        tensor: Tensor,
        instrument='Acoustic Grand Piano',
        fs=22050,
        n_jobs=None,
        length=22050 * 6,
) -> np.ndarray:
    # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    batch_array = tensor.detach().cpu().numpy()

    def f(array):
        # print(array.shape)
        waveform = np.zeros([length], dtype=np.float32)
        res = ndarray_to_waveform(array, instrument=instrument, fs=fs)

        res_length = min(len(res), length)
        waveform[:res_length] = res[:res_length]

        return waveform

    # waveforms = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(f)(array) for array in batch_array)
    waveforms = [f(array) for array in batch_array]  

    batch_waveform = np.stack(waveforms)

    return batch_waveform


def tensor_to_wav(tensor: Tensor, filename: str, instrument='Acoustic Grand Piano', fs=22050):
    pm = tensor_to_pm(tensor, instrument=instrument)
    pm_to_wav(pm, filename, rate=fs)


def pitch_histogram_string_to_control(control: str) -> Control:
    # pitch_histogram, note_density = control.split(';')
    pitch_histogram = control
    note_density = 4  

    pitch_histogram = list(filter(len, pitch_histogram.split(',')))
    if len(pitch_histogram) == 0:
        pitch_histogram = np.ones(12) / 12
    else:
        pitch_histogram = np.array(list(map(float, pitch_histogram)))
        assert pitch_histogram.size == 12
        assert np.all(pitch_histogram >= 0)
        pitch_histogram = pitch_histogram / pitch_histogram.sum() \
            if pitch_histogram.sum() else np.ones(12) / 12
    note_density = int(note_density)
    assert note_density in range(len(ControlSeq.note_density_bins))
    control = Control(pitch_histogram, note_density)
    return control


def pitch_histogram_string_to_control_tensor(control_str: str) -> Tensor:
    """

    :param control_str: e.g. '2,0,1,1,0,1,0,1,1,0,0,1'
    :return: torch.Size([12])
    """
    control = pitch_histogram_string_to_control(control_str)
    controls = torch.from_numpy(control.to_pitch_histogram_array())
    return controls


def cut_pm(pm: PrettyMIDI, duration: float) -> PrettyMIDI:
    new_pm = copy.deepcopy(pm)
    new_pm.adjust_times([0, duration], [0, duration])
    return new_pm


def concat_pms(pms: List[PrettyMIDI]) -> PrettyMIDI:
    res_pm = copy.deepcopy(pms[0])
    print('before: ', len(res_pm.instruments[0].notes))
    for i, pm in enumerate(pms[1:]):
        cur_pm = copy.deepcopy(pm)
        cur_last_time = cur_pm.get_end_time()
        res_last_time = res_pm.get_end_time()
        cur_pm.adjust_times([0., cur_last_time], [res_last_time, res_last_time + cur_last_time])
        res_pm.instruments[0].notes.extend(cur_pm.instruments[0].notes)
        print('after: ', len(res_pm.instruments[0].notes))

    return res_pm
