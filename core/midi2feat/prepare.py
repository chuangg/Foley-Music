'''
PITCH_LOW, PITCH_HIGH: the range of detection with `pitch`(default=34,94)
VELOCITY_LOW, VELOCITY_HIGH, VELOCITY_STEP: the range of detection with `velocity` or `volume`(deafault=24,127,1)
CONTROL_BINS: never mind, may control instrument change in the feature(if needed)(default=2)
_fs : frames per second(higher, better, default=100)

Label size: len(_inv_dic) or compute = pitch_range * velocity_range(step) * control_bins
'''
from tqdm import tqdm
import os, sys
import os.path as P

# FIX: Lake of necessary info like control change, pitch-bend or so
import pretty_midi as pmidi
import numpy as np
import warnings
# Defin Low, mid, High by histogram

PITCH_LOW = 34
PITCH_HIGH = 94
VELOCITY_LOW = 34
VELOCITY_HIGH = 127
VELOCITY_STEP = 1
CONTROL_BINS = 2

feat_dim = 3
_fs = 100 # # fs = 100 -> resolution is 0.01s

def get_midi_path(root, ext_names):
    midis_path = {}
    for dirpath, dirs, files in os.walk(root):
        if not files:
            continue
        for f in files:
            base, ext = P.splitext(P.basename(f))
            if not ext in ext_names:
                continue
            this_path = P.join(dirpath, f)
            rel_path = P.relpath(dirpath, root)
            midis_path[base] = [this_path, rel_path]
    return midis_path

histogram_pitch = np.zeros(128, dtype=int)
histogram_velocity = np.zeros(128, dtype=int)
def plot_histogram(midis):
    def count(midi: pmidi.PrettyMIDI):  # count pitch and velocity
        global histogram_pitch, histogram_velocity
        assert len(midi.instruments) == 1, "more than 1 instr"
        for ins in midi.instruments:
            assert ins.is_drum == False, "Drum"
            assert len(ins.pitch_bends) == 0, "has pitch_bends"
            histogram, _ = np.histogram(
                [n.pitch for n in ins.notes],
                bins=np.arange(129),  # 128+1, 1~128(edge)
                weights=None,
                density=False)
            histogram_pitch += histogram
            histogram, _ = np.histogram([n.velocity for n in ins.notes],
                                        bins=np.arange(129),
                                        weights=None,
                                        density=False)
            histogram_velocity += histogram
            for n in ins.notes:
                if n.velocity < 13:
                    print(1)


    for midi in tqdm(midis.values()):
        count(midi)

    print(histogram_pitch)
    print(histogram_velocity)

    from matplotlib import pyplot as plt
    plt.bar(np.arange(128), histogram_pitch)
    plt.xticks(np.arange(128),fontsize=2)
    plt.xlim(0,128)
    plt.ylabel('histogram_pitch')
    plt.savefig('histogram_pitch.png',dpi=960)
    plt.close()

    plt.bar(np.arange(128), histogram_velocity)
    plt.xticks(np.arange(128),fontsize=1)
    plt.xlim(0,128)
    plt.ylabel('histogram_velocity')
    plt.savefig('histogram_velocity.png',dpi=960)


def _range(id, low=27, high=127, step=1, method=1):  # start from 1
    if method == 0:
        if id < low:
            id = low
        if id > high:
            id = high
        return (id - low) // step + 1 # start from 1
    elif method == 1:
        if id < low:
            return 1
        elif id > high:
            return 2
        else:
            return (id - low) // step + 3 # start from 3


def _inverse_range(id, low=27, high=127, step=1, method=1):  # start from 1
    if method == 0:
        return (id -1) * step + low
    elif method == 1:
        if id == 1:
            return low  # or a random sample?
        if id == 2:
            return high  # or sample?
        else:
            return (id - 3) * step + low  # or sample from step


def create_dict(inverse_range=True, clean=False):
    pitchs = set()
    velocity = set()
    control = list(i + 1 for i in range(CONTROL_BINS)) # 1,2
    for i in range(128):
        pitchs.add(_range(i, PITCH_LOW, PITCH_HIGH))
        velocity.add(_range(i, VELOCITY_LOW, VELOCITY_HIGH, VELOCITY_STEP))
    pitchs = list(pitchs)
    pitchs.sort()
    velocity = list(velocity)
    velocity.sort()

    # 1-base
    dic = np.zeros((max(pitchs) + 1, max(velocity) + 1, max(control) + 1)).astype(np.int64)
    inv_dic = {0: np.array([0, 0, 0])}
    id = 1
    for i in pitchs:
        for j in velocity:
            for k in control:
                dic[i][j][k] = id
                inv_dic[id] = np.array([i, j, k])
                id += 1

    return dic, inv_dic

_dic, _inv_dic = create_dict() # default

def find_continue(arr):
    # 1, 2, 3, 10, 11, 12, 13 -> return tuple((1,3), (10, 13))
    cont = []

    curid = 0
    while curid < len(arr):
        start = curid
        while curid < len(arr) - 1 and arr[curid] + 1 == arr[curid + 1]:
            curid += 1
        cont.append([start, curid])
        curid += 1
    return cont

def get_feat(midi: pmidi.PrettyMIDI, fs=_fs, dic=_dic):
    global feat_dim
    end = int(fs * midi.get_end_time())
    rolls = np.zeros((len(midi.instruments), end)).astype(np.int64)

    for idx, ins in enumerate(midi.instruments):
        # Allocate a matrix of zeros - we will add in as we go
        end = int(fs * ins.get_end_time())
        roll = np.zeros((feat_dim, end)).astype(np.int64)

        for note in ins.notes:
            method = 1
            start, end = int(note.start * fs), int(note.end * fs)
            pitch = _range(note.pitch, PITCH_LOW, PITCH_HIGH)
            velocity = _range(note.velocity, VELOCITY_LOW, VELOCITY_HIGH, VELOCITY_STEP)
            if pitch == 0 or velocity == 0:
                raise Exception

            # if roll[2,end-1] != 0 or roll[2, start] != 0:
            if np.any(roll[2,start:end] != 0): # overwrite method
                print('Warn: Overwrite', start, end)
                # print(np.where(roll[2,start:end] == 2)[0])
                method = 2

            if method == 1:
                roll[0, start:end] = pitch
                roll[1, start:end] = velocity
                roll[2, start:end] = 1
                roll[2, start] = 2 # means note on(2), time_shift(1) -> direct overwrite
            elif method == 2:
                modify_place = np.where(roll[2,start:end] == 0)[0] + start
                cuts = find_continue(modify_place)
                for start, end in cuts:
                    # print(start, end, end='\t')
                    start = modify_place[start]
                    end = modify_place[end] + 1 # from end-index to end-slice
                    # print(start, end, end='\n')
                    roll[0, start:end] = pitch
                    roll[1, start:end] = velocity
                    roll[2, start:end] = 1
                    roll[2, start] = 2 # means note on(2), time_shift(1) -> direct overwrite


        # Encode to one-hot
        for i in range(roll.shape[1]):
            p, v, c = roll[:, i]
            rolls[idx, i] = dic[p][v][c]

    return rolls

# midis_feats = []
# for _, midi in tqdm(enumerate(midis)):
#     midis_feats.append(get_feat(midi))


def inverse(rolls, ori_midi, fs=_fs, inv_dic=_inv_dic):
    if isinstance(ori_midi, pmidi.PrettyMIDI):
        def get_prog(ins_idx):
            return ori_midi.instruments[ins_idx].program
    elif isinstance(ori_midi, int):
        def get_prog(ins_idx):
            return ori_midi

    new_midi = pmidi.PrettyMIDI()
    for ins_idx, roll in enumerate(rolls):  # instrument idx
        new_instr = pmidi.Instrument(program=get_prog(ins_idx))
        instr_end_time = roll.shape[0] # 1-dim(one-hot)
        
        # recover one-hot -> feat_dim
        feat = np.zeros((feat_dim, roll.shape[0])).astype(np.int64)
        for idx, value in enumerate(roll):
            feat[:, idx] = inv_dic[int(value)]

        note_on = np.where(feat[2,:] == 2)[0]
        for time in note_on:
            end_time = time + 1
            while end_time < instr_end_time and feat[2,end_time] == 1:
                end_time += 1
            end_time -= 1 # back
            new_note = pmidi.Note(
                velocity = _inverse_range(feat[1, time], VELOCITY_LOW, VELOCITY_HIGH, VELOCITY_STEP),
                pitch =  _inverse_range(feat[0, time], PITCH_LOW, PITCH_HIGH),
                start=time / fs,
                end=end_time / fs)
            new_instr.notes.append(new_note)

        new_midi.instruments.append(new_instr)
    return new_midi

# new_midis = []
# for midi_idx, midi_feats in tqdm(enumerate(midis_feats)):
#     new_midis.append(inverse(midi_feats, midis[midi_idx]))
# new_midis[0].write('../test_out.mid')

def write_feats(midis_path, midis_feats, out_path): # remain path
    for id, [this_path, rel_path] in midis_path.items():
        rel_path =  P.join(out_path, rel_path)
        p = P.join(rel_path, id)
        print(p)
        if not P.exists(rel_path):
            os.makedirs(rel_path)

        np.save(p, midis_feats[id])