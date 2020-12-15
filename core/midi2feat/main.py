'''
Some configs are shown in prepare.py

# Here is the wrapper of: 
- historgam: get plot of `pitch` and `velocity` to get best choice
- get_feats: from midi to feats
- inverse: get back to midi

# Some configs are shown here:
- root: data-root, will detect `mid` file use os.walk()
- out_root: output dir
- ext_names: support midi extensions

# Test code
- change root to '.', getting access to `test.mid`
- test recover with the last part of code
- do online-recovery (a better way to debug, rather than way-2)
'''

from .prepare import *

#%% region Config
root = P.abspath('../midi')
# root = P.abspath('.')

out_root = P.abspath('../data_tmp') 

ext_names = ['.mid', '.midi', '.MID']

midis_path = get_midi_path(root, ext_names)

midis = {k: pmidi.PrettyMIDI(p[0]) for k, p in midis_path.items()}

# Plot pitch, velocity distribution in bar
plot_histogram(midis)

# endregion

#%% region compute feats and recover midi

#NOTE: Main Part

midis_feats = {}
for k, midi in tqdm(midis.items()):
        midis_feats[k] = get_feat(midi)

recover_midis = {}
for k, midi_feats in tqdm(midis_feats.items()):
    # Use midis[midi_idx] old to identify program(instrument)
    recover_midis[k] = inverse(midi_feats, midis[k])
# recover_midis.write('../test-out.mid')

write_feats(midis_path, midis_feats, out_root)

# endregion
#%% Test recover code
# midi_feat = np.load('data/bn/ScoSep_4_bn_28_Fugue.npy')
# midi = inverse(midi_feat, midis['ScoSep_4_bn_28_Fugue'])
# midi.write('test-out.mid')
