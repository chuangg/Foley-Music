# Foley Music: Learning to Generate Music from Videos

This repo holds the code for the framework presented on ECCV 2020.

**Foley Music: Learning to Generate Music from Videos**
Chuang Gan, Deng Huang, Peihao Chen, Joshua B. Tenenbaum, and Antonio Torralba

[paper](https://arxiv.org/abs/2007.10984)

## Usage Guide

### Prerequisites

The training and testing in PGCN is reimplemented in PyTorch for the ease of use.
- Pytorch 1.4

Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

### Data Preparation

#### Download Datasets

The extracted pose and midi for training and audio generation can be [downloaded here](http://data.csail.mit.edu/clevrer/data_pose_midi.tar) and unzip to ``./data`` folder.

The original datasets (including videos) can be found: 
- URMP: can be downloaded [here](http://www2.ece.rochester.edu/projects/air/projects/URMP.html)
- MUSIC: can be downloaded [here](https://github.com/roudimit/MUSIC_dataset)
- AtinPiano: proposed by [At Your Fingertips: Automatic Piano Fingering Detection](https://openreview.net/forum?id=H1MOqeHYvB). The dataset can be downloaded [here](https://drive.google.com/file/d/1kDPZSA7ppOaup9Q1Dab7bW4OXNh9mAQA/view)


### Training

For URMP
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/URMP/violin.conf -e exps/urmp-vn
```

For AtinPiano
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/AtinPiano.conf -e exps/atinpiano
```

For MUSIC
```bash
CUDA_VISIBLE_DEVICES=6 python train.py -c config/MUSIC/accordion.conf -e exps/music-accordion
```


### Generating MIDI, sounds and videos

For URMP
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Violin'
python test_URMP.py exps/urmp-vn/checkpoint.pth.tar -o exps/urmp-vn/generate -i Violin -v $VIDEO_PATH -i $INSTRUMENT_NAME
```



For AtinPiano
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Acoustic Grand Piano'
python test_AtinPiano_MUSIC.py exps/atinpiano/checkpoint.pth.tar -o exps/atinpiano/generation -v $VIDEO_PATH -i $INSTRUMENT_NAME
```

For MUSIC
```bash
VIDEO_PATH=/path/to/video
INSTRUMENT_NAME='Accordion'
python test_AtinPiano_MUSIC.py exps/music-accordion/checkpoint.pth.tar -o exps/music-accordion/generation -v $VIDEO_PATH -i $INSTRUMENT_NAME
```

Notes:
- Instrument name ($INSTRUMENT_NAME) can be found [here](https://github.com/craffel/pretty-midi/blob/master/pretty_midi/constants.py#L7)

- If you do not have the video file or you want to generate MIDI and audio only, you can add ``-oa`` flag to skip the generation of video.

## Other Info

### Citation

Please cite the following paper if you feel our work useful to your research.

```
@inproceedings{FoleyMusic2020,
  author    = {Chuang Gan and
               Deng Huang and
               Peihao Chen and
               Joshua B. Tenenbaum and
               Antonio Torralba},
  title     = {Foley Music: Learning to Generate Music from Videos},
  booktitle = {ECCV},
  year      = {2020},
}
```
