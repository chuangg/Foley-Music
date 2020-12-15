from .seq2seq import seq2seq_baseline, Seq2Seq
from torch import nn, Tensor
from typing import Optional


class PoseSeq2Seq(nn.Module):

    def __init__(self, pose_encoder: nn.Module, seq2seq: Seq2Seq):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.seq2seq = seq2seq

    def forward(self, pose, events, teacher_forcing_ratio=0.5):
        pose_feature: Tensor = self.pose_encoder(pose)  # [B, C, T, V, M] -> [B, C, T]
        pose_feature = pose_feature.permute(2, 0, 1)  # [B, C, T] -> [T, B, C]
        outputs = self.seq2seq(pose_feature, events, teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs


class PoseSeq2SeqTransformer(nn.Module):

    def __init__(self, pose_seq2seq: PoseSeq2Seq):
        super().__init__()
        self.pose_seq2seq = pose_seq2seq

    def forward(
            self,
            pose: Tensor,
            tgt: Tensor,
            use_mask=True,
            pad_idx=242,
            control: Optional[Tensor] = None
    ):
        events = tgt.transpose(0, 1).contiguous()  # [B, T] -> [T, B] RNN输入
        out: Tensor = self.pose_seq2seq(pose, events, teacher_forcing_ratio=1.)  # mt相当于全用
        out = out.transpose(0, 1).contiguous()  # [T, B, C] -> [B, T, C]
        return out


def pose_seq2seq_baseline(
        emb_dim=256,
        hid_dim=512,
        layout='body25',
        num_encoder_layers=1,
        num_decoder_layers=1,
        use_faster=False
):
    # from .st_gcn.st_gcn_dilated import st_gcn_baseline
    from .st_gcn.st_gcn_aaai18 import st_gcn_baseline

    in_channels = 2 if layout == 'hands' else 3

    pose_encoder = st_gcn_baseline(in_channels, emb_dim, layout=layout)
    seq2seq = seq2seq_baseline(
        240 + 3,
        # 240 + 2,
        enc_emb_dim=emb_dim,
        dec_emb_dim=emb_dim,
        enc_hid_dim=hid_dim,
        dec_hid_dim=hid_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_faster=use_faster
    )
    model = PoseSeq2Seq(pose_encoder, seq2seq)
    return model


def pose_seq2seq_midi2feat(emb_dim=256, hid_dim=512, layout='body25'):
    # from .st_gcn.st_gcn_dilated import st_gcn_baseline
    from .st_gcn.st_gcn_aaai18 import st_gcn_baseline
    pose_encoder = st_gcn_baseline(3, emb_dim, layout=layout)
    seq2seq = seq2seq_baseline(
        # 240 + 3,
        10896 + 1,
        enc_emb_dim=emb_dim,
        dec_emb_dim=emb_dim,
        enc_hid_dim=hid_dim,
        dec_hid_dim=hid_dim
    )
    model = PoseSeq2Seq(pose_encoder, seq2seq)
    return model


def pose_seq2seq_transformer(emb_dim=256, hid_dim=512, layout='body25', num_encoder_layers=1, num_decoder_layers=1):
    pose_seq2seq = pose_seq2seq_baseline(
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        layout=layout,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_faster=False,
    )
    model = PoseSeq2SeqTransformer(pose_seq2seq)
    return model
