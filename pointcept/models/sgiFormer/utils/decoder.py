"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import torch
import torch.nn as nn
from functools import partial
from pointcept.models.builder import MODELS
from .data import split_offset
from torch_scatter import scatter_mean
from .position_embedding import PositionEmbeddingCoordsSine


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_masks=None, pe=None, query_pe=None):
        """
        query Tensor (b, n_q, d_model)
        """
        outputs = []
        for i in range(len(source)):
            q_pos = query_pe[i] if query_pe is not None else None
            pos = pe[i] if pe is not None else None
            q = self.with_pos_embed(query[i], q_pos)
            k = self.with_pos_embed(source[i], pos)
            v = source[i]
            attn_mask = attn_masks[i] if attn_masks else None
            output, _ = self.attn(q, k, v, attn_mask=attn_mask)
            output = self.dropout(output) + query[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        outputs = []
        for i in range(len(x)):
            pos = pe[i] if pe is not None else None
            q = k = self.with_pos_embed(x[i], pos)
            output, _ = self.attn(q, k, x[i])
            output = self.dropout(output) + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            output = self.net(x[i])
            output = output + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


@MODELS.register_module("SPFormerDecoder")
class SPFormerDecoder(nn.Module):
    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        iter_pred=False,
        attn_mask=False,
        use_query_pos=False,
        use_score=False,
        use_param_query=False,
    ):
        super().__init__()
        self.use_score = use_score
        self.num_layer = num_layer
        self.num_query = num_query
        self.num_class = num_class
        self.use_param_query = use_param_query
        self.use_query_pos = use_query_pos
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )

        if use_param_query:
            # learnable query
            self.query = nn.Embedding(num_query, d_model)
        else:
            raise ValueError("No query method is selected.")

        if use_query_pos:
            self.pe = nn.Embedding(num_query, d_model)
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1)
        )
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )
        self.x_mask = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

    def _forward_head(self, query, mask_feats):
        pred_labels = []
        pred_masks, attn_masks = [], []
        pred_scores = [] if self.use_score else None
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            if self.use_score:
                pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum("nd, md->nm", norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_labels, pred_scores, pred_masks, attn_masks

    def _get_query(self, input_dict):
        sp_feat = input_dict["sp_feat"]
        B = len(sp_feat)
        pe = (
            self.pe.weight.unsqueeze(0).repeat(B, 1, 1)
            if getattr(self, "pe", None)
            else None
        )

        assert self.use_param_query
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)

        return query, pe

    def forward_iter_pred(self, input_dict):
        sp_feat = input_dict["sp_feat"]

        pred_labels, pred_masks, pred_scores = [], [], []

        inst_feats = [self.input_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feat]

        query, pe = self._get_query(input_dict)

        pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
            query, mask_feats
        )
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, attn_mask, query_pe=pe)
            query = self.self_attn_layers[i](query, pe)
            query = self.ffn_layers[i](query)

            pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
                query, mask_feats
            )
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
        return {
            "labels": pred_label,
            "masks": pred_mask,
            "scores": pred_score,
            "aux_outputs": [
                {"labels": a, "masks": b, "scores": c}
                for a, b, c in zip(
                    pred_labels[:-1],
                    pred_masks[:-1],
                    pred_scores[:-1],
                )
            ],
        }

    def forward(self, input_dict):
        if self.iter_pred:
            return self.forward_iter_pred(input_dict)
        return None


@MODELS.register_module("SGIFormerDecoder")
class SGIFormerDecoder(nn.Module):
    def __init__(
        self,
        dec_num_layer=3,
        num_sample_query=200,
        num_learn_query=200,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        attn_mask=True,
        use_score=False,
        alpha=0.4,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_score = use_score
        self.dec_num_layer = dec_num_layer
        self.num_class = num_class
        self.d_model = d_model
        self.attn_mask = attn_mask
        self.alpha = alpha

        # voxel wise cls and bias
        self.vx_seg_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, num_class + 1),
        )
        self.vx_bias_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, 3),
        )

        self.vx_feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.rep_layer = nn.Sequential(
            nn.Linear(d_model, num_sample_query),
            nn.LayerNorm(num_sample_query),
            nn.ReLU(),
        )
        self.query_learn = nn.Embedding(num_learn_query, d_model)

        self.sp_feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.x_mask = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

        self.sp_pos = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=d_model,
            normalize=True,
        )
        self.feat_query_attn_layers = nn.ModuleList([])
        self.feat_self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(self.dec_num_layer):
            self.feat_query_attn_layers.append(
                CrossAttentionLayer(d_model, nhead, dropout)
            )
            self.feat_self_attn_layers.append(
                SelfAttentionLayer(d_model, nhead, dropout)
            )

            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1)
        )
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )

    def _forward_head(self, query, mask_feats):
        pred_labels = []
        pred_masks, attn_masks = [], []
        pred_scores = [] if self.use_score else None
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            if self.use_score:
                pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum("nd, md->nm", norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_labels, pred_scores, pred_masks, attn_masks

    def _get_query(self, input_dict, vx_logits=None, vx_bias=None):
        vx_feat = input_dict["vx_feat"]
        inv, sp = input_dict["inv"], input_dict["sp"]

        vx_logits = split_offset(vx_logits, input_dict["offset"].int())
        vx_score = [x.softmax(dim=-1)[:, :-1] for x in vx_logits]

        vx_bias = split_offset(vx_bias, input_dict["offset"].int())
        shift_coord = [
            _bias + _coord for _bias, _coord in zip(vx_bias, input_dict["vx_coord"])
        ]
        sp_coord = [
            scatter_mean(_scoord[_inv], _sp, dim=0)
            for _scoord, _inv, _sp in zip(shift_coord, inv, sp)
        ]

        vx_feat_proj = [self.vx_feat_proj(x) for x in vx_feat]
        sel_vx_feat = []
        for i, (_score) in enumerate(vx_score):
            max_score, _ = _score.max(dim=-1)
            _, topk_idx = max_score.topk(
                int(self.alpha * _score.shape[0]), sorted=False
            )
            sel_vx_feat.append(vx_feat_proj[i][topk_idx, :])

        rep = [self.rep_layer(x) for x in sel_vx_feat]
        activation = [torch.softmax(x.T, dim=-1) for x in rep]
        query = [act @ x for act, x in zip(activation, sel_vx_feat)]
        query_learn = self.query_learn.weight.unsqueeze(0).repeat(len(query), 1, 1)
        query = [torch.cat((x, y), dim=0) for x, y in zip(query, query_learn)]

        return query, sp_coord, vx_feat_proj

    def forward(self, input_dict):
        sp_feat = input_dict["sp_feat"]
        vx_seg = self.vx_seg_head(input_dict["feats"])
        vx_bias = self.vx_bias_head(input_dict["feats"])

        query, sp_coord, vx_feat_proj = self._get_query(input_dict, vx_seg, vx_bias)
        sp_pos = []
        for _coord in sp_coord:
            p_min, p_max = _coord.min(0)[0], _coord.max(0)[0]
            pos_emb = self.sp_pos(
                _coord.unsqueeze(0),
                num_channels=self.d_model,
                input_range=(p_min.unsqueeze(0), p_max.unsqueeze(0)),
            )[0]
            sp_pos.append(pos_emb)
        sp_feats = [self.sp_feat_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feats]

        pred_labels, pred_masks, pred_scores = [], [], []
        pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
            query, mask_feats
        )
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(self.dec_num_layer):
            source = [x + y for x, y in zip(sp_feats, sp_pos)]
            query = self.cross_attn_layers[i](source, query, attn_mask)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)

            sp_feats = self.feat_query_attn_layers[i](query, sp_feats, query_pe=sp_pos)
            sp_feats = self.feat_self_attn_layers[i](sp_feats, sp_pos)

            pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
                query, mask_feats
            )
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
        return {
            "labels": pred_label,
            "masks": pred_mask,
            "scores": pred_score,
            "aux_outputs": [
                {"labels": a, "masks": b, "scores": c}
                for a, b, c in zip(
                    pred_labels[:-1],
                    pred_masks[:-1],
                    pred_scores[:-1],
                )
            ],
            "vx_seg": vx_seg,
            "vx_bias": vx_bias,
        }


@MODELS.register_module("ScanNetPPSGIFormerDecoder")
class ScanNetPPSGIFormerDecoder(nn.Module):
    def __init__(
        self,
        dec_num_layer=3,
        num_sample_query=200,
        num_learn_query=200,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        attn_mask=True,
        use_score=False,
        alpha=0.4,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_score = use_score
        self.dec_num_layer = dec_num_layer
        self.num_class = num_class
        self.d_model = d_model
        self.attn_mask = attn_mask
        self.alpha = alpha

        # voxel wise cls and bias
        self.vx_seg_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, num_class + 1),
        )
        self.vx_bias_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, 3),
        )

        self.vx_feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.rep_layer = nn.Sequential(
            nn.Linear(d_model, num_sample_query),
            nn.LayerNorm(num_sample_query),
            nn.ReLU(),
        )
        self.query_learn = nn.Embedding(num_learn_query, d_model)

        self.sp_feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.x_mask = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

        self.sp_pos = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=d_model,
            normalize=True,
        )
        self.feat_query_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(self.dec_num_layer):
            self.feat_query_attn_layers.append(
                CrossAttentionLayer(d_model, nhead, dropout)
            )

            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1)
        )
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )

    def _forward_head(self, query, mask_feats):
        pred_labels = []
        pred_masks, attn_masks = [], []
        pred_scores = [] if self.use_score else None
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            if self.use_score:
                pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum("nd, md->nm", norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_labels, pred_scores, pred_masks, attn_masks

    def _get_query(self, input_dict, vx_logits=None, vx_bias=None):
        vx_feat = input_dict["vx_feat"]
        inv, sp = input_dict["inv"], input_dict["sp"]

        vx_logits = split_offset(vx_logits, input_dict["offset"].int())
        vx_score = [x.softmax(dim=-1)[:, :-1] for x in vx_logits]

        vx_bias = split_offset(vx_bias, input_dict["offset"].int())
        shift_coord = [
            _bias + _coord for _bias, _coord in zip(vx_bias, input_dict["vx_coord"])
        ]
        sp_coord = [
            scatter_mean(_scoord[_inv], _sp, dim=0)
            for _scoord, _inv, _sp in zip(shift_coord, inv, sp)
        ]

        vx_feat_proj = [self.vx_feat_proj(x) for x in vx_feat]
        sel_vx_feat = []
        for i, (_score) in enumerate(vx_score):
            max_score, _ = _score.max(dim=-1)
            _, topk_idx = max_score.topk(
                int(self.alpha * _score.shape[0]), sorted=False
            )
            sel_vx_feat.append(vx_feat_proj[i][topk_idx, :])

        rep = [self.rep_layer(x) for x in sel_vx_feat]
        activation = [torch.softmax(x.T, dim=-1) for x in rep]
        query = [act @ x for act, x in zip(activation, sel_vx_feat)]
        query_learn = self.query_learn.weight.unsqueeze(0).repeat(len(query), 1, 1)
        query = [torch.cat((x, y), dim=0) for x, y in zip(query, query_learn)]

        return query, sp_coord, vx_feat_proj

    def forward(self, input_dict):
        sp_feat = input_dict["sp_feat"]
        vx_seg = self.vx_seg_head(input_dict["feats"])
        vx_bias = self.vx_bias_head(input_dict["feats"])

        query, sp_coord, vx_feat_proj = self._get_query(input_dict, vx_seg, vx_bias)
        sp_pos = []
        for _coord in sp_coord:
            p_min, p_max = _coord.min(0)[0], _coord.max(0)[0]
            pos_emb = self.sp_pos(
                _coord.unsqueeze(0),
                num_channels=self.d_model,
                input_range=(p_min.unsqueeze(0), p_max.unsqueeze(0)),
            )[0]
            sp_pos.append(pos_emb)
        sp_feats = [self.sp_feat_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feats]

        pred_labels, pred_masks, pred_scores = [], [], []
        pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
            query, mask_feats
        )
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(self.dec_num_layer):
            source = [x + y for x, y in zip(sp_feats, sp_pos)]
            query = self.cross_attn_layers[i](source, query, attn_mask)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)

            sp_feats = self.feat_query_attn_layers[i](query, sp_feats, query_pe=sp_pos)
            # Since memory is limited, we do not use self attention for ScanNet++
            # sp_feats = self.feat_self_attn_layers[i](sp_feats, sp_pos)

            pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
                query, mask_feats
            )
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
        return {
            "labels": pred_label,
            "masks": pred_mask,
            "scores": pred_score,
            "aux_outputs": [
                {"labels": a, "masks": b, "scores": c}
                for a, b, c in zip(
                    pred_labels[:-1],
                    pred_masks[:-1],
                    pred_scores[:-1],
                )
            ],
            "vx_seg": vx_seg,
            "vx_bias": vx_bias,
        }
