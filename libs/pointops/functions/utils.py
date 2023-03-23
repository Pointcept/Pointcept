import torch
from pointops import knn_query, ball_query, grouping


def knn_query_and_group(feat,
                        xyz,
                        offset=None,
                        new_xyz=None,
                        new_offset=None,
                        idx=None,
                        nsample=None,
                        with_xyz=False
                        ):
    if idx is None:
        assert nsample is not None
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx


def ball_query_and_group(feat,
                         xyz,
                         offset=None,
                         new_xyz=None,
                         new_offset=None,
                         idx=None,
                         max_radio=None,
                         min_radio=0,
                         nsample=None,
                         with_xyz=False
                         ):
    if idx is None:
        assert nsample is not None and offset is not None
        assert max_radio is not None and min_radio is not None
        idx, _ = ball_query(nsample, max_radio, min_radio, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx


def query_and_group(nsample,
                    xyz,
                    new_xyz,
                    feat,
                    idx,
                    offset,
                    new_offset,
                    dilation=0,
                    with_feat=True,
                    with_xyz=True,
                    ):
    """
    input: coords: (n, 3), new_xyz: (m, 3), color: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, nsample, c+3), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz

    if idx is None:
        num_samples_total = 1 + (nsample - 1) * (dilation + 1)
        # num points in a batch might < num_samples_total => [n1, n2, ..., nk, ns, ns, ns, ...]
        idx_no_dilation, _ = knn_query(num_samples_total, xyz, offset, new_xyz,
                                       new_offset)  # (m, nsample * (d + 1))
        idx = []
        batch_end = offset.tolist()
        batch_start = [0] + batch_end[:-1]
        new_batch_end = new_offset.tolist()
        new_batch_start = [0] + new_batch_end[:-1]
        for i in range(offset.shape[0]):
            if batch_end[i] - batch_start[i] < num_samples_total:
                soft_dilation = (batch_end[i] - batch_start[i] - 1) / (nsample - 1) - 1
            else:
                soft_dilation = dilation
            idx.append(idx_no_dilation[new_batch_start[i]: new_batch_end[i],
                       [int((soft_dilation + 1) * i) for i in range(nsample)]])
        idx = torch.cat(idx, dim=0)

    if not with_feat:
        return idx

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # (m, nsample, 3)
    # grouped_xyz = grouping(coords, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)  # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, nsample, c)
    # grouped_feat = grouping(color, idx) # (m, nsample, c)

    if with_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1), idx  # (m, nsample, 3+c)
    else:
        return grouped_feat, idx


def offset2batch(offset):
    return torch.cat([
        torch.tensor([i] * (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o)
        for i, o in enumerate(offset)
    ], dim=0).long().to(offset.device)


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).int()