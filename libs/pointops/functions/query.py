import torch
from torch.autograd import Function

from pointops._C import knn_query_cuda, random_ball_query_cuda, ball_query_cuda


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, offset, new_xyz=None, new_offset=None):
        """
        input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample) -1 is placeholder, dist2: (m, nsample)
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), dtype=torch.int, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float, device=xyz.device)
        knn_query_cuda(
            m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2
        )
        return idx, torch.sqrt(dist2)


class RandomBallQuery(Function):
    """Random Ball Query.

    Find nearby points in spherical space.
    """

    @staticmethod
    def forward(
        ctx, nsample, max_radius, min_radius, xyz, offset, new_xyz=None, new_offset=None
    ):
        """
        input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        assert min_radius < max_radius

        m = new_xyz.shape[0]
        order = []
        for k in range(offset.shape[0]):
            s_k, e_k = (0, offset[0]) if k == 0 else (offset[k - 1], offset[k])
            order.append(
                torch.randperm(e_k - s_k, dtype=torch.int32, device=offset.device) + s_k
            )
        order = torch.cat(order, dim=0)
        idx = torch.zeros((m, nsample), dtype=torch.int, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float, device=xyz.device)
        random_ball_query_cuda(
            m,
            nsample,
            min_radius,
            max_radius,
            order,
            xyz,
            new_xyz,
            offset.int(),
            new_offset.int(),
            idx,
            dist2,
        )
        return idx, torch.sqrt(dist2)


class BallQuery(Function):
    """Ball Query.

    Find nearby points in spherical space.
    """

    @staticmethod
    def forward(
        ctx, nsample, max_radius, min_radius, xyz, offset, new_xyz=None, new_offset=None
    ):
        """
        input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        assert min_radius < max_radius

        m = new_xyz.shape[0]
        idx = torch.zeros((m, nsample), dtype=torch.int, device=xyz.device)
        dist2 = torch.zeros((m, nsample), dtype=torch.float, device=xyz.device)
        ball_query_cuda(
            m,
            nsample,
            min_radius,
            max_radius,
            xyz,
            new_xyz,
            offset.int(),
            new_offset.int(),
            idx,
            dist2,
        )
        return idx, torch.sqrt(dist2)


knn_query = KNNQuery.apply
ball_query = BallQuery.apply
random_ball_query = RandomBallQuery.apply
