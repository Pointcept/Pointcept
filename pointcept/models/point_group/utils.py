import torch
from torch.autograd import Function
import pointgroup_ops


class BallQueryBatchP(Function):
    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        """
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) int
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        """

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.zeros(n * meanActive, dtype=torch.int, device=coords.device)
            start_len = torch.zeros((n, 2), dtype=torch.int, device=coords.device)
            nActive = pointgroup_ops.ballquery_batch_p(
                coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius
            )
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class Clustering:
    def __init__(
        self,
        ignored_labels,
        class_mapping,
        thresh=0.03,
        closed_points=300,
        min_points=50,
        propose_points=100,
        score_func=torch.max,
    ) -> None:
        self.ignored_labels = ignored_labels
        self.thresh = thresh
        self.closed_points = closed_points
        self.min_points = min_points
        self.class_mapping = class_mapping
        self.propose_points = propose_points
        self.score_func = score_func

    def cluster(self, vertices, scores):
        labels = torch.max(scores, 1)[1]  # (N) long, cuda
        proposals_idx, proposals_offset = self.cluster_(vertices, labels)

        ## debug
        # import ipdb; ipdb.set_trace()
        # colors = np.array(create_color_palette())[labels.cpu()]
        # write_triangle_mesh(vertices, colors, None, 'semantics.ply')

        # scatter
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, vertices.shape[0]), dtype=torch.int
        )  # (nProposal, N), int, cuda
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        labels = labels[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]

        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = proposals_pointnum > self.propose_points

        proposals_pred = proposals_pred[npoint_mask]
        labels = labels[npoint_mask]
        return proposals_pred, labels

    def cluster_(self, vertices, labels):
        """
        :param batch_idxs: (N), int, cuda
        :labels: 0-19
        """
        batch_idxs = torch.zeros_like(labels)

        mask_non_ignored = torch.ones_like(labels).bool()
        for ignored_label in self.ignored_labels:
            mask_non_ignored = mask_non_ignored & (
                self.class_mapping[labels] != ignored_label
            )
        object_idxs = mask_non_ignored.nonzero().view(-1)

        vertices_ = vertices[object_idxs].float()
        labels_ = labels[object_idxs].int()

        if vertices_.numel() == 0:
            return torch.zeros((0, 2)).int(), torch.zeros(1).int()

        batch_idxs_ = batch_idxs[object_idxs].int()
        batch_offsets_ = torch.FloatTensor([0, object_idxs.shape[0]]).int().cuda()

        idx, start_len = ballquery_batch_p(
            vertices_, batch_idxs_, batch_offsets_, self.thresh, self.closed_points
        )
        proposals_idx, proposals_offset = bfs_cluster(
            labels_.cpu(), idx.cpu(), start_len.cpu(), self.min_points
        )
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

        return proposals_idx, proposals_offset

    def get_instances(self, vertices, scores):
        proposals_pred, labels = self.cluster(vertices, scores)
        instances = {}
        for proposal_id in range(len(proposals_pred)):
            clusters_i = proposals_pred[proposal_id]
            score = scores[clusters_i.bool(), labels[proposal_id]]
            score = self.score_func(score)
            instances[proposal_id] = {}
            instances[proposal_id]["conf"] = score.cpu().numpy()
            instances[proposal_id]["label_id"] = self.class_mapping.cpu()[
                labels[proposal_id]
            ]
            instances[proposal_id]["pred_mask"] = clusters_i.cpu().numpy()
        return instances


class BFSCluster(Function):
    @staticmethod
    def forward(ctx, semantic_label, ball_query_idxs, start_len, threshold):
        """
        :param ctx:
        :param semantic_label: (N), int
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        """

        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = semantic_label.new()
        cluster_offsets = semantic_label.new()

        pointgroup_ops.bfs_cluster(
            semantic_label,
            ball_query_idxs,
            start_len,
            cluster_idxs,
            cluster_offsets,
            N,
            threshold,
        )

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


bfs_cluster = BFSCluster.apply
