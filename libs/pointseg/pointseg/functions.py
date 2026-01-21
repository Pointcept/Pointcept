import torch

from pointseg._C import segment_mesh_fn, segment_point_fn


def segment_mesh(vertices, faces, kThresh=0.01, segMinVerts=20):
    """segment a mesh (CPU)

    Args:
        vertices (torch.Tensor): vertices of shape==(nv, 3)
        faces (torch.Tensor): faces of shape==(nf, 3)
        kThresh (float): segmentation cluster threshold parameter (larger values lead to larger segments)
        segMinVerts (int): the minimum number of vertices per-segment, enforced by merging small clusters into larger segments
    Returns:
        index (torch.Tensor): the cluster index (starts from 0)
    """
    index = segment_mesh_fn(vertices, faces, kThresh, segMinVerts)
    index = torch.unique(index, return_inverse=True)[1]
    return index


def segment_point(vertices, normals, edges, kThresh=0.01, segMinVerts=20):
    """segment a point cloud (CPU)

    Args:
        vertices (torch.Tensor): vertices of shape==(nv, 3)
        normals (torch.Tensor): normals of shape==(nf, 3)
        edges (torch.Tensor): edges of shape==(ne, 2)
        kThresh (float): segmentation cluster threshold parameter (larger values lead to larger segments)
        segMinVerts (int): the minimum number of vertices per-segment, enforced by merging small clusters into larger segments
    Returns:
        index (torch.Tensor): the cluster index (starts from 0)
    """
    index = segment_point_fn(vertices, normals, edges, kThresh, segMinVerts)
    index = torch.unique(index, return_inverse=True)[1]
    return index
