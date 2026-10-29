"""Geometric utilities for rotation-equivariant point cloud processing.

This file is partially adapted from geom_torch.py of Canonical Capsules
(https://github.com/canonical-capsules/canonical-capsules) by Weiwei Sun,
Andrea Tagliasacchi, Boyang Deng, Sara Sabour, Soroosh Yazdani, Geoffrey
Hinton and Kwang Moo Yi, licensed under the Apache License, Version 2.0.
See the repository README for the citation.
"""

import torch.nn.functional as F
import torch
import numpy as np
import numpy
import math
from torch_geometric.utils import to_dense_batch
from scipy.spatial.transform import Rotation
from scipy.special import erfc, erfcinv

def sample_on_circle_rotated(normals: torch.Tensor, K: int) -> torch.Tensor:
    """
    Args:
        normals: [B, 1, 3, P] — normal vectors (not necessarily unit)
        K: number of points per normal

    Returns:
        [B, 1, 3, P, K] — K sampled points in plane orthogonal to each normal
    """
    B, _, _, P = normals.shape
    device = normals.device

    # Step 1: Sample K points in XY unit circle (Z=0 plane)
    theta = 2 * torch.pi * torch.rand(B, 1, 1, P, K, device=device)
    r = 1
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros_like(x)
    p = torch.cat([x, y, z], dim=2)  # [B, 1, 3, P, K]

    # Step 2: Normalize input normals
    n = F.normalize(normals, dim=2)  # [B, 1, 3, P]
    z_axis = torch.tensor([0., 0., 1.], device=device).view(1, 1, 3, 1)  # [1, 1, 3, 1]
    z_axis = z_axis.expand(B, 1, 3, P)  # [B, 1, 3, P]

    # Rotation axis and angle
    v = torch.cross(z_axis, n, dim=2)  # [B, 1, 3, P]
    s = torch.norm(v, dim=2, keepdim=True)  # sin(theta), [B, 1, 1, P]
    c = (z_axis * n).sum(dim=2, keepdim=True)  # cos(theta), [B, 1, 1, P]

    # Normalize rotation axis (avoid NaN)
    v = F.normalize(v + 1e-8, dim=2)  # [B, 1, 3, P]
    k = v.unsqueeze(-1)  # [B, 1, 3, P, 1]

    # Expand s and c to match [B, 1, 1, P, K]
    s = s.unsqueeze(-1).expand(B, 1, 1, P, K)
    c = c.unsqueeze(-1).expand(B, 1, 1, P, K)

    # Rodrigues’ formula
    k_cross_p = torch.cross(k.expand(-1, -1, -1, -1, K), p, dim=2)  # [B, 1, 3, P, K]
    k_dot_p = (k * p).sum(dim=2, keepdim=True)  # [B, 1, 1, P, K]

    rotated = (
        p * c +
        k_cross_p * s +
        k * k_dot_p * (1 - c)
    )  # [B, 1, 3, P, K]

    # Handle edge cases (aligned/anti-aligned with z)
    aligned = s < 1e-6  # [B, 1, 1, P, K]
    same_dir = c > 0
    rotated = torch.where(aligned & same_dir, p, rotated)
    rotated = torch.where(aligned & ~same_dir, -p, rotated)

    return rotated  # [B, 1, 3, P, K]

def sample_in_circle_rotated(normals: torch.Tensor, K: int) -> torch.Tensor:
    """
    Args:
        normals: [B, 1, 3, P] — normal vectors (not necessarily unit)
        K: number of points per normal

    Returns:
        [B, 1, 3, P, K] — K sampled points in plane orthogonal to each normal
    """
    B, _, _, P = normals.shape
    device = normals.device

    # Step 1: Sample K points in XY unit circle (Z=0 plane)
    theta = 2 * torch.pi * torch.rand(B, 1, 1, P, K, device=device)
    r = torch.sqrt(torch.rand(B, 1, 1, P, K, device=device))  # uniform sampling in disk
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros_like(x)
    p = torch.cat([x, y, z], dim=2)  # [B, 1, 3, P, K]

    # Step 2: Normalize input normals
    n = F.normalize(normals, dim=2)  # [B, 1, 3, P]
    z_axis = torch.tensor([0., 0., 1.], device=device).view(1, 1, 3, 1)  # [1, 1, 3, 1]
    z_axis = z_axis.expand(B, 1, 3, P)  # [B, 1, 3, P]

    # Rotation axis and angle
    v = torch.cross(z_axis, n, dim=2)  # [B, 1, 3, P]
    s = torch.norm(v, dim=2, keepdim=True)  # sin(theta), [B, 1, 1, P]
    c = (z_axis * n).sum(dim=2, keepdim=True)  # cos(theta), [B, 1, 1, P]

    # Normalize rotation axis (avoid NaN)
    v = F.normalize(v + 1e-8, dim=2)  # [B, 1, 3, P]
    k = v.unsqueeze(-1)  # [B, 1, 3, P, 1]

    # Expand s and c to match [B, 1, 1, P, K]
    s = s.unsqueeze(-1).expand(B, 1, 1, P, K)
    c = c.unsqueeze(-1).expand(B, 1, 1, P, K)

    # Rodrigues’ formula
    k_cross_p = torch.cross(k.expand(-1, -1, -1, -1, K), p, dim=2)  # [B, 1, 3, P, K]
    k_dot_p = (k * p).sum(dim=2, keepdim=True)  # [B, 1, 1, P, K]

    rotated = (
        p * c +
        k_cross_p * s +
        k * k_dot_p * (1 - c)
    )  # [B, 1, 3, P, K]

    # Handle edge cases (aligned/anti-aligned with z)
    aligned = s < 1e-6  # [B, 1, 1, P, K]
    same_dir = c > 0
    rotated = torch.where(aligned & same_dir, p, rotated)
    rotated = torch.where(aligned & ~same_dir, -p, rotated)

    return rotated  # [B, 1, 3, P, K]


def batched_knn(query, key, k = 21):
    """
    query: [B, D, N]
    key: [B, D, M]
    returns: (indices, values)
        - indices: [B, N, k]
        - values: [B, N, k, D]
    """
    # Transpose to [B, N, D] and [B, M, D]
    query = query.transpose(1, 2)  # [B, N, D]
    key = key.transpose(1, 2)      # [B, M, D]

    # Compute pairwise squared distances: [B, N, M]
    query_norm = (query ** 2).sum(dim=2, keepdim=True)       # [B, N, 1]
    key_norm = (key ** 2).sum(dim=2).unsqueeze(1)            # [B, 1, M]
    inner_prod = torch.bmm(query, key.transpose(1, 2))       # [B, N, M]

    dists = query_norm + key_norm - 2 * inner_prod           # [B, N, M]

    # Get k smallest distances
    knn_dists, knn_indices = dists.topk(k, dim=2, largest=False, sorted=True)  # [B, N, k]

    # Gather actual neighbor vectors from key
    # knn_indices: [B, N, k], key: [B, M, D] → values: [B, N, k, D]
    B, N, k = knn_indices.shape
    _, M, D = key.shape

    # Expand knn_indices to gather over last dim
    knn_indices_exp = knn_indices.unsqueeze(-1).expand(B, N, k, D)  # [B, N, k, D]
    key_exp = key.unsqueeze(1).expand(B, N, M, D)                   # [B, N, M, D]
    knn_values = torch.gather(key_exp, dim=2, index=knn_indices_exp)  # [B, N, k, D]

    return knn_indices, knn_values, knn_dists

def construct_transform_matrix(rotation, translation):
    mat = torch.cat([rotation, translation], dim = 2)
    return mat

def invert_pose(mat):
    # mat: B x 3 x 4
    # Affine transform with:
    # rotation: B x 3 x 3
    # translation: B x 3 x 1
    # Invert a rotation and translation
    rotation = mat[:, :, :3]
    rotation_inv = torch.transpose(rotation, 1, 2)
    translation = mat[:, :, 3:]
    translation_inv = - torch.matmul(rotation_inv, translation)
    # Rebuild matrix
    mat_inv = torch.cat((rotation_inv, translation_inv), 2)
    return mat_inv

def transform_graph(graph, mat):
    # graph: PyG DataBatch of graphs
    # mat: B x 3 x 4
    graph = graph.clone()
    position_dense, position_mask = to_dense_batch(graph.position, graph.batch)
    position_dense = position_dense.transpose(-1, -2)
    ones = torch.ones([position_dense.shape[0], 1, position_dense.shape[2]]).to(position_dense.device)
    position_dense = torch.cat([position_dense, ones], dim=1) # B*4*N
    position_dense = torch.matmul(mat, position_dense)
    position_t = position_dense.transpose(-1, -2)[position_mask]
    graph.position = position_t
    normal_dense, normal_mask = to_dense_batch(graph.normal, graph.batch)
    normal_dense = normal_dense.transpose(-1, -2)
    normal_dense = torch.cat([normal_dense, ones], dim=1)
    normal_dense = torch.matmul(mat, normal_dense)
    normal_t = normal_dense.transpose(-1, -2)[normal_mask]
    graph.normal = normal_t
    if hasattr(graph, 'curvature_embedding'):
        curvature_embedding_dense, curvature_embedding_mask = to_dense_batch(graph.curvature_embedding, graph.batch)
        curvature_embedding_dense = curvature_embedding_dense.transpose(-1, -2)
        curvature_embedding_dense = torch.cat([curvature_embedding_dense, ones], dim=1)
        curvature_embedding_dense = torch.matmul(mat, curvature_embedding_dense)
        curvature_embedding_t = curvature_embedding_dense.transpose(-1, -2)[curvature_embedding_mask]
        graph.curvature_embedding = curvature_embedding_t
    return graph
    
def rotate_graph_random(graph, random_range="uni-180-0.2"):
    Rs, ts = random_pose(N=graph.ptr.size(0) - 1, range=random_range)
    Rs = Rs.astype(np.float32)
    Rs = torch.from_numpy(Rs).to(graph.position.device)
    # Set translation to zero vectors
    ts = torch.from_numpy(ts)
    ts = torch.zeros(ts.size()).to(graph.position.device)
    transforms = construct_transform_matrix(Rs, ts)
    graph_rotated = transform_graph(graph, transforms)
    return graph_rotated, transforms

def transform_pc(xyz, mat):
    # Transform a point cloud by applying rotation and translation contained in a 3 x 4 matrix to points
    # xyz: B x 3 x N
    # mat: B x 3 x 4
    return transform_xyz(xyz, mat)

def transform_pc_normals(xyz_normals, mat):
    # Transform a point cloud with normal vectors by applying rotation and translation to points and normals separately
    # xyz_normals: B x 2 x 3 x N
    # mat: B x 3 x 4
    # rotation: B x 3 x 3
    # translation: B x 3 x 1
    points = xyz_normals[:, 0, :, :]
    normals = xyz_normals[:, 1, :, :]
    points_transformed = transform_xyz(points, mat)
    rotation = mat[:, :3, :3]
    normals_transformed = torch.matmul(rotation, normals)
    xyz_normals_transformed = torch.stack((points_transformed, normals_transformed), 1)
    return xyz_normals_transformed

def transform_pc_normals_random(x, random_range="uni-180-0.2"):
    Rs, ts = random_pose(N=x.size(0), range=random_range)
    Rs, ts = Rs.astype(np.float32), ts.astype(np.float32)
    Rs = torch.from_numpy(Rs).to(x.device)
    ts = torch.from_numpy(ts).to(x.device)
    transforms = construct_transform_matrix(Rs, ts)
    x_transformed = transform_pc_normals(x, transforms)
    return x_transformed, transforms

def rotate_pc_normals_random(x, random_range="uni-180-0.2"):
    Rs, ts = random_pose(N=x.size(0), range=random_range)
    Rs = Rs.astype(np.float32)
    Rs = torch.from_numpy(Rs).to(x.device)
    # Set translation to zero vectors
    ts = torch.from_numpy(ts)
    ts = torch.zeros(ts.size()).to(x.device)
    transforms = construct_transform_matrix(Rs, ts)
    x_rotated = transform_pc_normals(x, transforms)
    return x_rotated, transforms

def transform_xyz(xyz, mat):
    # xyz: B3N, mat: B34
    ones = torch.ones([xyz.shape[0], 1, xyz.shape[2]]).to(xyz.device)
    xyz = torch.cat([xyz, ones], dim=1) # B*4*N
    xyz_t = torch.matmul(mat, xyz) # B3N: Rx + t
    return xyz_t

def procruste_pose(pts0, pts1, conf=None, std_noise=1e-8):
    # pts0(Bx3xN), pts1(Bx3xN)
    indim = pts0.shape[1]
    if conf is None:
        conf = 1 / pts0.shape[-1]
    else:
        conf = (conf / conf.sum(1, keepdim=True)).unsqueeze(1) # (B1N)
    if std_noise > 0:
        pts0 = pts0 + torch.normal(0, std_noise, size=pts0.shape).to(pts0.device)
        pts1 = pts1 + torch.normal(0, std_noise, size=pts0.shape).to(pts1.device)
    center_pts0 = (pts0 * conf).sum(dim=2, keepdim=True)
    center_pts1 = (pts1 * conf).sum(dim=2, keepdim=True)

    pts0_centered = pts0 - center_pts0
    pts1_centered = pts1 - center_pts1

    cov = torch.matmul(pts0_centered * conf, pts1_centered.transpose(2, 1))
    
    # Faster but having memory issue. 
    # U, _, V = torch.svd(cov.cpu())
    # U = U.cuda()
    # V = V.cuda()
    # d = torch.eye(indim).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    # d[:, -1, -1] = torch.det(torch.matmul(V, U.transpose(2, 1))) # scalar
    # Vd = torch.matmul(V, d)
    # Rs = torch.matmul(Vd, U.transpose(2, 1))

    # Slower. 
    Rs = []
    for i in range(pts0.shape[0]):
        U, S, V = torch.svd(cov[i])
        d = torch.det(torch.matmul(V, U.transpose(1, 0))) # scalar
        Vd = torch.cat([V[:, :-1], V[:, -1:] * d], dim=-1)
        R = torch.matmul(Vd, U.transpose(1, 0))
        Rs += [R]
    Rs = torch.stack(Rs, dim=0)

    ts = center_pts1 - torch.matmul(Rs, center_pts0) # R * pts0 + t = pts1
    return Rs, ts


# Credit to: Daniel Rebain 
def randn_tail(shape, limits):
    comp_widths = erfc(limits / np.pi)
    widths = 1.0 - comp_widths
    inv_x = comp_widths * np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    x = erfcinv(inv_x) * np.pi
    return x

# Credit to: Daniel Rebain 
def random_rot(shape, limit=180):
    """
    Modification of the Gaussian method to limit the angle directly.
    by Daniel Rebain
    shape: (N, )
    limit: max_angle
    rot: (N, 3, 3)
    """
    limit = limit / 180.0 * np.pi

    vp = np.random.randn(*shape, 3)
    d2 = np.sum(vp**2, axis=-1)
    c2theta = np.cos(0.5 * limit)**2
    wp_limit = np.sqrt(c2theta * d2 / (1.0 - c2theta))

    comp_widths = erfc(wp_limit / np.pi)
    # widths = 1.0 - comp_widths
    inv_x = comp_widths * np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    wp = erfcinv(inv_x) * np.pi

    wp *= 2.0 * np.random.randint(2, size=shape) - 1.0
    q = np.concatenate([vp, wp[:, None]], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    rot = Rotation.from_quat(q).as_matrix()
    return rot

def random_t(shape, limit=0):
    t = np.random.uniform(-1, 1, (*shape, 3, 1)) * limit 
    return t

def random_pose(N, range="uni-180-0"):
    limit_rot, limit_t = range.split("-")[1:]

    rots = random_rot((N, ), limit=float(limit_rot))
    ts = random_t((N, ), limit=float(limit_t))
    return rots, ts

def trans_pc_random(x, random_range="uni-180-0.2", return_pose="rt"):
    # x(B3N)
    indim = x.shape[1]
    if indim == 2:
        return trans_pc_random_euler(x, return_pose)
    Rs, ts = random_pose(N=len(x), range=random_range)
    Rs, ts = Rs.astype(np.float32), ts.astype(np.float32)

    Rs = torch.from_numpy(Rs).to(x.device)
    ts = torch.from_numpy(ts).to(x.device)
    x_rot = torch.matmul(Rs, x) + ts 
    # wrap qt_gt
    ts_gt = - torch.matmul(Rs.transpose(2, 1), ts) # -R't 
    Rs_gt = Rs.transpose(2, 1)
    return [Rs_gt, ts_gt], x_rot
    
def trans_pc_random_euler(x, return_pose="rt"):
    # x(B3N)
    indim = x.shape[1]
    Rs = []
    ts = []
    v_ang_x, v_ang_y, v_ang_z = 2, 2, 2
    v_t = 0.2

    for i in range(len(x)):
        anglex = np.random.uniform() * np.pi * v_ang_x
        angley = np.random.uniform() * np.pi * v_ang_y
        anglez = np.random.uniform() * np.pi * v_ang_z

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        if indim == 3:
            Rx = np.array([[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]])
            R = Rx.dot(Ry).dot(Rz)
            Rs += [R]
            ts += [v_t * np.random.uniform(-1, 1, 3)[:, None]]
        elif indim == 2:
            # use one angle
            R = np.array([[cosx, -sinx],
                           [sinx, cosx]])
            Rs += [R]
            ts += [v_t * np.random.uniform(-1, 1, 2)[:, None]]

    Rs = np.stack(Rs).astype(np.float32)
    ts = np.stack(ts).astype(np.float32)
    Rs = torch.from_numpy(Rs).to(x.device)
    ts = torch.from_numpy(ts).to(x.device)
    x_rot = torch.matmul(Rs, x) + ts 
    # wrap qt_gt
    ts_gt = - torch.matmul(Rs.transpose(2, 1), ts) # -R't 
    Rs_gt = Rs.transpose(2, 1)
    return [Rs_gt, ts_gt], x_rot


def get_rots(indim, bin_size=0.05):
    degsx = np.arange(0, 1, bin_size) * np.pi * 2 
    degsy = np.arange(0, 1, bin_size) * np.pi * 2 
    degsz = np.arange(0, 1, bin_size) * np.pi * 2 
    Rs = []

    for anglex, angley, anglez in zip(degsx, degsy, degsz):
        if indim == 2:
            cosx = np.cos(anglex)
            sinx = np.sin(anglex)
            R = np.array([[cosx, -sinx],
                           [sinx, cosx]]).astype(np.float32)
        else:
            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)
            Rx = np.array([[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]])
            R = Rx.dot(Ry).dot(Rz).astype(np.float32)

        Rs += [R[None]]
    return Rs
