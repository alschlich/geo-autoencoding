import torch
import pytorch3d
from torch_geometric.utils import to_dense_batch
from pytorch3d.loss import chamfer_distance
from geom_utils import transform_graph, invert_pose
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance

def equivariance_loss (keypoint_coords_1, keypoint_coords_2, mat_2):
    # sum(l2norm(keypoint_coords_1 - T_b^(-1) * keypoint_coords_2)) / keypoint_count
    inverse_mat_2 = invert_pose(mat_2)
    transform = transform_graph(keypoint_coords_2, inverse_mat_2)
    sub = keypoint_coords_1.position - transform.position
    norm = torch.linalg.vector_norm(sub, dim = 1)
    sum = torch.sum(norm, 0)
    div = sum / keypoint_coords_1.position.size(0)
    return div

def invariance_loss (encoding_1, encoding_2):
    # l2norm(encoding_1 - encoding_2) / encoding_dim
    sub = encoding_1 - encoding_2
    norm = torch.linalg.vector_norm(sub, dim = 1)
    div = norm / encoding_1.size(1)
    batch_sum = torch.sum(div)
    return batch_sum

def canonicalization_loss (keypoint_coords_predicted, keypoint_coords_regressed, mat_canon):
    # sum(l2norm((rotation * keypoint_coords_predicted + translation) - keypoint_coords_regressed)) / keypoint_count
    transform = transform_pc(keypoint_coords_predicted, mat_canon)
    sub = transform - keypoint_coords_regressed
    norm = torch.linalg.vector_norm(sub, dim = 1)
    sum = torch.sum(norm, dim = 1)
    div = sum / keypoint_coords_predicted.size(2)
    batch_sum = torch.sum(div)
    return batch_sum

def reconstruction_loss (points_reconstructed_canon, points_input, mat_decanon):
    # chamfer_distance(decanonicalizing_rotation * points_reconstructed + decanonicalizing_translation, points_input)
    transform = transform_pc_normals(points_reconstructed_canon, mat_decanon)
    # Permute for pytorch3d batch logic
    permute_reconstructed = transform.permute(0, 2, 1)
    permute_input = points_input.permute(0, 2, 1)
    # Compute chamfer distances and sum
    chamfer_distance_losses = chamfer_distance(permute_reconstructed[:, :, :3], permute_input[:, :, :3], x_normals = permute_reconstructed[:, :, 3:], y_normals = permute_input[:, :, 3:])
    chamfer_distance_loss = chamfer_distance_losses[0] + chamfer_distance_losses[1]
    batch_sum = torch.sum(chamfer_distance_loss)
    return batch_sum

def chamfer_loss (graph_reconstructed, graph_input):
    # Reshape for pytorch3d batch logic
    permute_points_reconstructed, _ = to_dense_batch(graph_reconstructed.position, graph_reconstructed.batch)
    permute_points_reconstructed = permute_points_reconstructed.permute(0, 2, 1)
    permute_normals_reconstructed, _ = to_dense_batch(graph_reconstructed.normal, graph_reconstructed.batch)
    permute_normals_reconstructed = permute_normals_reconstructed.permute(0, 2, 1)
    permute_points_input, _ = to_dense_batch(graph_input.position, graph_input.batch)
    permute_points_input = permute_points_input.permute(0, 2, 1)
    permute_normals_input, _ = to_dense_batch(graph_input.normal, graph_input.batch)
    permute_normals_input = permute_normals_input.permute(0, 2, 1)
    chamfer_losses = chamfer_distance(permute_points_reconstructed, permute_points_input, x_normals = permute_normals_reconstructed, y_normals = permute_normals_input)
    chamfer_loss = chamfer_losses[0] + chamfer_losses[1]
    batch_sum = torch.sum(chamfer_loss)
    return batch_sum

def mesh_distance_loss (graph_reconstructed, meshes):
    # Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html#point_mesh_face_distance
    # Convert from graph
    points_reconstructed, _ = to_dense_batch(graph_reconstructed.position, graph_reconstructed.batch)
    #normals_reconstructed, _ = to_dense_batch(graph_reconstructed.normal, graph_reconstructed.batch)
    pcls = Pointclouds(points = points_reconstructed)
    
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N
    
    # Individual weight
    point_dist = point_dist * 200
    
    return point_dist