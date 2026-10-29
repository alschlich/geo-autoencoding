import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch_geometric
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.structures import Meshes

def visualize_closeness(position, dist):
    TH = 1
    fig = plt.figure()
    position = position.detach().cpu().numpy()
    dist = dist[0].detach().cpu().numpy()
    xs_1 = position[0, 0, :]
    ys_1 = position[0, 1, :]
    zs_1 = position[0, 2, :]
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.set_zlim(-TH, TH)
    ax_1.set_xlim(-TH, TH)
    ax_1.set_ylim(-TH, TH)
    vmin = np.percentile(dist, 5)
    vmax = np.percentile(dist, 95)
    ax_1.scatter(xs_1, ys_1, zs_1, marker=".", s=80, alpha=0.9, edgecolor="none", c=dist, cmap='viridis', vmin=vmin, vmax=vmax)
    ax_1.scatter(xs_1[0], ys_1[0], zs_1[0], s=150, marker="o", alpha=0.9, edgecolor="none")
    plt.axis('off')
    #sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    #cbar = plt.colorbar(sm, ax=ax_1, pad=0.1)
    #cbar.set_label('Distance')
    return fig
    

def visualize_mesh_graph_in_points_out(graph_in, mesh_in, graph_out):
    TH = 1
    
    verts = mesh_in.verts_packed().cpu().numpy()   # (V, 3)
    faces = mesh_in.faces_packed().cpu().numpy()   # (F, 3)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')

    # --- Solid faces ---
    face_vertices = verts[faces]  # (F, 3, 3)
    mesh_collection = Poly3DCollection(face_vertices, 
                                       facecolors='lightgray', 
                                       edgecolors='black', 
                                       linewidths=0.5)
    ax.add_collection3d(mesh_collection)

    # --- Set plot limits ---
    ax.set_xlim(-TH, TH)
    ax.set_ylim(-TH, TH)
    ax.set_zlim(-TH, TH)

    # --- Labels and view ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis('off')
    
    # Plot input graph
    ax_1 = fig.add_subplot(132, projection="3d")
    ax_1.set_zlim(-TH, TH)
    ax_1.set_xlim(-TH, TH)
    ax_1.set_ylim(-TH, TH)
    
    position_in = graph_in.position.detach().cpu().numpy()
    xs_1 = position_in[:, 0]
    ys_1 = position_in[:, 1]
    zs_1 = position_in[:, 2]
    
    ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none")
    plt.axis('off')
    
    # Plot output graph
    ax_1 = fig.add_subplot(133, projection="3d")
    ax_1.set_zlim(-TH, TH)
    ax_1.set_xlim(-TH, TH)
    ax_1.set_ylim(-TH, TH)
    
    position_out = graph_out.position.detach().cpu().numpy()
    xs_2 = position_out[:, 0]
    ys_2 = position_out[:, 1]
    zs_2 = position_out[:, 2]
    
    ax_1.scatter(xs_2, ys_2, zs_2, marker=".", alpha=0.9, edgecolor="none")
    plt.axis('off')
    
    return fig

def visualize_in_out_graph_batch(graph_in_batch, graph_out_batch):
    figs = []
    list_in = graph_in_batch.to_data_list()
    list_out = graph_out_batch.to_data_list()
    for i, graph_in in enumerate(list_in):
        graph_out = list_out[i]
        figs.append(visualize_in_out_graph(graph_in, graph_out))
    return figs

def visualize_in_out_graph(graph_in, graph_out):
    # Assume attributes to be of size [1024, 3]
    TH = 1
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig = plt.figure()
    
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.set_title("Input")
    ax_1.set_zlim(-TH, TH)
    ax_1.set_xlim(-TH, TH)
    ax_1.set_ylim(-TH, TH)
    
    # Extract positions
    position_in = graph_in.position.detach().cpu().numpy()
    xs_1 = position_in[:, 0]
    ys_1 = position_in[:, 1]
    zs_1 = position_in[:, 2]
    position_out = graph_out.position.detach().cpu().numpy()
    xs_2 = position_out[:, 0]
    ys_2 = position_out[:, 1]
    zs_2 = position_out[:, 2]
    # Plot points
    ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none")
    ax_1.scatter(xs_2, ys_2, zs_2, marker=".", alpha=0.9, edgecolor="none")
    
    # # Extract normals
    # normal_in = graph_in.normal.detach().cpu().numpy()
    # us_1 = normal_in[:, 0]
    # vs_1 = normal_in[:, 1]
    # ws_1 = normal_in[:, 2]
    # normal_out = graph_out.normal.detach().cpu().numpy()
    # us_2 = normal_out[:, 0]
    # vs_2 = normal_out[:, 1]
    # ws_2 = normal_out[:, 2]
    # # Plot normals
    # ax_1.quiver(xs_1, ys_1, zs_1, us_1, vs_1, ws_1, length=0.02, color=colors[0], normalize=True)
    # ax_1.quiver(xs_2, ys_2, zs_2, us_2, vs_2, ws_2, length=0.02, color=colors[1], normalize=True)
    
    return fig

def visualize_graph_batch(graph_batch, show_curvature=False, show_curvature_vectors=False, show_edges=False, show_normals=False):
    figs = []
    list = graph_batch.to_data_list()
    for graph in list:
        figs.append(visualize_graph(graph, show_curvature, show_curvature_vectors, show_edges, show_normals))
    return figs

def visualize_graph(graph, show_curvature=False, show_curvature_vectors=False, show_edges=False, show_normals=False):
    # Assume attributes to be of size [1024, 3]
    TH = 1
    fig = plt.figure()
    
    ax_1 = fig.add_subplot(111, projection="3d")
    ax_1.set_title("Input")
    ax_1.set_zlim(-TH, TH)
    ax_1.set_xlim(-TH, TH)
    ax_1.set_ylim(-TH, TH)

    # Extract positions
    position = graph.position.detach().cpu().numpy()
    xs_1 = position[:, 0]
    ys_1 = position[:, 1]
    zs_1 = position[:, 2]
    
    # Plot edges
    if show_edges:
        for i, j in graph.edge_index.t():
            x_vals = [graph.position[i][0], graph.position[j][0]]
            y_vals = [graph.position[i][1], graph.position[j][1]]
            z_vals = [graph.position[i][2], graph.position[j][2]]
            ax_1.plot(x_vals, y_vals, z_vals, c='gray')
    
    # Plot curvature and legend
    if show_curvature:
        
        assert not show_curvature_vectors
        # Extract curvature values for color and scaling
        curvature = graph.curvature.detach().cpu().numpy()

        # Normalize curvature values for colormap
        norm = Normalize(vmin=curvature.min(), vmax=curvature.max())
        colors = cm.viridis(norm(curvature))
        
        # Add legend
        sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        cbar = plt.colorbar(sm, ax=ax_1, pad=0.1)
        cbar.set_label('Curvature')
    else:
        colors = None
    
    # Plot mean curvature vectors
    if show_curvature_vectors:
        assert not show_curvature
        curvature_vectors = graph.curvature_embedding.detach().cpu().numpy()
        curvature_vectors_length = np.linalg.norm(curvature_vectors, axis=1)
        norm = plt.Normalize(curvature_vectors_length.min(), curvature_vectors_length.max())
        colors = cm.viridis(norm(curvature_vectors_length))
        sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
        cbar = plt.colorbar(sm, ax=ax_1, pad=0.1)
        cbar.set_label('Mean Curvature Vector Length')
        us = curvature_vectors[:, 0]
        vs = curvature_vectors[:, 1]
        ws = curvature_vectors[:, 2]
        ax_1.quiver(xs_1, ys_1, zs_1, us, vs, ws, color=colors, length=0.02, normalize=True)

    # Plot points
    ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none", c=colors)
    
    # Plot normals with color from curvature
    if show_normals:
        # Extract normal vectors
        normal = graph.normal.detach().cpu().numpy()
        us = normal[:, 0]
        vs = normal[:, 1]
        ws = normal[:, 2]
        ax_1.quiver(xs_1, ys_1, zs_1, us, vs, ws, length=0.02, color=colors, normalize=True)
    
    return fig

def visualize_intermediate_results(points_in, keypoints, keypoints_regressed, keypoints_canon, points_out_canon, points_out, file_path):
    # Assume 3 x N
    points_in = points_in.permute(1, 0)
    keypoints = keypoints.permute(1, 0)
    keypoints_regressed = keypoints_regressed.permute(1, 0)
    keypoints_canon = keypoints_canon.permute(1, 0)
    points_out_canon = points_out_canon.permute(1, 0)
    points_out = points_out.permute(1, 0)
    
    TH = 1
    fig = plt.figure()
    
    # Plot input
    ax_1 = fig.add_subplot(331, projection="3d")
    ax_1.set_title("Input")
    ax_1.set_zlim(-TH,TH)
    ax_1.set_xlim(-TH,TH)
    ax_1.set_ylim(-TH,TH)
    xs_1 = points_in[:, 0]
    ys_1 = points_in[:, 1]
    zs_1 = points_in[:, 2]
    ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none")
    
    # Plot keypoints
    ax_2 = fig.add_subplot(332, projection="3d")
    ax_2.set_title("Keypoints")
    ax_2.set_zlim(-TH,TH)
    ax_2.set_xlim(-TH,TH)
    ax_2.set_ylim(-TH,TH)
    xs_2 = keypoints[:, 0]
    ys_2 = keypoints[:, 1]
    zs_2 = keypoints[:, 2]
    ax_2.scatter(xs_2, ys_2, zs_2, marker=".", alpha=0.9, edgecolor="none")
    
    # Plot regressed keypoints
    ax_3 = fig.add_subplot(333, projection="3d")
    ax_3.set_title("Regressed Keypoints")
    ax_3.set_zlim(-TH,TH)
    ax_3.set_xlim(-TH,TH)
    ax_3.set_ylim(-TH,TH)
    xs_3 = keypoints_regressed[:, 0]
    ys_3 = keypoints_regressed[:, 1]
    zs_3 = keypoints_regressed[:, 2]
    ax_3.scatter(xs_3, ys_3, zs_3, marker=".", alpha=0.9, edgecolor="none")
    
    # Plot canonicalized keypoints
    ax_4 = fig.add_subplot(334, projection="3d")
    ax_4.set_title("Canonicalized Keypoints")
    ax_4.set_zlim(-TH,TH)
    ax_4.set_xlim(-TH,TH)
    ax_4.set_ylim(-TH,TH)
    xs_4 = keypoints_canon[:, 0]
    ys_4 = keypoints_canon[:, 1]
    zs_4 = keypoints_canon[:, 2]
    ax_4.scatter(xs_4, ys_4, zs_4, marker=".", alpha=0.9, edgecolor="none")
    
    # Plot canonical output
    ax_5 = fig.add_subplot(335, projection="3d")
    ax_5.set_title("Canonical Output")
    ax_5.set_zlim(-TH,TH)
    ax_5.set_xlim(-TH,TH)
    ax_5.set_ylim(-TH,TH)
    xs_5 = points_out_canon[:, 0]
    ys_5 = points_out_canon[:, 1]
    zs_5 = points_out_canon[:, 2]
    ax_5.scatter(xs_5, ys_5, zs_5, marker=".", alpha=0.9, edgecolor="none")
    
    # Plot final decanonicalized output
    ax_6 = fig.add_subplot(336, projection="3d")
    ax_6.set_title("Output")
    ax_6.set_zlim(-TH,TH)
    ax_6.set_xlim(-TH,TH)
    ax_6.set_ylim(-TH,TH)
    xs_6 = points_out[:, 0]
    ys_6 = points_out[:, 1]
    zs_6 = points_out[:, 2]
    ax_6.scatter(xs_6, ys_6, zs_6, marker=".", alpha=0.9, edgecolor="none")
    
    fig.set_size_inches(16, 9)
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

def visualize_in_keypoints_out_batch(pts_batch_in, keypoints_batch, pts_batch_out, file_path):
    # Assume B x 3 x N
    pts_batch_in = pts_batch_in.permute(0, 2, 1)
    keypoints_batch = keypoints_batch.permute(0, 2, 1)
    pts_batch_out = pts_batch_out.permute(0, 2, 1)
    TH = 1
    for i, pts_in in enumerate(pts_batch_in):
        fig = plt.figure()
        ax_1 = fig.add_subplot(311, projection="3d")
        ax_1.set_title("Input")
        ax_1.set_zlim(-TH,TH)
        ax_1.set_xlim(-TH,TH)
        ax_1.set_ylim(-TH,TH)
        ax_2 = fig.add_subplot(312, projection="3d")
        ax_2.set_title("Keypoints")
        ax_2.set_zlim(-TH,TH)
        ax_2.set_xlim(-TH,TH)
        ax_2.set_ylim(-TH,TH)
        ax_3 = fig.add_subplot(313, projection="3d")
        ax_3.set_title("Output")
        ax_3.set_zlim(-TH,TH)
        ax_3.set_xlim(-TH,TH)
        ax_3.set_ylim(-TH,TH)
        xs_1 = pts_in[:, 0]
        ys_1 = pts_in[:, 1]
        zs_1 = pts_in[:, 2]
        ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none")
        xs_2 = keypoints_batch[i, :, 0]
        ys_2 = keypoints_batch[i, :, 1]
        zs_2 = keypoints_batch[i, :, 2]
        ax_2.scatter(xs_2, ys_2, zs_2, marker=".", alpha=0.9, edgecolor="none")
        xs_3 = pts_batch_out[i, :, 0]
        ys_3 = pts_batch_out[i, :, 1]
        zs_3 = pts_batch_out[i, :, 2]
        ax_3.scatter(xs_3, ys_3, zs_3, marker=".", alpha=0.9, edgecolor="none")
        plt.savefig(file_path + str(i) + ".png" , bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def visualize_in_out_batch(pts_batch_in, pts_batch_out, file_path):
    # Assume B x 3 x N
    pts_batch_in = pts_batch_in.permute(0, 2, 1)
    pts_batch_out = pts_batch_out.permute(0, 2, 1)
    TH = 1
    for i, pts_in in enumerate(pts_batch_in):
        fig = plt.figure()
        ax_1 = fig.add_subplot(211, projection="3d")
        ax_1.set_title("Input")
        ax_1.set_zlim(-TH,TH)
        ax_1.set_xlim(-TH,TH)
        ax_1.set_ylim(-TH,TH)
        ax_2 = fig.add_subplot(212, projection="3d")
        ax_2.set_title("Output")
        ax_2.set_zlim(-TH,TH)
        ax_2.set_xlim(-TH,TH)
        ax_2.set_ylim(-TH,TH)
        xs_1 = pts_in[:, 0]
        ys_1 = pts_in[:, 1]
        zs_1 = pts_in[:, 2]
        ax_1.scatter(xs_1, ys_1, zs_1, marker=".", alpha=0.9, edgecolor="none")
        xs_2 = pts_batch_out[i, :, 0]
        ys_2 = pts_batch_out[i, :, 1]
        zs_2 = pts_batch_out[i, :, 2]
        ax_2.scatter(xs_2, ys_2, zs_2, marker=".", alpha=0.9, edgecolor="none")
        plt.savefig(file_path + str(i) + ".png" , bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()