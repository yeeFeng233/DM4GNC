import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import collections
import os
from collections import defaultdict
import scipy.sparse as sp
import umap
import seaborn as sns



def t_SNE(latents, labels, n_components=2, perplexity=30, n_iter=5000, random_state=0, 
          normalize=False, save_path=None, title="t-SNE Visualization",split_markers = False):
    """
    turn high-dimensional latents to low-dimensional visualization
    
    parameters:
        latents: high-dimensional feature representation, shape: (n_samples, n_features)
        labels: corresponding labels, shape: (n_samples,)
        n_components: dimension of the low-dimensional representation, default: 2
        perplexity: perplexity parameter of t-SNE, default: 30
        n_iter: maximum number of iterations, default: 1000
        random_state: random seed, default: 42
        normalize: whether to normalize input data, default: True
        save_path: path to save the image, if None then not save
        title: title of the image
        split_markers: whether to use different markers to distinguish between two parts of data, default: False
    """
    if torch.is_tensor(latents):
        latents = latents.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    if normalize:
        scaler = StandardScaler()
        latents_scaled = scaler.fit_transform(latents)
    else:
        latents_scaled = latents
    
    
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                max_iter=n_iter, 
                random_state=random_state,
                init='random',
                learning_rate='auto')
    
    embedded = tsne.fit_transform(latents_scaled)
    
    # create visualization
    fig, ax = plt.subplots(figsize=(10,8))
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))

    if split_markers:
        total_samples = len(labels)
        split_point = total_samples // 2
    
    # draw scatter plot for each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        x_coords = embedded[mask, 0]
        y_coords = embedded[mask, 1]
        if split_markers:
            label_indices = np.where(mask)[0]
            
            for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                original_idx = label_indices[j]
                marker = 's' if original_idx >= split_point else 'o'
                ax.scatter(x, y, 
                        c=[colors[i]], 
                        alpha=0.7, 
                        s=20, 
                        edgecolors='white', 
                        linewidth=0.5,
                        marker=marker)
        else:
            ax.scatter(x_coords, y_coords, 
                    c=[colors[i]], 
                    label=f'Class {label}', 
                    alpha=0.7, 
                    s=20, 
                    edgecolors='white', 
                    linewidth=0.5)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f't-SNE Component 1', fontsize=12)
    ax.set_ylabel(f't-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                frameon=True, fancybox=True, shadow=True)
    
    x_margin = (embedded[:, 0].max() - embedded[:, 0].min()) * 0.1
    y_margin = (embedded[:, 1].max() - embedded[:, 1].min()) * 0.1
    ax.set_xlim(embedded[:, 0].min() - x_margin, embedded[:, 0].max() + x_margin)
    ax.set_ylim(embedded[:, 1].min() - y_margin, embedded[:, 1].max() + y_margin)
    
    # add class statistics
    class_counts = collections.Counter(labels)
    stats_text = "class statistics:\n"
    for label in sorted(unique_labels):
        stats_text += f"Class {label}: {class_counts[label]} samples\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.show()


def analyze_neighbor_class_distribution(adj, pred_adj, labels, root_dir, only_generated=False,
                                       thresholds=[0.5, 0.7, 0.9], figsize=(16, 10)): 
    # Create save directory
    os.makedirs(root_dir, exist_ok=True)
    
    # Convert data formats
    if isinstance(adj, torch.Tensor):
        adj_np = adj.cpu().numpy() if adj.is_cuda else adj.numpy()
    elif sp.issparse(adj):
        adj_np = adj.toarray()
    elif isinstance(adj, np.ndarray):
        adj_np = adj
    else:
        raise ValueError(f"Unsupported adjacency matrix type: {type(adj)}")
    
    if isinstance(pred_adj, torch.Tensor):
        pred_adj_np = pred_adj.cpu().numpy() if pred_adj.is_cuda else pred_adj.numpy()
    else:
        pred_adj_np = pred_adj
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    else:
        labels_np = labels
    
    # Get class information
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    
    # 1. Analyze neighbor class distribution of real adjacency matrix
    neighbor_class_dist_real = defaultdict(lambda: defaultdict(int))
    

    for node_idx in range(adj.shape[0]):
        node_class = labels_np[node_idx]
        # Get all neighbors of this node
        neighbors = np.where(adj_np[node_idx] > 0)[0]
        
        for neighbor_idx in neighbors:
            neighbor_class = labels_np[neighbor_idx]
            neighbor_class_dist_real[node_class][neighbor_class] += 1
    
    # Plot neighbor class distribution of real adjacency matrix
    # Adjust figure size based on number of classes
    width = max(16, n_classes * 4)
    height = max(10, 8)
    fig, axes = plt.subplots(1, n_classes, figsize=(width, height))
    if n_classes == 1:
        axes = [axes]
    
    for i, class_id in enumerate(unique_classes):
        ax = axes[i]
        
        # Prepare data
        neighbor_counts = []
        class_labels = []
        for neighbor_class in unique_classes:
            count = neighbor_class_dist_real[class_id].get(neighbor_class, 0)
            neighbor_counts.append(count)
            class_labels.append(f"Class {neighbor_class}")
        
        # Calculate percentages
        total_neighbors = sum(neighbor_counts)
        if total_neighbors > 0:
            percentages = [count / total_neighbors * 100 for count in neighbor_counts]
        else:
            percentages = [0] * len(neighbor_counts)
        
        # Plot bar chart
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
        bars = ax.bar(class_labels, neighbor_counts, color=colors)
        
        # Add percentage labels
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'Class {class_id} Neighbor Distribution\n(Total neighbors: {total_neighbors})', fontsize=12)
        ax.set_xlabel('Neighbor Class', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels and adjust spacing
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    plt.suptitle('Real Adjacency Matrix: Neighbor Class Distribution by Node Class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    save_path = os.path.join(root_dir, 'real_adj_neighbor_class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Real adjacency matrix neighbor class distribution plot saved to: {save_path}")
    
    # 2. Analyze neighbor class distribution of predicted adjacency matrix at different thresholds
    results = {
        'real_distribution': dict(neighbor_class_dist_real),
        'pred_distributions': {}
    }
    
    if only_generated:
        n1,n2 = adj.shape[0], pred_adj.shape[0]
        node_indices = range(n1,n2)
        pred_adj_binary = pred_adj_np[:,:n1]
    else:
        n2 = pred_adj.shape[0]
        node_indices = range(n2)

    for threshold in thresholds:
        neighbor_class_dist_pred = defaultdict(lambda: defaultdict(int))
        
        # Binarize predicted adjacency matrix based on threshold
        pred_adj_binary = (pred_adj_np > threshold).astype(int)
        
        for node_idx in node_indices:
            node_class = labels_np[node_idx]
            # Get all neighbors of this node
            neighbors = np.where(pred_adj_binary[node_idx] > 0)[0]
            
            for neighbor_idx in neighbors:
                neighbor_class = labels_np[neighbor_idx]
                neighbor_class_dist_pred[node_class][neighbor_class] += 1
        
        results['pred_distributions'][threshold] = dict(neighbor_class_dist_pred)
        
        # Plot neighbor class distribution of predicted adjacency matrix
        # Adjust figure size based on number of classes
        width = max(16, n_classes * 4)
        height = max(10, 8)
        fig, axes = plt.subplots(1, n_classes, figsize=(width, height))
        if n_classes == 1:
            axes = [axes]
        
        for i, class_id in enumerate(unique_classes):
            ax = axes[i]
            
            # Prepare data
            neighbor_counts = []
            class_labels = []
            for neighbor_class in unique_classes:
                count = neighbor_class_dist_pred[class_id].get(neighbor_class, 0)
                neighbor_counts.append(count)
                class_labels.append(f"Class {neighbor_class}")
            
            # Calculate percentages
            total_neighbors = sum(neighbor_counts)
            if total_neighbors > 0:
                percentages = [count / total_neighbors * 100 for count in neighbor_counts]
            else:
                percentages = [0] * len(neighbor_counts)
            
            # Plot bar chart
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
            bars = ax.bar(class_labels, neighbor_counts, color=colors)
            
            # Add percentage labels
            for j, (bar, pct) in enumerate(zip(bars, percentages)):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'Class {class_id} Neighbor Distribution\n(Total neighbors: {total_neighbors})', fontsize=12)
            ax.set_xlabel('Neighbor Class', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels and adjust spacing
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        
        plt.suptitle(f'Predicted Adjacency Matrix (Threshold={threshold}): Neighbor Class Distribution by Node Class', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        save_path = os.path.join(root_dir, f'pred_adj_neighbor_class_distribution_threshold_{threshold}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        print(f"Predicted adjacency matrix neighbor class distribution plot (threshold={threshold}) saved to: {save_path}")
    
    # 3. Plot comparison across different thresholds
    # Adjust figure size for better spacing
    width = max(20, 5 * (len(thresholds) + 1))
    height = max(12, 5 * n_classes)
    fig, axes = plt.subplots(n_classes, len(thresholds) + 1, figsize=(width, height))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_id in enumerate(unique_classes):
        # Plot real distribution
        ax = axes[i, 0]
        neighbor_counts = []
        for neighbor_class in unique_classes:
            count = neighbor_class_dist_real[class_id].get(neighbor_class, 0)
            neighbor_counts.append(count)
        
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
        ax.bar(range(n_classes), neighbor_counts, color=colors)
        ax.set_title(f'Class {class_id} - Real', fontsize=12)
        ax.set_xlabel('Neighbor Class', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels([f'C{c}' for c in unique_classes], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot predicted distributions at different thresholds
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j + 1]
            neighbor_counts = []
            for neighbor_class in unique_classes:
                count = results['pred_distributions'][threshold].get(class_id, {}).get(neighbor_class, 0)
                neighbor_counts.append(count)
            
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
            ax.bar(range(n_classes), neighbor_counts, color=colors)
            ax.set_title(f'Class {class_id} - Threshold={threshold}', fontsize=12)
            ax.set_xlabel('Neighbor Class', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels([f'C{c}' for c in unique_classes], fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Neighbor Class Distribution Comparison (Real vs Different Threshold Predictions)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    save_path = os.path.join(root_dir, 'neighbor_class_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Neighbor class distribution comparison plot saved to: {save_path}")
    
    return results

def visualize_umap_comparison(x, y, x_syn=None, y_syn=None, save_path=None):
    """
    Visualizes and compares two sets of embeddings using UMAP.

    The UMAP transformation is learned from the first set of embeddings ('x')
    and then applied to both 'x' and the second set ('x_syn').

    Args:
        x (torch.Tensor): The original embeddings (shape: [n_samples, embed_dim]).
        y (torch.Tensor): The labels for the original embeddings (shape: [n_samples]).
        x_syn (torch.Tensor): The synthetic embeddings to compare (shape: [n_samples, embed_dim]).
    """
    print("Converting PyTorch tensors to NumPy arrays for processing.")
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    x_syn_np = x_syn.detach().cpu().numpy() if x_syn is not None else None
    y_syn_np = y_syn.detach().cpu().numpy() if y_syn is not None else None
    num_classes = len(np.unique(y_np))

    # --- 2. UMAP Fitting and Transforming ---
    print("Initializing and fitting UMAP model on the original data 'x'...")
    # We create a UMAP reducer object.
    # The model will be trained (fit) only on the original data.
    reducer = umap.UMAP(
        n_neighbors=10,   # Affects local vs. global structure balance
        min_dist=0.1,     # Controls how tightly points are packed
        n_components=2,   # We want a 2D projection
        random_state=42   # For reproducible results
    )
    reducer.fit(x_np)
    print("UMAP model has been fitted.")

    print("Transforming both original 'x' and synthetic 'x_syn' embeddings...")
    # Use the FITTED reducer to transform both datasets.
    # This projects them into the SAME 2D space.
    x_umap = reducer.transform(x_np)
    x_syn_umap = reducer.transform(x_syn_np) if x_syn_np is not None else None
    print("Transformation complete.")

    # --- 3. Plotting the Comparison ---
    print("Generating the comparison scatter plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot the original data embeddings, colored by their labels
    scatter1 = ax.scatter(
        x_umap[:, 0],
        x_umap[:, 1],
        c=y_np,
        cmap=plt.cm.get_cmap("jet", num_classes),
        alpha=0.6,
        s=30, # Smaller size for original points
        label="Original Data (x)"
    )

    # Overlay the synthetic data embeddings on the same plot
    # We use a different marker ('x') and color (black) to distinguish them
    if x_syn_umap is not None:
        if y_syn_np is not None:
            scatter2 = ax.scatter(
                x_syn_umap[:, 0],
                x_syn_umap[:, 1],
                c=y_syn_np,
                cmap=plt.cm.get_cmap("jet", num_classes),
                marker='x',
                alpha=0.6,
                s=30, # Slightly larger for visibility
                label="Synthetic Data (x_syn)"
            )
        else:
            scatter2 = ax.scatter(
                x_syn_umap[:, 0],
                x_syn_umap[:, 1],
                c='k',
                marker='x',
                alpha=0.2,
                s=30, # Slightly larger for visibility
                label="Synthetic Data (x_syn)"
            )

    # --- Plot Customization ---
    # Create a colorbar for the class labels of the original data
    cbar = fig.colorbar(scatter1, ax=ax, ticks=np.arange(num_classes))
    cbar.set_label('Class Label (for Original Data)', rotation=270, labelpad=20)

    # Create a legend for the two different datasets
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Original Data (x)', markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], marker='x', color='black', label='Synthetic Data (x_syn)', linestyle='None', markersize=10)
    ], fontsize=12)


    ax.set_title("UMAP Comparison of Original vs. Synthetic Embeddings", fontsize=18)
    ax.set_xlabel("UMAP Dimension 1", fontsize=14)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    print("Visualization complete.")

def calculate_degree(adj, plot=True, save_path=None, title="Degree Distribution", bins='auto', figsize=(10, 6)):
    """
    Calculate the degree of the adjacency matrix and plot the degree distribution histogram
    
    Args:
        adj: the adjacency matrix, support the following formats:
             - torch.Tensor
             - numpy.ndarray  
             - scipy.sparse matrix 
        plot: whether to plot the histogram, default True
        save_path: the path to save the picture, default None (not save)
        title: the title of the plot
        bins: the number of bins of the histogram, default 'auto'
        figsize: the size of the plot, default (10, 6)
    
    Returns:
        dict: the dictionary of the degree statistics
    """
    
    if isinstance(adj, torch.Tensor):
        if adj.is_cuda:
            adj_np = adj.cpu().numpy()
        else:
            adj_np = adj.numpy()
    elif sp.issparse(adj):
        adj_np = adj.toarray()
    elif isinstance(adj, np.ndarray):
        adj_np = adj
    else:
        raise ValueError(f"Unsupported adjacency matrix type: {type(adj)}")
    
    if adj_np.ndim != 2:
        raise ValueError(f"Adjacency matrix should be 2-dimensional, but got {adj_np.ndim} dimensions")
    
    degrees = np.sum(adj_np, axis=1)
    if len(degrees) == 0:
        print("No edges in the graph")
        return None
    
    
    # calculate statistics
    num_nodes = adj_np.shape[0]
    total_edges = int(np.sum(adj_np) / 2)
    min_degree = int(np.min(degrees))
    max_degree = int(np.max(degrees))
    mean_degree = float(np.mean(degrees))
    std_degree = float(np.std(degrees))
    
    stats = {
        'degrees': degrees,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'mean_degree': mean_degree,
        'std_degree': std_degree,
        'total_edges': total_edges,
        'num_nodes': num_nodes
    }
    
    # plot degree distribution histogram
    if plot:
        plt.figure(figsize=figsize)
        sns.set_style("whitegrid")

        n, bins, patches = plt.hist(degrees, bins=bins, alpha=0.7, color='skyblue', 
                                   edgecolor='black', linewidth=0.5)

        plt.axvline(mean_degree, color='red', linestyle='--', 
                   label=f'Mean degree: {mean_degree:.2f}')
        plt.axvline(mean_degree + std_degree, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean + std: {mean_degree + std_degree:.2f}')
        plt.axvline(mean_degree - std_degree, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean - std: {mean_degree - std_degree:.2f}')
        
        plt.xlabel('Degree', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{title}\n(Number of nodes: {num_nodes}, Number of edges: {total_edges})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        textstr = f'Minimum degree: {min_degree}\nMaximum degree: {max_degree}\nMean degree: {mean_degree:.2f}\nStandard deviation: {std_degree:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.7, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Degree distribution histogram saved to: {save_path}")
    
    return stats