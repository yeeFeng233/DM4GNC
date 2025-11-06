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



def t_SNE(latents, labels, x_syn=None, y_syn=None, n_components=2, perplexity=30, n_iter=5000, random_state=0, 
          normalize=False, save_path=None, title="t-SNE Visualization"):
    """
    Turn high-dimensional latents to low-dimensional visualization with optional synthetic data comparison.
    
    Parameters:
        latents: high-dimensional feature representation, shape: (n_samples, n_features)
        labels: corresponding labels, shape: (n_samples,)
        x_syn: synthetic embeddings to compare (optional), shape: (n_samples_syn, n_features)
        y_syn: labels for synthetic embeddings (optional), shape: (n_samples_syn,)
        n_components: dimension of the low-dimensional representation, default: 2
        perplexity: perplexity parameter of t-SNE, default: 30
        n_iter: maximum number of iterations, default: 5000
        random_state: random seed, default: 0
        normalize: whether to normalize input data, default: False
        save_path: path to save the image, if None then not save
        title: title of the image
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(latents):
        latents = latents.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if x_syn is not None and torch.is_tensor(x_syn):
        x_syn = x_syn.cpu().numpy()
    if y_syn is not None and torch.is_tensor(y_syn):
        y_syn = y_syn.cpu().numpy()

    # Determine if we have synthetic data
    has_synthetic = x_syn is not None and y_syn is not None
    
    # Combine data if synthetic data is provided
    if has_synthetic:
        combined_data = np.vstack([latents, x_syn])
        combined_labels = np.concatenate([labels, y_syn])
        n_original = len(latents)
    else:
        combined_data = latents
        combined_labels = labels
        n_original = len(latents)

    # Normalize if required
    if normalize:
        scaler = StandardScaler()
        combined_data_scaled = scaler.fit_transform(combined_data)
    else:
        combined_data_scaled = combined_data
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                max_iter=n_iter, 
                random_state=random_state,
                init='random',
                learning_rate='auto')
    
    embedded = tsne.fit_transform(combined_data_scaled)
    
    # Split embedded data back into original and synthetic
    embedded_original = embedded[:n_original]
    labels_original = combined_labels[:n_original]
    
    if has_synthetic:
        embedded_synthetic = embedded[n_original:]
        labels_synthetic = combined_labels[n_original:]
    
    # Create visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    unique_labels = np.unique(combined_labels)
    n_classes = len(unique_labels)
    
    # Define deep, high-contrast colors
    deep_colors = [
        '#1f77b4',  # Deep blue
        '#d62728',  # Deep red
        '#2ca02c',  # Deep green
        '#ff7f0e',  # Deep orange
        '#9467bd',  # Deep purple
        '#8c564b',  # Deep brown
        '#e377c2',  # Deep pink
        '#7f7f7f',  # Deep gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#8B0000',  # Dark red
        '#006400',  # Dark green
        '#00008B',  # Dark blue
        '#8B008B',  # Dark magenta
        '#FF8C00',  # Dark orange
        '#483D8B',  # Dark slate blue
        '#2F4F4F',  # Dark slate gray
        '#8B4513',  # Saddle brown
        '#4B0082',  # Indigo
        '#800000',  # Maroon
    ]
    
    # Create color mapping for labels
    color_map = {label: deep_colors[i % len(deep_colors)] for i, label in enumerate(unique_labels)}
    colors_original = [color_map[label] for label in labels_original]
    
    # Plot original data with circle markers
    scatter1 = ax.scatter(
        embedded_original[:, 0],
        embedded_original[:, 1],
        c=colors_original,
        alpha=0.7,
        s=35,
        marker='o',
        edgecolors='black',
        linewidth=0.6,
        label='Original Data'
    )
    
    # Plot synthetic data with 'x' markers if available
    if has_synthetic:
        colors_synthetic = [color_map[label] for label in labels_synthetic]
        scatter2 = ax.scatter(
            embedded_synthetic[:, 0],
            embedded_synthetic[:, 1],
            c=colors_synthetic,
            alpha=0.7,
            s=35,
            marker='x',
            linewidth=2.0,
            label='Synthetic Data'
        )
    
    # Create legend for classes and data types
    legend_elements = []
    
    # Add class color legend
    for label in sorted(unique_labels):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      label=f'Class {label}',
                      markerfacecolor=color_map[label], 
                      markersize=10,
                      markeredgecolor='black',
                      markeredgewidth=0.6)
        )
    
    # Add separator and data type markers if synthetic data exists
    if has_synthetic:
        legend_elements.append(plt.Line2D([0], [0], linestyle='', marker=''))  # Separator
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Original Data', 
                      markerfacecolor='gray', markersize=10,
                      markeredgecolor='black', markeredgewidth=0.6)
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker='x', color='gray', label='Synthetic Data', 
                      linestyle='None', markersize=10, markeredgewidth=2.0)
        )
    
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right', 
             frameon=True, fancybox=True, shadow=True)
    
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=14)
    ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set margins
    x_margin = (embedded[:, 0].max() - embedded[:, 0].min()) * 0.1
    y_margin = (embedded[:, 1].max() - embedded[:, 1].min()) * 0.1
    ax.set_xlim(embedded[:, 0].min() - x_margin, embedded[:, 0].max() + x_margin)
    ax.set_ylim(embedded[:, 1].min() - y_margin, embedded[:, 1].max() + y_margin)
    
    # Add class statistics
    class_counts_original = collections.Counter(labels_original)
    stats_text = "Class statistics (Original):\n"
    for label in sorted(unique_labels):
        count_orig = class_counts_original.get(label, 0)
        stats_text += f"Class {label}: {count_orig} samples\n"
    
    if has_synthetic:
        class_counts_synthetic = collections.Counter(labels_synthetic)
        stats_text += "\nClass statistics (Synthetic):\n"
        for label in sorted(unique_labels):
            count_syn = class_counts_synthetic.get(label, 0)
            stats_text += f"Class {label}: {count_syn} samples\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"t-SNE visualization saved to {save_path}")
    plt.show()
    print("t-SNE visualization complete.")


def analyze_neighbor_class_distribution(adj, pred_adj, labels, pred_labels, root_dir,
                                       thresholds=[0.5, 0.7, 0.9]): 
    """
    Analyze and visualize neighbor class distribution for real and predicted graphs.
    
    Args:
        adj: Real adjacency matrix (n x n)
        pred_adj: Predicted adjacency matrix (n x n), same size as adj
        labels: Real node labels (n,)
        pred_labels: Predicted node labels (n,)
        root_dir: Directory to save results
        thresholds: List of thresholds for binarizing pred_adj
    
    Returns:
        results: Dictionary containing distribution statistics
    """
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
    elif sp.issparse(pred_adj):
        pred_adj_np = pred_adj.toarray()
    elif isinstance(pred_adj, np.ndarray):
        pred_adj_np = pred_adj
    else:
        raise ValueError(f"Unsupported predicted adjacency matrix type: {type(pred_adj)}")
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    else:
        labels_np = labels
    
    if isinstance(pred_labels, torch.Tensor):
        pred_labels_np = pred_labels.cpu().numpy() if pred_labels.is_cuda else pred_labels.numpy()
    else:
        pred_labels_np = pred_labels
    
    # Verify dimensions match
    if adj_np.shape != pred_adj_np.shape:
        raise ValueError(f"adj shape {adj_np.shape} != pred_adj shape {pred_adj_np.shape}")
    if len(labels_np) != adj_np.shape[0]:
        raise ValueError(f"labels length {len(labels_np)} != adj shape {adj_np.shape[0]}")
    if len(pred_labels_np) != adj_np.shape[0]:
        raise ValueError(f"pred_labels length {len(pred_labels_np)} != adj shape {adj_np.shape[0]}")
    
    # Get class information from real labels
    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    
    # 1. Analyze neighbor class distribution of real adjacency matrix
    print(f"\n{'='*60}")
    print("Analyzing REAL graph neighbor class distribution...")
    print(f"{'='*60}")
    neighbor_class_dist_real = defaultdict(lambda: defaultdict(int))

    for node_idx in range(adj_np.shape[0]):  # Fixed: use adj_np instead of adj
        node_class = labels_np[node_idx]
        # Get all neighbors of this node
        neighbors = np.where(adj_np[node_idx] > 0)[0]
        
        for neighbor_idx in neighbors:
            neighbor_class = labels_np[neighbor_idx]
            neighbor_class_dist_real[node_class][neighbor_class] += 1
    
    print(f"Real graph: {adj_np.shape[0]} nodes, {int(adj_np.sum()/2)} edges")
    
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
    
    plt.suptitle('Real Graph: Neighbor Class Distribution by Node Class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    save_path = os.path.join(root_dir, 'real_neighbor_class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"✓ Real graph neighbor class distribution saved to: {save_path}")
    
    # 2. Analyze neighbor class distribution of predicted adjacency matrix at different thresholds
    print(f"\n{'='*60}")
    print("Analyzing PREDICTED graph neighbor class distribution...")
    print(f"{'='*60}")
    
    results = {
        'real_distribution': dict(neighbor_class_dist_real),
        'pred_distributions': {}
    }
    
    # Analyze all nodes in predicted graph
    n_nodes = adj_np.shape[0]
    node_indices = range(n_nodes)

    for threshold in thresholds:
        print(f"\nProcessing threshold = {threshold}...")
        neighbor_class_dist_pred = defaultdict(lambda: defaultdict(int))
        
        # Binarize predicted adjacency matrix based on threshold
        pred_adj_binary = (pred_adj_np > threshold).astype(int)
        num_edges = int(pred_adj_binary.sum() / 2)
        
        for node_idx in node_indices:
            node_class = pred_labels_np[node_idx]  # Use predicted labels
            # Get all neighbors of this node
            neighbors = np.where(pred_adj_binary[node_idx] > 0)[0]
            
            for neighbor_idx in neighbors:
                neighbor_class = pred_labels_np[neighbor_idx]  # Use predicted labels
                neighbor_class_dist_pred[node_class][neighbor_class] += 1
        
        results['pred_distributions'][threshold] = dict(neighbor_class_dist_pred)
        print(f"  Predicted graph at threshold {threshold}: {n_nodes} nodes, {num_edges} edges")
        
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
        
        plt.suptitle(f'Predicted Graph (Threshold={threshold}): Neighbor Class Distribution by Node Class', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        save_path = os.path.join(root_dir, f'pred_neighbor_class_distribution_threshold_{threshold}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        print(f"  ✓ Predicted graph distribution (threshold={threshold}) saved")
    
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
    
    plt.suptitle('Neighbor Class Distribution Comparison (Real vs Predicted at Different Thresholds)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    save_path = os.path.join(root_dir, 'neighbor_class_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"\n✓ Comparison plot saved to: {save_path}")
    print(f"{'='*60}\nAnalysis complete!\n{'='*60}")
    
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

    # Define deep, high-contrast colors (same as t_SNE)
    deep_colors = [
        '#1f77b4',  # Deep blue
        '#d62728',  # Deep red
        '#2ca02c',  # Deep green
        '#ff7f0e',  # Deep orange
        '#9467bd',  # Deep purple
        '#8c564b',  # Deep brown
        '#e377c2',  # Deep pink
        '#7f7f7f',  # Deep gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#8B0000',  # Dark red
        '#006400',  # Dark green
        '#00008B',  # Dark blue
        '#8B008B',  # Dark magenta
        '#FF8C00',  # Dark orange
        '#483D8B',  # Dark slate blue
        '#2F4F4F',  # Dark slate gray
        '#8B4513',  # Saddle brown
        '#4B0082',  # Indigo
        '#800000',  # Maroon
    ]
    
    # Create color mapping for labels
    unique_labels = np.unique(y_np)
    color_map = {label: deep_colors[i % len(deep_colors)] for i, label in enumerate(unique_labels)}
    colors_original = [color_map[label] for label in y_np]

    # Plot the original data embeddings, colored by their labels
    scatter1 = ax.scatter(
        x_umap[:, 0],
        x_umap[:, 1],
        c=colors_original,
        alpha=0.7,
        s=35,
        marker='o',
        edgecolors='black',
        linewidth=0.6,
        label="Original Data (x)"
    )

    # Overlay the synthetic data embeddings on the same plot
    # We use a different marker ('x') to distinguish them
    if x_syn_umap is not None:
        if y_syn_np is not None:
            colors_synthetic = [color_map[label] for label in y_syn_np]
            scatter2 = ax.scatter(
                x_syn_umap[:, 0],
                x_syn_umap[:, 1],
                c=colors_synthetic,
                marker='x',
                alpha=0.7,
                s=35,
                linewidth=2.0,
                label="Synthetic Data (x_syn)"
            )
        else:
            scatter2 = ax.scatter(
                x_syn_umap[:, 0],
                x_syn_umap[:, 1],
                c='k',
                marker='x',
                alpha=0.2,
                s=35,
                linewidth=2.0,
                label="Synthetic Data (x_syn)"
            )

    # --- Plot Customization ---
    # Create legend for classes and data types
    legend_elements = []
    
    # Add class color legend
    for label in sorted(unique_labels):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      label=f'Class {label}',
                      markerfacecolor=color_map[label], 
                      markersize=10,
                      markeredgecolor='black',
                      markeredgewidth=0.6)
        )
    
    # Add separator and data type markers
    legend_elements.append(plt.Line2D([0], [0], linestyle='', marker=''))  # Separator
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', label='Original Data (x)', 
                  markerfacecolor='gray', markersize=10,
                  markeredgecolor='black', markeredgewidth=0.6)
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='x', color='gray', label='Synthetic Data (x_syn)', 
                  linestyle='None', markersize=10, markeredgewidth=2.0)
    )
    
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right',
             frameon=True, fancybox=True, shadow=True)


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