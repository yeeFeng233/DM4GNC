import random
import torch
from torch_geometric.utils import degree
import numpy as np


def split_nodes_by_degree(data, imb_level='low', seed=42):
    torch.manual_seed(seed)

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    deg_threshold = torch.quantile(deg.float(), 0.8)  
    high_deg_mask = deg >= deg_threshold  

    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1  

    total_train_size = num_nodes // 10  
    total_val_size   = num_nodes // 10  

    train_per_class = total_train_size // num_classes
    val_per_class   = total_val_size   // num_classes

    if imb_level == 'low':
        ratio = 0.1
    elif imb_level == 'mid':
        ratio = 0.2
    elif imb_level == 'high':
        ratio = 0.3

    train_index_list = []
    val_index_list   = []
    test_index_list  = []

    assigned_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        class_nodes = torch.where(data.y == c)[0]

        class_high_deg_nodes = class_nodes[high_deg_mask[class_nodes]]
        class_low_deg_nodes  = class_nodes[~high_deg_mask[class_nodes]]

        needed_train = train_per_class
        num_high_deg_needed = int(ratio * train_per_class)

        num_high_deg_for_train = min(num_high_deg_needed, needed_train)

        high_deg_train_nodes = class_high_deg_nodes[
            torch.randperm(len(class_high_deg_nodes))[:num_high_deg_for_train]
        ]

        remaining_for_train = needed_train - num_high_deg_for_train

        low_deg_train_nodes = class_low_deg_nodes[
            torch.randperm(len(class_low_deg_nodes))[:remaining_for_train]
        ]

        class_train_nodes = torch.cat([high_deg_train_nodes, low_deg_train_nodes], dim=0)
        train_index_list.append(class_train_nodes)
        assigned_mask[class_train_nodes] = True

        needed_val = val_per_class
        remain_nodes = class_nodes[~assigned_mask[class_nodes]]
        val_nodes = remain_nodes[torch.randperm(len(remain_nodes))[:needed_val]]
        val_index_list.append(val_nodes)
        assigned_mask[val_nodes] = True

        remain_nodes_after_val = class_nodes[~assigned_mask[class_nodes]]
        test_nodes = remain_nodes_after_val
        test_index_list.append(test_nodes)
        assigned_mask[test_nodes] = True

    train_index_tensor = torch.cat(train_index_list, dim=0)
    val_index_tensor   = torch.cat(val_index_list,   dim=0)
    test_index_tensor  = torch.cat(test_index_list,  dim=0)

    data.train_index = train_index_tensor.cpu().numpy()
    data.val_index   = val_index_tensor.cpu().numpy()
    data.test_index  = test_index_tensor.cpu().numpy()

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_index_tensor] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_index_tensor] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_index_tensor] = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    return data