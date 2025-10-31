import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,recall_score,f1_score,accuracy_score,precision_score
import torch
import sklearn

# get accuracy
def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# get roc score and ap score
def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):
    adj_rec = adj_rec.cpu()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def link_prediction_accuracy(pred_adj, true_adj,edge_index=None):
    if edge_index is not None:
        y_pred = pred_adj[edge_index[0],edge_index[1]]
        y_true = true_adj[edge_index[0],edge_index[1]]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.int().detach().cpu().numpy()
    else:
        if isinstance(pred_adj, torch.Tensor):
            y_pred = pred_adj.view(-1).detach().cpu().numpy()
        if isinstance(true_adj, torch.Tensor):
            y_true = true_adj.view(-1).int().detach().cpu().numpy()

    ls_recall, ls_f1, ls_acc, ls_precision, =  [], [], [], []
    print("edges_num: ",y_true.sum().item())
    
    for threshold in np.arange(0.1, 1, 0.1):
        y_pred_threshold = (y_pred > threshold).astype(int)
        print(f"threshold: {threshold:.2f} | edges_num: {y_pred_threshold.sum().item()}")

        recall = recall_score(y_true, y_pred_threshold, average='binary')
        f1 = f1_score(y_true, y_pred_threshold, average='binary')
        acc = accuracy_score(y_true, y_pred_threshold)
        precision = precision_score(y_true, y_pred_threshold, average='binary', zero_division=0)

        print(f"threshold: {threshold:.2f} | recall: {recall:.5f} | precision: {precision:.5f} | f1: {f1:.5f} | acc: {acc:.5f} ")
        ls_recall.append(recall)
        ls_f1.append(f1)
        ls_acc.append(acc)
        ls_precision.append(precision)
    
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    print(f"auc: {auc:.5f}")
    
    return (ls_recall, ls_f1, ls_acc, ls_precision), auc

