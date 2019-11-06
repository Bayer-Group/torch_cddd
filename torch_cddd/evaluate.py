import numpy as np
import torch
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, roc_auc_score
from torch_cddd.data import TOKENS, batch_to_device


def evaluate_qsar(model, dataloader, device, clf_type):
    model.eval()
    emb = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_tensor, input_length, target_tensor, labels_, = batch_to_device(batch, device)
            embedding = model.encode(input_tensor, input_length)
            emb.append(embedding.detach().cpu().numpy())
            labels.append(labels_.detach().cpu().numpy())
        emb = np.concatenate(emb)
        emb = (emb - emb.mean(axis=0)) / emb.std(axis=0)
        labels = np.concatenate(labels).squeeze()
        score = fit_and_eval_qsar_model(emb, labels, clf_type)
        return score

def fit_and_eval_qsar_model(x, y, clf_type):
    idxs = [i for i in range(len(y))]
    train_idxs = idxs[:int(0.8 * len(y))]
    test_idxs = idxs[int(0.8 * len(y)):]
    if clf_type == "SVC":
        clf = SVC(C=5.0, probability=True, gamma="auto")
        clf.fit(x[train_idxs], y[train_idxs])
        pred_prob = clf.predict_proba(x[test_idxs])[:, 1]
        score = roc_auc_score(y[test_idxs], pred_prob)
    elif clf_type == "SVR":
        clf = SVR(C=5.0, gamma="auto")
        clf.fit(x[train_idxs], y[train_idxs])
        pred = clf.predict(x[test_idxs])
        score = r2_score(y[test_idxs], pred)
    return score


def evaluate_reconstruction(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        num_matches = 0
        num_total = 0
        for batch in dataloader:
            input_tensor, input_length, target_tensor, labels, = batch_to_device(batch, device)
            loss, out = model.forward(input_tensor, input_length, target_tensor, labels)
            eval_loss += loss.detach().cpu().numpy()
            nm, nt = sequence_match(
                seq_pred=out.detach().cpu().numpy(),
                seq_true=target_tensor.detach().cpu().numpy())
            num_matches += nm
            num_total += nt
        eval_loss /= len(dataloader)
        mean_acc = num_matches / num_total
    return eval_loss, mean_acc


def sequence_match(seq_pred, seq_true):
    # no SOS token
    seq_true = seq_true[:, 1:]
    mask = np.array(seq_true == TOKENS.index("PAD"), dtype=np.int)
    matches = np.array((seq_pred == seq_true), dtype=np.int)
    num_matches = np.ma.array(matches, mask=mask).sum()
    num_total = seq_true.shape[0] * seq_true.shape[1] - mask.sum()
    return num_matches, num_total
