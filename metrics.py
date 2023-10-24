import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def get_token_acc(preds, labels):
    # print(f"Metrice get token acc")
    preds = np.argmax(preds, -1)

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    acc_token = accuracy_score(labels, preds)

    return acc_token

def get_token_acc_fixed(labels, inps, _model, mlm_collator, p, size=100):
    # print(f"Metrice get token acc fixed")
    _model.eval()
    inps = inps[np.random.choice(inps.shape[0], size= size, replace=False)]
    labels = labels[np.random.choice(labels.shape[0], size= size, replace=False)]

    org_ids = np.zeros_like(inps)
    mask = labels != -100
    org_ids[mask] = labels[mask]
    org_ids[~mask] = inps[~mask]

    org_ids, org_labels = mlm_collator.masking_tokens(torch.Tensor(org_ids).to(_model.device).long(), p)
    org_logits = _model(org_ids)['logits'].detach().cpu().numpy()
    org_acc = get_token_acc(org_logits, org_labels.detach().cpu().numpy())
    return org_acc


def get_entropy(preds, labels):
    # print(f"Metrice get entropy")
    # preds, labels, inps = eval_preds
    mask = labels != -100
    log_probs = np.log(np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True))
    entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    entropy = entropy[mask].mean(axis=-1)
    return entropy


def get_perr(preds, labels):
    print(f"Metrice get perr")
    preds = np.argmax(preds, -1)
    mask = labels != -100
    p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, labels, mask))).mean()
    return p_err


def get_memorization(_data: torch.Tensor, _model, mask_token_id, size=100):
    # print(f"Metrice get memorization")
    if ~ isinstance(_data, torch.Tensor):
        _data = torch.tensor(_data).long()
        _data = _data.to(_model.device)

    _inps = _data[np.random.choice(_data.size(0), size= size, replace=False)].clone()
    _batch_size = _inps.size(0)
    _chunk_size = _inps.size(1)

    mask = np.random.choice(np.arange(_chunk_size).astype(int), size=(_batch_size, ))
    mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to(_model.device), num_classes=_chunk_size).bool()

    _labels = _inps.clone()
    _labels[~mask_onehot] = -100

    _inps[mask_onehot] = mask_token_id

    with torch.no_grad():
        _model.eval()
        _output = _model(_inps)
    _logits = _output['logits'].detach().cpu().numpy()
    _preds = _logits.argmax(-1)

    _labels = _labels.detach().cpu().numpy()
    mask_onehot = mask_onehot.detach().cpu().numpy()
    return np.mean(_preds[mask_onehot] == _labels[mask_onehot])


# def get_memorization_trained(_logits, _labels):
#     _preds = _logits.argmax(-1)
#     _mask = _labels != -100
#     _labels = _labels.detach().cpu().numpy()
#     mask_onehot = mask_onehot.detach().cpu().numpy()
#     return np.mean(_preds[mask_onehot] == _labels[mask_onehot])


# def get_perr(preds, labels):
#     print(f"Metrice get perr")
#     preds = np.argmax(preds, -1)
#     mask = labels != -100
#     p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, labels, mask))).mean()
#     return p_err