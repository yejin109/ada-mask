import torch
import numpy as np
import json
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--target_1', type=str)
parser.add_argument('--target_2', type=str)


def load_files(_fname):
    with open(f'{output_dir}/{_fname}.json', 'r') as f:
        _log = json.load(f)

    _repre = torch.load(f'{output_dir}/{_fname}.pt')

    return _log, _repre


def measure_repre(_reps1, _reps2):
    norms1 = []
    norms2 = []
    normsdiff = []
    coses = []
    noise_coses = []
    for _rep1, _rep2 in zip(_reps1, _reps2):
        # Norm
        for _token1, _token2 in zip(_rep1[0], _rep2[0]):
            noise = _token1 - _token2
            # print(noise.size(), _token1.size(), _token2.size())
            _norm1 = torch.norm(_token1)
            _norm2 = torch.norm(_token2)

            _norm_diff = torch.norm(noise)
            
            # Cosine sim
            cos = (_token1 * _token2).sum()/(_norm1*_norm2)
            noise_cos = F.cosine_similarity(noise, _token2, dim=0)
            # noise_cos = ((_token1 - _token2) * _token2).sum()/(_norm_diff*_norm2)

            norms1.append(_norm1.item())
            norms2.append(_norm2.item())
            normsdiff.append(_norm_diff.item())
            coses.append(cos.item())
            noise_coses.append(noise_cos.item())
    norms1 = np.mean(norms1)
    norms2 = np.mean(norms2)
    normsdiff = np.mean(normsdiff)
    coses = np.mean(coses)
    noise_coses = np.mean(noise_coses)

    return norms1, norms2, normsdiff, coses, noise_coses

if __name__ == '__main__':
    args = parser.parse_args()
    output_dir = '/home/ay0119/bert-tiny-main/results/noise_test'

    log_1, repre_1 = load_files(args.target_1)
    log_2, repre_2 = load_files(args.target_2)

    measures = measure_repre(repre_1, repre_2)
    emb_dim = repre_1[0].size(-1)
    log = {
        'target_1_grad' : log_1['grad_norm'],
        'target_2_grad' : log_2['grad_norm'],
        'norm1': measures[0],
        'norm2': measures[1],
        'normdiff': measures[2],
        'norm1_normalized': measures[0]/emb_dim,
        'norm2_normalized': measures[1]/emb_dim,
        'normdiff_normalized': measures[2]/emb_dim,
        'cos': measures[3],
        'noise_cos': measures[4]
    }
    with open(f'{output_dir}/{args.target_1.split("-")[0]}.json', 'w') as f:
        json.dump(log, f)