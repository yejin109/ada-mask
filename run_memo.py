import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.autograd.functional import hessian
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn.functional import one_hot

from datasets import load_dataset, concatenate_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--name_or_path', default='/data1/ay0119/ckpts/231009-200603-Const-0.15-124/checkpoint--1020000', type=str)
parser.add_argument('--model_type', default='google/bert_uncased_L-2_H-128_A-2', type=str)
parser.add_argument('--device', default='cuda:0', type=str)

parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--seed', default=124, type=int)
parser.add_argument('--b_eval', default=8, type=int)

def tokenize_function(examples, _tokenizer=None):
    result = _tokenizer(examples["text"])
    if _tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples, _chunk_size=None):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // _chunk_size) * _chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i: i + _chunk_size] for i in range(0, total_length, _chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess(_dataset, _tokenizer):
    tokenized_datasets = _dataset.map(tokenize_function,
                                     batched=True, remove_columns=list(_dataset.features.keys()),
                                     fn_kwargs={'_tokenizer': _tokenizer})
    chunk_size = min(512, _tokenizer.model_max_length)
    lm_datasets = tokenized_datasets.map(group_texts,
                                         batched=True,
                                         fn_kwargs={'_chunk_size': chunk_size})
    lm_datasets = lm_datasets
    lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return lm_datasets


def get_memorization_mlm(_data: torch.Tensor, _model, mask_token_id):
    def get_mask(_datum):
        special_token_indicator = np.array(tokenizer.get_special_tokens_mask(_datum, already_has_special_tokens=True)).astype(bool)
        ids = np.arange(len(_datum)).astype(int)
        res = np.random.choice(ids[~special_token_indicator])
        return res
    _inps = _data.clone()
    _batch_size = _inps.size(0)
    _chunk_size = _inps.size(1)

    mask = list(map(lambda x: get_mask(x), _data))
    # mask = np.random.choice(np.arange(_chunk_size).astype(int), size=(_batch_size, ))
    mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to('cuda:0'), num_classes=_chunk_size).bool()

    _labels = _inps.clone()
    _labels[~mask_onehot] = -100

    _inps[mask_onehot] = mask_token_id

    with torch.no_grad():
        _output = _model(_inps)
    _logits = _output['logits'].detach().cpu().numpy()
    _preds = _logits.argmax(-1)

    _labels = _labels.detach().cpu().numpy()
    mask_onehot = mask_onehot.detach().cpu().numpy()
    return np.mean(_preds[mask_onehot] == _labels[mask_onehot])

def evaluate(_model, _dataset):
    _model = _model.to('cuda:0')
    org_ids = _dataset['input_ids']
    _loss = 0
    _acc = 0
    _entropy = 0
    _rankme_emb = 0
    _rankme_repre = 0
    _cnt = 0
    _memo = 0

    _model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(org_ids), args.b_eval)):
            org_id = org_ids[i*args.b_eval:(i+1)*args.b_eval].to('cuda:0')
            if len(org_id) == 0:
                break
            # if args.model_type in cfgs['MLM']:
            #     _memo += get_memorization_mlm(org_id, _model, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
            # elif args.model_type in cfgs['CLM']:
            #     raise KeyError("NOT IMPLEMENTED")
            _memo += get_memorization_mlm(org_id, _model, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))

            # org_id, org_labels = mlm_collator.masking_tokens(org_id, args.p)
            #
            # org_output = _model(org_id, labels=org_labels, output_hidden_states=True)
            #
            # embs = org_output['hidden_states'][0].cpu().detach().numpy()
            # repre = org_output['hidden_states'][-1].cpu().detach().numpy()
            #
            # org_labels = org_labels.detach().cpu().numpy()
            # org_loss = org_output['loss'].detach().cpu().numpy()
            # org_logits = org_output['logits'].detach().cpu().numpy()
            #
            # _loss += org_loss.item()
            # _acc += get_token_acc(org_logits, org_labels)
            # _entropy += get_entropy(org_logits, org_labels)
            # _rankme_emb += get_rankme(embs)
            # _rankme_repre += get_rankme(repre)
            _cnt += 1

    _memo /= _cnt
    _loss /= _cnt
    _acc /= _cnt
    _entropy /= _cnt
    _rankme_emb /= _cnt
    _rankme_repre /= _cnt

    return _loss, _acc, _entropy, _rankme_emb, _rankme_repre, _memo


def print_fmt(_name, _val):
    res = f'{_name} {np.mean(_val)} +- {np.std(_val)} : {np.mean(_val): .2f}({np.std(_val): .2f})'
    print(res)
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    cache_dir = '/data1/ay0119/hf-cache'
    output_dir = '/home/ay0119/bert-tiny-main/results/memo_test'

    np.random.seed(args.seed)

    bookcorpus = load_dataset('bookcorpus', split='train', cache_dir=cache_dir)
    bookcorpus = bookcorpus.select(np.random.choice(len(bookcorpus), args.sample_size))

    wiki = load_dataset('wikipedia', '20220301.en', cache_dir=cache_dir, split='train')
    wiki = wiki.remove_columns(["id", 'title', 'url'])
    wiki = wiki.select(np.random.choice(len(wiki), args.sample_size))
    dataset = concatenate_datasets([bookcorpus, wiki])

    model = AutoModelForMaskedLM.from_pretrained(args.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    dataset = preprocess(dataset, tokenizer)

    print(dataset)
    losses = []
    acces = []
    entropies = []
    rankme_embs = []
    rankme_repreps = []
    memorizations = []
    for i in range(5):
        loss, acc, entropy, rankme_emb, rankme_repre, memorization = evaluate(model, dataset)
        memorizations.append(memorization)
        losses.append(loss)
        acces.append(acc)
        entropies.append(entropy)
        rankme_embs.append(rankme_emb)
        rankme_repreps.append(rankme_repre)

    memo_avg = np.mean(memorization)
    memo_std = np.std(memorization)
    memo_log = {
        f'{args.name_or_path}':{
            'Mean':f'{memo_avg}',
            'Std':f'{memo_std}',
            }
    }
    # try:
    #     with open(f'{output_dir}/memo.json', 'r') as f:
    #         _log = json.load(f)
    #         _log = dict(_log, **memo_log)
    # except:
    #     _log = memo_log

    # with open(f'{output_dir}/memo.json', 'w') as f:
    #     json.dump(_log, f)        
    log = print_fmt('Memorization', memorizations)
    with open(f'{output_dir}/memo.txt', 'a') as f:
        f.write(f'{args.name_or_path}\n\t{log}\n')