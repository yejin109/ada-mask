import os
import argparse
import numpy as np
from tqdm import tqdm
import json

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
parser.add_argument('--device', default='cuda:4', type=str)

parser.add_argument('--sample_idx', default=100, type=int)
parser.add_argument('--seed', default=124, type=int)

parser.add_argument('--mask_num', default=1, type=int)
parser.add_argument('--mask_idx_1', default=3, type=int)
parser.add_argument('--mask_idx_2', default=None, type=int)


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
    total_length = max((total_length // _chunk_size) * _chunk_size, 1)

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


def TotalGradNorm(parameters, norm_type=2):
    _cnt = 0
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
        _cnt += 1
    total_norm = total_norm**(1. / norm_type)
    return total_norm/_cnt


def print_fmt(_name, _val):
    print(f'{_name} {np.mean(_val)} +- {np.std(_val)} : {np.mean(_val): .2f}({np.std(_val): .2f})')


def get_mask_noise(_data: torch.Tensor, _model, mask_token_id):
    def get_mask(_datum):
        special_token_indicator = np.array(tokenizer.get_special_tokens_mask(_datum, already_has_special_tokens=True)).astype(bool)
        ids = np.arange(len(_datum)).astype(int)
        res = np.random.choice(ids[~special_token_indicator], size=args.mask_num)
        return res
    _inps = _data.clone()
    
    _chunk_size = _inps.size(1)

    # mask = list(map(lambda x: get_mask(x), _data))
    # mask = np.random.choice(np.arange(_chunk_size).astype(int), size=(_batch_size, ))
    # mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to(args.device), num_classes=_chunk_size).bool()[0]
    mask_onehot = torch.zeros_like(_inps)
    mask_onehot[0, 1] = 1
    mask_onehot[0, args.mask_idx_1] = 1
    if args.mask_idx_2 is not None:
        mask_onehot[0, args.mask_idx_2] = 1
    mask_onehot = mask_onehot.bool()
    
    _labels = _inps.clone()
    _labels[~mask_onehot] = -100

    _inps[mask_onehot] = mask_token_id

    _output = _model(_inps, labels=_labels, return_dict=True)
    loss= _output.loss
    loss.backward()
    res = {
        'masked_text' : tokenizer.decode(_inps.detach().cpu().tolist()[0]),
        'hidden_states' : _output['hidden_states'],
        'grad_norm': TotalGradNorm(_model.parameters()),
    }


    return res

def evaludate(_model, _dataset):
    _model = _model.to(args.device)
    org_ids = _dataset['input_ids']

    org_text = tokenizer.decode(org_ids[0].tolist())
    print('original text : ', ''.join(org_text))

    org_ids = org_ids.to(args.device)
    outputs = get_mask_noise(org_ids, _model, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    log = {
        'original_text': org_text,
        'grad_norm': outputs['grad_norm'],
        'masked_text':outputs['masked_text']
    }

    # Logging
    fname = f'{args.sample_idx}-{args.mask_idx_1}'
    if args.mask_idx_2 is not None:
        fname += f'-{args.mask_idx_2}'
    with open(f'{output_dir}/{fname}.json', 'w') as f:
        json.dump(log, f)
    torch.save(outputs['hidden_states'], f'{output_dir}/{fname}.pt')
    print(outputs['masked_text'])


if __name__ == '__main__':
    args = parser.parse_args()
    cache_dir = '/data1/ay0119/hf-cache'
    output_dir = '/home/ay0119/bert-tiny-main/results/noise_test'

    np.random.seed(args.seed)

    bookcorpus = load_dataset('bookcorpus', split='train', cache_dir=cache_dir)
    bookcorpus = bookcorpus.select([args.sample_idx])
    # bookcorpus = bookcorpus.select([np.random.choice(len(bookcorpus), args.sample_size)])
    dataset = bookcorpus
    print(dataset['text'])
    # wiki = load_dataset('wikipedia', '20220301.en', cache_dir=cache_dir, split='train')
    # wiki = wiki.remove_columns(["id", 'title', 'url'])
    # wiki = wiki.select(np.random.choice(len(wiki), args.sample_size))
    # dataset = concatenate_datasets([bookcorpus, wiki])

    model = AutoModelForMaskedLM.from_pretrained(args.name_or_path, **{'output_hidden_states': True})
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    dataset = preprocess(dataset, tokenizer)

    print(dataset)   

    evaludate(model, dataset) 

