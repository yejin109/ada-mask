import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.autograd.functional import hessian
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.nn.functional import one_hot

from datasets import load_dataset


# name_or_path = './ckpts/231009-200603-Const-0.15-124/checkpoint--1020000'
# model_type = 'google/bert_uncased_L-2_H-128_A-2'
# p = 0.15
# # device = 'cuda:0'
# device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--name_or_path', default='./ckpts/231009-200603-Const-0.15-124/checkpoint--1020000', type=str)
parser.add_argument('--model_type', default='google/bert_uncased_L-2_H-128_A-2', type=str)
parser.add_argument('--device', default='cuda:0', type=str)

parser.add_argument('--p', default=0.15, type=float)


def masking_tokens(_inputs, mask_prob):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = _inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mask_prob)

    if tokenizer is not None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    if (masked_indices.sum() == 0).item():
        masked_indices[:, np.random.choice(masked_indices.size(-1), 1)] = True
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    _inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token) if tokenizer is not None else 103

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(_inputs.device)
    _inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return _inputs, labels


def loss_fn(_logits):
    _logits_exp = torch.exp(_logits)
    loss_batch = (_logits_exp * labels_onehot).sum(dim=-1) / _logits_exp.sum(dim=-1)
    loss = loss_batch.mean()
    return loss


if __name__ == '__main__':
    args = parser.parse_args()

    device = args.device
    name_or_path = args.name_or_path
    model_type = args.model_type
    p = args.p
    cache_dir = '/data1/ay0119/hf-cache'
    ckpt_dir = '/data1/ckpt'

    dataset = load_dataset('bookcorpus', split='train[:50]', cache_dir=cache_dir)
    texts = dataset['text']
    # texts = [
    #     'usually , he would be tearing around the living room , playing with his toys .'
    # ]

    model = AutoModelForMaskedLM.from_pretrained(name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # hessian_norms = []
    # text = [text]
    # inputs = tokenizer(texts)
    # print(inputs)
    # inputs = {k: torch.LongTensor(v) for k, v in inputs.items()}
    # inputs['input_ids'], inputs['labels'] = masking_tokens(inputs['input_ids'], p)

    # outputs = model.bert(
    #     inputs['input_ids'].to(device),
    #     attention_mask=inputs['attention_mask'].to(device),
    #     token_type_ids=inputs['token_type_ids'].to(device)
    # )

    # sequence_output = outputs[0]
    # logits = model.cls(sequence_output)

    # labels = inputs['labels'].to(device)
    # mask = labels != -100

    # logits = logits[mask][0]
    # labels = labels[mask][0]

    # labels_onehot = one_hot(labels, num_classes=logits.size(-1))

    # hessian_val = hessian(loss_fn, inputs=logits)

    # hessian_norms = []
    # for i in tqdm(range(hessian.size(0))):
    #     _hess = hessian_val[i, :, i, :]
    #     hessian_norm = torch.norm(_hess)
    #     hessian_norms.append(hessian_norm.item())
    #     torch.cuda.empt_cache()

    # print(name_or_path, np.mean(hessian_norms))

    hessian_norms = []
    for text in tqdm(texts):
        text = [text]
        inputs = tokenizer(text)
        inputs = {k: torch.LongTensor(v) for k, v in inputs.items()}
        inputs['input_ids'], inputs['labels'] = masking_tokens(inputs['input_ids'], p)

        outputs = model.bert(
            inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            token_type_ids=inputs['token_type_ids'].to(device)
        )

        sequence_output = outputs[0]
        logits = model.cls(sequence_output)

        labels = inputs['labels'].to(device)
        mask = labels != -100

        logits = logits[mask][0]
        labels = labels[mask][0]

        labels_onehot = one_hot(labels, num_classes=logits.size(-1))

        hessian_val = hessian(loss_fn, inputs=logits)
        hessian_norm = torch.norm(hessian_val)
        hessian_norms.append(hessian_norm.item())

    print(name_or_path, np.mean(hessian_norms))
