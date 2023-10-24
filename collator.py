import os
import torch
import numpy as np


class CustomMLMCollator:
    def __init__(self, tk, mask_prob, logger, ctx_len):
        self.ctx_len = ctx_len
        self.tokenizer = tk
        self.mask_prob = mask_prob
        self.logger = logger

    def __call__(self, inputs, use_init_p=False):
        # self.logger.info("Before collating")
        # for i, inp in enumerate(inputs):
        #     self.logger.info(f"{i}th inp")
        #     for k, v in inp.items():
        #         self.logger.info(f"{k}\n\t{v.size()}")
        # first = inputs[0]
        # agg = torch.cat(([i['input_ids'] for i in inputs]))
        # self.logger.info(f'inps keys : {first.keys()}')
        # self.logger.info(f'the first inp : \n{len(first["input_ids"])},{first["input_ids"]}\n{len(first["attention_mask"])},{first["attention_mask"]}')
        # self.logger.info(f'inp with len {len(inputs)}')
        # self.logger.info(f'inp lens : {[len(i["input_ids"]) for i in inputs]}')
        # self.logger.info(f'concat inps with size {agg.size()}\n {agg}')

        # examples = list(map(lambda x: x["input_ids"], inputs))
        # batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        # batch = {
        #         "input_ids": _torch_collate_batch(examples, self.tokenizer, ctx_len=self.ctx_len)
        #     }
        
        first = inputs[0]["input_ids"]
        agg = torch.cat([i['input_ids'] for i in inputs])
        agg = agg.reshape((-1, first.size(-1)))
        batch = {}
        if use_init_p:
            _p = self.mask_prob
        else:
            _p = float(os.environ['MASKING_P'])
        # self.logger.info("Before aggregating")
        # self.logger.info(f"{agg.size()}")

        batch['input_ids'], batch['labels'] = self.masking_tokens(agg, _p)
        # batch['input_ids'], batch['labels'] = self.masking_tokens(batch['input_ids'], _p)

        batch['attention_mask'] = torch.cat(([i['attention_mask'] for i in inputs])).reshape((-1, first.size(-1)))
        # batch['attention_mask'] = torch.cat(([i['attention_mask'] for i in inputs]))
        # self.logger.info("After collating")
        # for k, v in batch.items():
        #     self.logger.info(f"{k}\n\t{v.size()}")
        return batch

    def masking_tokens(self, inputs, mask_prob):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mask_prob)
        # self.logger.info(f"Labels : {labels.size()}")
        if self.tokenizer is not None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token) if self.tokenizer is not None else 103

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def _torch_collate_batch(examples, tokenizer, ctx_len, pad_to_multiple_of = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    if tokenizer.padding_side == "right":
        result = result[:, :ctx_len]
    else:
        result = result[:, -ctx_len:]
        # result[i, -example.shape[0] :] = example
    return result