import os
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from data_process import tokenize_function, group_texts
import functools
import operator


# if __name__ == '__main__':
#     max_seq_length = 512
#     model_type = 'google/bert_uncased_L-2_H-128_A-2'
#     cache_dir = '/data1/ay0119/hf-cache'

#     tokenizer = AutoTokenizer.from_pretrained(model_type)

#     bookcorpus = load_dataset("bookcorpus", cache_dir=cache_dir, split='train')
#     wiki = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir, split='train')
#     wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
#     dataset = concatenate_datasets([bookcorpus, wiki])
#     print(dataset)
#     tokenized_datasets = dataset.map(tokenize_function,
#                                     batched=True, remove_columns=list(dataset.features.keys()),
#                                     fn_kwargs={'_tokenizer': tokenizer}, 
#                                     load_from_cache_file=True,
#                                     desc='Tokenize function')

#     lm_datasets = tokenized_datasets.map(group_texts,
#                                          batched=True,
#                                          fn_kwargs={'_chunk_size': max_seq_length}, 
#                                          load_from_cache_file=True,
#                                          desc='Group Text')
    
#     print(lm_datasets)
#     print(lm_datasets.select([0]))

# Optimized
def mlm_process(raw_inps, tokenizer, ctx_len):
    # print(raw_inps)
    tokenized = tokenizer(raw_inps["text"])    

    all_input_ids = functools.reduce(operator.iconcat, tokenized["input_ids"], [])
    all_attn_masks = functools.reduce(operator.iconcat, tokenized["attention_mask"], [])
    # all_input_ids = tokenized["input_ids"]
    # all_attn_masks = tokenized["attention_mask"]
    total_length = max((len(all_input_ids) // ctx_len) * ctx_len, 1)
    # total_length = (len(all_input_ids) // ctx_len) * ctx_len
    concat_ex = {
        'input_ids' : all_input_ids,
        'attention_mask' : all_attn_masks
    }
    # all_input_ids = np.array(all_input_ids)
    # all_attn_masks = np.array(all_attn_masks)
    # result = {
    #     'input_ids' : np.split(all_input_ids[:total_length], ctx_len)[:-1],
    #     'attention_mask' : np.split(all_attn_masks[:total_length], ctx_len)[:-1],
    # }
    # print(len(all_input_ids), total_length)
    result = {
        # k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        # for k, t in concat_ex.items()
        k: list(map(lambda i: all_input_ids[i: i+ctx_len], range(0, total_length, ctx_len)))
        for k in concat_ex.keys()
    }

    # result['labels'] = result['input_ids'].copy()
    return result


def flatten(inps):
    print([len(i) for i in inps["input_ids"]])
    result = {
        k: functools.reduce(operator.iconcat, v, [])
        for k, v in inps.items()
    }
    return result


if __name__ == '__main__':
    # max_seq_length = 512
    max_seq_length = 8
    model_type = 'google/bert_uncased_L-2_H-128_A-2'
    cache_dir = '/data1/ay0119/hf-cache'

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    bookcorpus = load_dataset("bookcorpus", cache_dir=cache_dir, split='train')
    wiki = load_dataset("wikipedia", "20220301.en", cache_dir=cache_dir, split='train')
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    dataset = concatenate_datasets([bookcorpus, wiki])
    print(dataset)

    # raw_inps = dataset.select([2000000, 400000])
    # ctx_len = max_seq_length

    # tokenized = tokenizer(raw_inps["text"])    
    # print('tokenized', tokenized)

    # all_input_ids = functools.reduce(operator.iconcat, tokenized["input_ids"], [])
    # all_attn_masks = functools.reduce(operator.iconcat, tokenized["attention_mask"], [])
    # # print('all_input_ids', all_input_ids)
    # # print('all_attn_masks', all_attn_masks)
    # concat_ex = {
    #     'input_ids' : all_input_ids,
    #     'attention_mask' : all_attn_masks
    # }

    # total_length = max((len(all_input_ids) // ctx_len) * ctx_len, 1)
    # print('total_length', total_length, len(all_input_ids), ctx_len)

    # # all_input_ids = np.array(all_input_ids)
    # # all_attn_masks = np.array(all_attn_masks)
    # # result = {
    # #     'input_ids' : np.split(all_input_ids[:total_length], ctx_len),
    # #     'attention_mask' : np.split(all_attn_masks[:total_length], ctx_len),
    # # }
    # result = {
    #     k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
    #     for k, t in concat_ex.items()
    # }
    # print('result.input_ids', list(map(lambda x: len(x), result['input_ids'])))
    # print('result.attention_mask', list(map(lambda x: len(x), result['attention_mask'])))


    dataset = dataset.select(list(range(10)))
    print(dataset)

    lm_datasets = dataset.map(mlm_process,
                            batched=True,
                            # batched=False,
                            fn_kwargs={'ctx_len': max_seq_length, 'tokenizer': tokenizer}, 
                            load_from_cache_file=False,
                            remove_columns=['text'],
                            desc='MLM Preprocess')
    print('Befire')
    print(lm_datasets)
    lm_datasets = lm_datasets.map(flatten,
                            batched=True,
                            # batched=False,
                            batch_size=5000,
                            load_from_cache_file=False,
                            desc='Flatten Preprocess')
    print('After')
    print(lm_datasets)
    # print(lm_datasets[0])
    # print(lm_datasets[0])
    # print({k: v for k, v in lm_datasets[0].items()})
    # print('='*100)
    # print({k: v for k, v in lm_datasets[1].items()})