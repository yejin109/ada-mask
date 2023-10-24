import time
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import functools
import operator


def get_dataset(args, tokenizer, logger=None):
    if args.data == 'wikipedia':
        dataset = load_dataset(args.data, '20220301.en', cache_dir=os.environ['CACHE_DIR'], split='train')
        dataset = dataset.remove_columns(["id", 'title', 'url'])
    elif args.data == 'bookcorpus':
        dataset = load_dataset(args.data, cache_dir=os.environ['CACHE_DIR'], split='train')
    elif args.data =='all':
        bookcorpus = load_dataset("bookcorpus", cache_dir=os.environ['CACHE_DIR'], split='train')
        wiki = load_dataset("wikipedia", "20220301.en", cache_dir=os.environ['CACHE_DIR'], split='train')
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
        dataset = concatenate_datasets([bookcorpus, wiki])
        if logger is not None:
            logger.info(f'bookcorpus raw data \n{bookcorpus}')
            logger.info(f'wikipedia raw data \n{wiki}')
            logger.info(f'concat raw data \n{dataset}')
    else:
        raise KeyError('Wrong datset')

    # Original
    # logger.info('Tokenize function started')
    # start = time.time()
    # tokenized_datasets = dataset.map(tokenize_function,
    #                                  batched=True, remove_columns=list(dataset.features.keys()),
    #                                  fn_kwargs={'_tokenizer': tokenizer}, 
    #                                  load_from_cache_file=True,
    #                                  desc='Tokenize function')
    # token_time = time.time() - start
    # logger.info('Tokenize function Done')

    # logger.info('Group text function started')
    # start = time.time()
    # lm_datasets = tokenized_datasets.map(group_texts,
    #                                      batched=True,
    #                                      fn_kwargs={'_chunk_size': args.max_seq_length}, 
    #                                      load_from_cache_file=True,
    #                                      desc='Group Text')
    # lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # group_time = time.time() - start
    # logger.info('Group text function Done')

    # Optimized
    logger.info('MLM process function started')
    start = time.time()
    lm_datasets = dataset.map(mlm_process,
                              batched=True,
                              fn_kwargs={'ctx_len': args.max_seq_length, 'tokenizer': tokenizer}, 
                            #   load_from_cache_file=False,
                              load_from_cache_file=True,
                              remove_columns=['text'],
                              desc='MLM Preprocess')
    # print(lm_datasets.column_names)
    # print('labels' not in lm_datasets.column_names)
    # print('label' in lm_datasets.column_names)
    # if ('labels' not in lm_datasets.column_names) and ('label' in lm_datasets.column_names):
    #     lm_datasets = lm_datasets.rename_column("label", "labels")
    lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    mlm_time = time.time() - start
    logger.info('MLM process function Done')

    # logger.info('Flatten function started')
    # start = time.time()
    # lm_datasets = dataset.map(flatten,
    #                           batched=True,
    #                           load_from_cache_file=True,
    #                           desc='Flatten')
    # flatten_time = time.time() - start
    # logger.info('Flatten function Done')

    dataset = lm_datasets

    logger.info('Split started')
    start = time.time()
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.train_test_split(args.train_test_split)
    # dataset['test'] = dataset['test'].shuffle(seed=args.seed).select(list(range(500)))
    split_time = time.time() - start
    logger.info('Split Done')
    if logger is not None:
        # logger.debug(f'tokenization took {token_time}')
        logger.debug(f'grouping took {mlm_time}')
        logger.debug(f'split took {split_time}')

    return dataset


def tokenize_function(examples, _tokenizer=None):
    result = _tokenizer(examples["text"])
    if _tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        # result["word_ids"] = list(map(lambda i: result.word_ids(i), range(len(result['input_ids']))))
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
    # result = {
    #     k: np.split(np.array(t)[:total_length], _chunk_size)[:-1].tolist()
    #     for k, t in concatenated_examples.items()
    # }

    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def mlm_process(raw_inps, tokenizer, ctx_len):
    tokenized = tokenizer(raw_inps["text"])    

    all_input_ids = functools.reduce(operator.iconcat, tokenized["input_ids"], [])
    all_attn_masks = functools.reduce(operator.iconcat, tokenized["attention_mask"], [])

    total_length = max((len(all_input_ids) // ctx_len) * ctx_len, 1)
    # total_length = (len(all_input_ids) // ctx_len) * ctx_len
    
    # all_input_ids = np.array(all_input_ids)
    # all_attn_masks = np.array(all_attn_masks)
    
    # result = {
    #     'input_ids' : np.split(all_input_ids[:total_length], ctx_len)[:-1],
    #     'attention_mask' : np.split(all_attn_masks[:total_length], ctx_len)[:-1],
    # }

    result = {
        'input_ids' : list(map(lambda i: all_input_ids[i: i+ctx_len], range(0, total_length, ctx_len))),
        'attention_mask' : list(map(lambda i: all_attn_masks[i: i+ctx_len], range(0, total_length, ctx_len))),
    }

    # result['labels'] = result['input_ids'].copy()
    return result


def flatten(inps):
    result = {
        k: functools.reduce(operator.iconcat, v, [])
        for k, v in inps.items()
    }
    return result