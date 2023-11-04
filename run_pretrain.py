import logging
import datetime
import os
import argparse

import torch.nn as nn
import torch
from datasets import load_dataset
from collator import CustomMLMCollator
from callbacks import CustomWandbCallback, AscMaskCallBack, AdaMaskCallBack, StepMaskCallBack, CosineMaskCallBack
import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score
from trainer import CustomTrainer
import wandb
import data_process
import metrics


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--device', default='cuda:4', type=str)
parser.add_argument('--cache_dir', required=True)

parser.add_argument('--model_type', default="google/bert_uncased_L-2_H-128_A-2", type=str)
parser.add_argument('--ckpt', default=None, type=str)

parser.add_argument('--data', default='bookcorpus', required=False, help='default bookcorpus, wikipedia', type=str)
parser.add_argument('--train_test_split', default=0.001, type=float)
parser.add_argument('--max_seq_length', default=512, type=int)
# parser.add_argument('--data_type', default='huggingface')

# parser.add_argument('--use_partial_data', default=False, required=False)
# parser.add_argument('--partial_data_size', default=4, type=int, required=False)
# parser.add_argument('--split', default=False, required=False,)

# train
parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate or warmup peak')
parser.add_argument('--lr_scheduler', default='linear', type=str)
parser.add_argument('--warmup_steps', default=10000, type=int)
# parser.add_argument('--warmup_peak', default=6e-4, type=float)

parser.add_argument('--adam_beta1', default=0.9, type=float)
parser.add_argument('--adam_beta2', default=0.999, type=float)
parser.add_argument('--adam_eps', default=1e-8, type=float)

parser.add_argument('--wd', default=1e-2, type=float)

parser.add_argument('--dropout', default=1e-1, type=float)

parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--b_train', default=128, type=int)
parser.add_argument('--max_steps', type=int, default=-1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

# Adaptive Masking
parser.add_argument('--p', default=.15, type=float)
parser.add_argument('--mask_tolerance', default=0.01, required=False, type=float)
parser.add_argument('--mask_increment', default=0.005, required=False, type=float)

parser.add_argument('--ada_token', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--ada_memo', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--cosine', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--step', default=False, required=False, action=argparse.BooleanOptionalAction)

# parser.add_argument('--entropy', default=False, required=False, action=argparse.BooleanOptionalAction)
# parser.add_argument('--mrd', default=False, required=False, action=argparse.BooleanOptionalAction)

# Validation
# parser.add_argument('--b_eval', default=256, type=int)
# parser.add_argument('--shard_eval', default=1000, type=int)

# Test
parser.add_argument('--train', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--test', default=False, required=False, action=argparse.BooleanOptionalAction)

# Log
parser.add_argument('--logging_steps', type=int, default=2000)
parser.add_argument('--save_steps', type=int, default=20000)

# Distributed
parser.add_argument("--local-rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument('--world_size', type=int, default=0)

def get_method_name():
    if args.ada_memo:
        return 'AdaMemo'
    elif args.ada_token:
        return 'AdaToken'
    elif args.cosine:
        return 'Cosine'
    elif args.step:
        return 'Step'
    else:
        return 'Const'

def train(_model, _dataset, _train_args, sharding_size=600):
    _run_name = [args.model_type, os.environ['MASKING_P'], str(args.seed)]
    # if args.ada_memo:
    #     _run_name.insert(1, 'AdaMemo')
    # elif args.ada_token:
    #     _run_name.insert(1, 'AdaToken')
    # # elif args.mrd:
    # #     _run_name.insert(1, 'MRD')
    # # elif args.entropy:
    # #     _run_name.insert(1, 'Entropy')
    # elif args.cosine:
    #     _run_name.insert(1, 'Cosine')
    # elif args.step:
    #     _run_name.insert(1, 'Step')
    # else:
    #     _run_name.insert(1, 'Const')
    _run_name.insert(1, get_method_name())
    _run_name = '-'.join(_run_name)
    logger.info(f"Run Name : {_run_name}")
    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps",  # candidates : steps, epochs, no
        run_name=_run_name,
        # include_inputs_for_metrics=True,
        include_inputs_for_metrics=False,
        **_train_args,
    )

    trainer = CustomTrainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        # eval_dataset=_dataset["test"].shard(num_shards=sharding_size, index=np.random.randint(0, sharding_size, size=1)),
        eval_dataset=_dataset["test"],
        data_collator=mlm_collator,
        compute_metrics=compute_metrics,
    )
    trainer.is_model_parallel=False

    trainer.add_callback(CustomWandbCallback)
    # if args.mrd:
    #     trainer.add_callback(AscMaskCallBack)
    if args.ada_token or args.ada_memo:
        trainer.add_callback(AdaMaskCallBack)
    if args.cosine:
        trainer.add_callback(CosineMaskCallBack)
    if args.step:
        trainer.add_callback(StepMaskCallBack)

    if args.test:
        eval_res = trainer.evaluate()
        logger.info(f"Evaludation : {eval_res['eval_loss']}")
    else:
        trainer.train()
        trainer.evaluate()

    trainer.save_model()


def write_env_var(_name, val):
    if _name not in os.environ.keys():
        os.environ[_name] = str(val)


# def get_token_acc(preds, labels):
#     preds = np.argmax(preds, -1)

#     preds = preds.reshape(-1)
#     labels = labels.reshape(-1)

#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]

#     acc_token = accuracy_score(labels, preds)

#     return acc_token


# def get_memorization(_data: torch.Tensor, _model, mask_token_id):
#     if ~ isinstance(_data, torch.Tensor):
#         _data = torch.tensor(_data).long()
#         _data = _data.to(_model.device)

#     _inps = _data.clone()
#     _batch_size = _inps.size(0)
#     _chunk_size = _inps.size(1)

#     mask = np.random.choice(np.arange(_chunk_size).astype(int), size=(_batch_size, ))
#     mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to(_model.device), num_classes=_chunk_size).bool()

#     _labels = _inps.clone()
#     _labels[~mask_onehot] = -100

#     _inps[mask_onehot] = mask_token_id

#     with torch.no_grad():
#         _output = _model(_inps)
#     _logits = _output['logits'].detach().cpu().numpy()
#     _preds = _logits.argmax(-1)

#     _labels = _labels.detach().cpu().numpy()
#     mask_onehot = mask_onehot.detach().cpu().numpy()
#     return np.mean(_preds[mask_onehot] == _labels[mask_onehot])


def compute_metrics(eval_preds, _model, eps=1e-6):
    # preds, labels, inps = eval_preds   
    # print(preds)
    # print(labels)
    # print(inps)
    # write_env_var('EVAL_CNT', str(0))
    # Memorize
    # preds, labels, inps = eval_preds    
    # with torch.no_grad():
    #     _memo = metrics.get_memorization(inps, _model,
    #                           mlm_collator.tokenizer.convert_tokens_to_ids(mlm_collator.tokenizer.mask_token))

    # Entropy
    # preds, labels, inps = eval_preds
    # entropy = metrics.get_entropy(preds, labels)
    # mask = labels != -100
    # log_probs = np.log(np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True))
    # entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    # entropy = entropy[mask].mean(axis=-1)

    # # RankMe
    # preds, labels, inps = eval_preds
    # with torch.no_grad():
    #     _model.eval()
    #     embs = _model.bert.embeddings(torch.Tensor(inps).to('cuda:0').long())
    #     embs = embs.cpu().detach().numpy()
    # _, singular_vals, _ = np.linalg.svd(embs)
    #
    # p = singular_vals / singular_vals.sum(axis=1, keepdims=True) + eps
    # rankme = np.exp((- p * np.log(p)).sum(axis=1)).mean()

    # P_err
    # preds, labels, inps = eval_preds
    # p_err = metrics.get_perr(preds, labels)
    # preds = np.argmax(preds, -1)
    # mask = labels != -100
    # p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, labels, mask))).mean()

    # Token_acc
    # preds, labels, inps = eval_preds
    preds, labels = eval_preds
    acc_token = metrics.get_token_acc(preds, labels)
    preds = None
    labels = None
    # Metric
    # _p = float(os.environ['MASKING_P'])
    # metric_cur = acc_token / (1 - _p)

    # RE-evaluation
    # with torch.no_grad():
    #     _model.eval()
    #     preds, labels, inps = eval_preds
    #     org_acc = metrics.get_token_acc_fixed(labels, inps, _model, mlm_collator, args.p)
        # org_ids = np.zeros_like(inps)
        # mask = labels != -100
        # org_ids[mask] = labels[mask]
        # org_ids[~mask] = inps[~mask]

        # org_ids, org_labels = mlm_collator.masking_tokens(torch.Tensor(org_ids).to(_model.device).long(), args.p)
        # org_logits = _model(org_ids)['logits'].detach().cpu().numpy()
        # org_acc = get_token_acc(org_logits, org_labels.detach().cpu().numpy())
        # org_labels = None
        # org_logits = None
        # org_ids = None

    write_env_var('P_TICKER', 'STAY')
    # write_env_var('TOKEN_ACC_ORG', str(org_acc))
    write_env_var('TOKEN_ACC', str(acc_token))
    # write_env_var('P_METRIC', str(metric_cur))
    # write_env_var('ENTROPY', str(entropy))
    # write_env_var('P_CNT', str(0))
    # write_env_var('MEMORIZATION', str(_memo))

    if args.ada_token:
        # trial 0
        # _tolerance = 0.01
        # trial 1 
        # _tolerance = 0.05
        _tolerance = args.mask_tolerance

        # _p_cnt = int(os.environ['P_CNT'])
        # V6 : use Token acc
        # current_acc = org_acc
        # past_acc = float(os.environ['TOKEN_ACC_ORG'])
        current_acc = acc_token
        past_acc = float(os.environ['TOKEN_ACC'])
        # V11 
        if current_acc > past_acc + _tolerance:
            os.environ['P_TICKER'] = 'UP'
        elif current_acc < past_acc - _tolerance:
            os.environ['P_TICKER'] = 'DOWN'
        else:
            os.environ['P_TICKER'] = 'STAY'
        # if current_acc > past_acc + _tolerance:
        #     os.environ['P_TICKER'] = 'UP'
        # elif current_acc < past_acc:
        #     os.environ['P_TICKER'] = 'DOWN'
        # else:
        #     os.environ['P_TICKER'] = 'STAY'

    #     current_acc = acc_token
    #     if current_acc > 0.3:
    #         os.environ['P_TICKER'] = 'UP'
    #     elif current_acc < 0.3:
    #         os.environ['P_TICKER'] = 'DOWN'
    #     else:
    #         os.environ['P_TICKER'] = 'STAY'
    # if args.ada_memo:
    #     _tolerance = 0.05

    #     # V7 : use memorization
    #     current_acc = _memo
    #     past_acc = float(os.environ['MEMORIZATION'])
    #     if current_acc > past_acc + _tolerance:
    #         os.environ['P_TICKER'] = 'UP'
    #     elif current_acc < past_acc - _tolerance:
    #         os.environ['P_TICKER'] = 'DOWN'
    #     else:
    #         os.environ['P_TICKER'] = 'STAY'
    # os.environ['EVAL_CNT'] = str(int(os.environ['EVAL_CNT'])+1)
    os.environ['TOKEN_ACC'] = str(acc_token)
    # os.environ['TOKEN_ACC_ORG'] = str(org_acc)
    # os.environ['MEMORIZATION'] = str(_memo)

    eval_res = {
        # 'P_err': p_err,
        'Token_acc': acc_token,
        # 'Metric': metric_cur,
        # 'Memorization': _memo,
        # 'RankMe': rankme,
        # 'Entropy': entropy,
        # 'Token_acc_org': org_acc,
        }
    torch.cuda.empty_cache()
    return eval_res


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    os.environ['CACHE_DIR'] = args.cache_dir

    os.environ['MASKING_P'] = str(args.p)
    os.environ['MASKING_P_INIT'] = str(args.p)
    os.environ['MASKING_INCRE'] = str(args.mask_increment)

    os.environ['LOGGING_STEP'] = str(args.logging_steps)
    os.environ['WANDB_PROJECT'] = args.data + ' - v10'
    os.environ['WANDB_WATCH'] = 'all'
    
    # WandB config
    for k, v in os.environ.items():
        os.environ[f'WANDB_{k}'] = str(v)

    os.environ['ITERATION_STEP'] = str(0)
    os.environ['EXP_NAME'] = '-'.join(
        ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

    os.environ['LOG_DIR'] = '/'.join(['/data1/ay0119/bert-tiny-main', args.data, args.model_type, '-'.join([ str(datetime.datetime.now().strftime("%y%m%d-%H%M")), get_method_name(), str(args.p), str(args.seed)])])
    os.makedirs(os.environ['LOG_DIR'], exist_ok=True)
    os.makedirs(os.path.join(os.environ['LOG_DIR'], 'batch'), exist_ok=True)

    logger = logging.getLogger('pretrain')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s, %(levelname)s] : %(message)s")
    handler = logging.FileHandler(f'{os.getenv("LOG_DIR")}/_debug.log')
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    logger.info(f'Cache at {os.getenv("CACHE_DIR")}')
    logger.info(f'Log Dir at {os.getenv("LOG_DIR")}')

    np.random.seed(args.seed)

    # _model_path = args.ckpt if args.ckpt is not None else args.model_type
    # logger.info(f'Model Path : {_model_path}')
    # V6 : Randomly initialize model
    if args.ckpt is not None:
        logger.info(f'Model from pretrained : {args.ckpt}')
        model = AutoModelForMaskedLM.from_pretrained(args.ckpt)
    else:
        logger.info(f'Model from scratch : {args.model_type}')
        config = AutoConfig.from_pretrained(args.model_type)
        model = AutoModelForMaskedLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    train_args = {'learning_rate': args.lr, 'lr_scheduler_type' : args.lr_scheduler, 'gradient_accumulation_steps': args.gradient_accumulation_steps,
                  'adam_beta1' : args.adam_beta1, 'adam_beta2': args.adam_beta2, 'adam_epsilon': args.adam_eps,
                  'warmup_steps' : args.warmup_steps,
                  'weight_decay': args.wd,
                  'num_train_epochs': args.epochs, 'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_train,
                  'do_eval': True, 'do_train': True,
                  'max_steps': args.max_steps, 'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

    # if args.data == 'wikipedia':
    #     dataset = load_dataset(args.data, '20220301.en', cache_dir=os.environ['CACHE_DIR'])
    #     dataset = dataset.remove_columns(["id", 'title', 'url'])
    # else:
    #     dataset = load_dataset(args.data, cache_dir=os.environ['CACHE_DIR'])

    # if args.use_partial_data:
    #     dataset['train'] = dataset['train'].shard(args.partial_data_size, index=0)
    # if 'unsupervised' in dataset.keys():
    #     del dataset['unsupervised']

    # tokenized_datasets = dataset.map(tokenize_function,
    #                                     batched=True, remove_columns=list(dataset['train'].features.keys()),
    #                                     fn_kwargs={'_tokenizer': tokenizer})
    # max_seq_length = args.max_seq_length
    # lm_datasets = tokenized_datasets.map(group_texts,
    #                                         batched=True,
    #                                         fn_kwargs={'_chunk_size': max_seq_length})
    # lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # dataset = lm_datasets
    # tokenized_datasets = None

    # if isinstance(dataset, dict) and ('test' not in dataset.keys()):
    #     dataset = dataset['train'].train_test_split(0.1)
    #     dataset['test'] = dataset['test'].train_test_split(0.01)['test']

    dataset = data_process.get_dataset(args, tokenizer, logger=logger)
    mlm_collator = CustomMLMCollator(tokenizer, args.p, logger=logger, ctx_len=args.max_seq_length)

    if args.train:
        logger.info(f'Dataset \n\t {dataset}')
        for k, v in vars(args).items():
            logger.debug(f'{k} = {v}')
        wandb.login(key='5bb26e3124589bc9e7a4d4aa19bd3ea2199e9d14')
        train(model, dataset, train_args)