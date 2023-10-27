import os
import argparse

from mteb import MTEB
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling


parser = argparse.ArgumentParser()

parser.add_argument('--name_or_path', default="google/bert_uncased_L-2_H-128_A-2", type=str)
parser.add_argument('--ckpt', default='/data1/ay0119/ckpts/231021-1840-Const-0.15-91284/checkpoint-250000', type=str)
# parser.add_argument('--cache_dir', required=True)
parser.add_argument('--cache_dir', default='/data1/ay0119/mteb', type=str)


class SentenceEncoder(SentenceTransformer):
    def _load_auto_model(self, model_name_or_path):
        transformer_model = Transformer(model_name_or_path, tokenizer_name_or_path=args.name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'mean')
        return [transformer_model, pooling_model]



def main():
    model = SentenceEncoder(args.ckpt)

    evaluation = MTEB(tasks=["ArguAna"])
    # evaluation = MTEB(task_types=['Clustering', 'Retrieval']) 
    # evaluation = MTEB(task_types=['Retrieval']) 
    results = evaluation.run(model, output_folder=args.cache_dir)
    print(results)




if __name__ == '__main__':
    args = parser.parse_args()

    args.cache_dir = f'{args.cache_dir}/{"/".join(args.ckpt.split("/")[-2:])}'
    os.makedirs(args.cache_dir, exist_ok=True)

    main()
    