from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertModel
from fruitify.datasets import Fruit2DefDataset
from fruitify.loaders import load_fruit2def
from fruitify.configs import BERT_MODEL, MBERT_MODEL
from fruitify.models import MonoLingFruit, UnalignedCrossLingFruit
from fruitify.paths import SAVED_DIR
import pytorch_lightning as pl
import torch
import argparse


def main():
    # --- arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit_type", type=str,
                        default="mono")
    parser.add_argument("--k", type=int,
                        default=5)
    parser.add_argument("--max_epochs", type=int,
                        default=20)
    args = parser.parse_args()
    fruit_type: str = args.fruit_type
    k: int = args.k
    max_epochs: int = args.max_epochs

    # --- instantiate the models --- #
    if fruit_type == "mono":
        bert_mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        fruitifier = MonoLingFruit(bert_mlm, k)
    elif fruit_type == "cross":
        mbert = BertModel(MBERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
        fruitifier = UnalignedCrossLingFruit(mbert, k)
    else:
        raise ValueError
    # --- load the data --- #
    fruit2def = load_fruit2def()
    dataset = Fruit2DefDataset(fruit2def, tokenizer, k)  # just use everything for training
    dataloader = DataLoader(dataset, batch_size=10,
                            shuffle=False, num_workers=1)
    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=max_epochs,
                         default_root_dir=SAVED_DIR)

    # --- start training --- #
    trainer.fit(model=fruitifier,
                train_dataloader=dataloader)


if __name__ == '__main__':
    main()
