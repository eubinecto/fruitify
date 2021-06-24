from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from fruitify.datasets import Fruit2DefDataset
from fruitify.loaders import load_fruit2def
from fruitify.configs import BERT_MODEL, MBERT_MODEL
from fruitify.models import MonoLingRD, UnalignedCrossLingRD, UnalignedCrossLingRDBertModel
from fruitify.paths import DATA_DIR
import pytorch_lightning as pl
import torch
import argparse
from fruitify.vocab import build_word2subs


def main():
    # --- arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit_type", type=str,
                        default="mono")
    parser.add_argument("--k", type=int,
                        default=3)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--max_epochs", type=int,
                        default=20)
    parser.add_argument("--batch_size", type=str,
                        default=40)
    parser.add_argument("--repeat", type=int,
                        default=20)

    args = parser.parse_args()
    fruit_type: str = args.fruit_type
    k: int = args.k
    lr: float = args.lr
    max_epochs: int = args.max_epochs
    batch_size: int = args.batch_size
    repeat: int = args.repeat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- instantiate the models --- #

    if fruit_type == "mono":
        bert_mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k).to(device)
        rd = MonoLingRD(bert_mlm, word2subs, k, lr)
        model_name = "mono_{epoch:02d}_{train_loss:.2f}"
    elif fruit_type == "cross":
        # based off of pre-trained multilingual bert
        bert_ucl = UnalignedCrossLingRDBertModel.from_pretrained(MBERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
        word2subs = build_word2subs(tokenizer, k).to(device)
        rd = UnalignedCrossLingRD(bert_ucl, word2subs, k, lr)
        model_name = "cross_{epoch:02d}_{train_loss:.2f}"
    else:
        raise ValueError
    rd.to(device)
    # --- load the data --- #
    fruit2def = load_fruit2def()
    dataset = Fruit2DefDataset(fruit2def, tokenizer, k)  # just use everything for training
    dataset.upsample(repeat)  # just populate the batch
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True,  # this should be set to True maybe?
                            num_workers=4)

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename=model_name
    )

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR)

    # --- start training --- #
    trainer.fit(model=rd,
                train_dataloader=dataloader)


if __name__ == '__main__':
    main()
