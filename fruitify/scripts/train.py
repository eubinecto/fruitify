from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from fruitify.datasets import MonoENFruit2Def, CrossFruit2Def
from fruitify.loaders import load_fruitify_dataset
from fruitify.configs import BERT_MODEL, MBERT_MODEL
from fruitify.models import ReverseDict
from fruitify.paths import DATA_DIR
import pytorch_lightning as pl
import torch
import argparse
from fruitify.vocab import build_word2subs


def main():
    # --- arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd_mode", type=str,
                        default="mono_en")
    parser.add_argument("--k", type=int,
                        default=3)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--max_epochs", type=int,
                        default=20)
    parser.add_argument("--batch_size", type=int,
                        default=40)
    parser.add_argument("--repeat", type=int,
                        default=10)

    args = parser.parse_args()
    rd_mode: str = args.rd_mode
    k: int = args.k
    lr: float = args.lr
    max_epochs: int = args.max_epochs
    batch_size: int = args.batch_size
    repeat: int = args.repeat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- instantiate the models & the data --- #
    fruitify_dataset = load_fruitify_dataset()

    if rd_mode == "mono_en":
        bert_mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k, rd_mode).to(device)
        rd = ReverseDict(bert_mlm, word2subs, k, lr)  # mono rd
        model_name = "mono_en_{epoch:02d}_{train_loss:.2f}"
        dataset = MonoENFruit2Def(fruitify_dataset, tokenizer, k)  # just use everything for training

    elif rd_mode == "cross":
        # based off of pre-trained multilingual bert
        mbert_mlm = BertForMaskedLM.from_pretrained(MBERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
        word2subs = build_word2subs(tokenizer, k, rd_mode).to(device)
        rd = ReverseDict(mbert_mlm, word2subs, k, lr)  # cross rd
        model_name = "cross_{epoch:02d}_{train_loss:.2f}"
        dataset = CrossFruit2Def(fruitify_dataset, tokenizer, k)
    else:
        raise ValueError
    # --- make sure to load rd to whatever device you are working on --- #
    rd.to(device)
    # --- load the data --- #
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
