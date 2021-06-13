"""
load a pre-trained fruitify, and play with it.
"""
import argparse

from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from fruitify.models import MonoLingRD, Fruitifier
from fruitify.paths import MONO_CKPT
from fruitify.configs import BERT_MODEL
from fruitify.vocab import VOCAB, build_word2subs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit_type", type=str,
                        default="mono")
    parser.add_argument("--desc", type=str,
                        default="A red fruit.")
    args = parser.parse_args()
    fruit_type: str = args.fruit_type
    desc: str = args.desc

    if fruit_type == "mono":
        config = BertConfig()
        bert_mlm = BertForMaskedLM(config)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=3)  # this is something I don't really like...
        rd = MonoLingRD.load_from_checkpoint(MONO_CKPT, bert_mlm=bert_mlm, word2subs=word2subs)
        rd.eval()  # this is necessary
    elif fruit_type == "cross":
        raise NotImplementedError
    else:
        raise ValueError

    fruits = VOCAB
    fruitifier = Fruitifier(rd, tokenizer, fruits)
    print("### desc: {} ###".format(desc))
    for results in fruitifier.fruitify(descriptions=[desc]):
        for idx, res in enumerate(results):
            print("{}:".format(idx), res)


if __name__ == '__main__':
    main()
