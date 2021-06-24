"""
load a pre-trained fruitify, and play with it.
"""
import argparse
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from fruitify.models import ReverseDict, Fruitifier
from fruitify.paths import MONO_EN_CKPT, CROSS_CKPT
from fruitify.configs import BERT_MODEL
from fruitify.vocab import build_word2subs, VOCAB_MONO_EN, VOCAB_CROSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd_mode", type=str,
                        default="mono_en")
    parser.add_argument("--desc", type=str,
                        default="The fruit that monkeys love")
    args = parser.parse_args()
    rd_mode: str = args.rd_mode
    desc: str = args.desc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rd_mode == "mono_en":
        config = BertConfig()
        bert_mlm = BertForMaskedLM(config)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=3, mode=rd_mode)  # this is something I don't really like...
        rd = ReverseDict.load_from_checkpoint(MONO_EN_CKPT, bert_mlm=bert_mlm, word2subs=word2subs)
        rd.eval()  # this is necessary
        rd = rd.to(device)
        vocab = VOCAB_MONO_EN
    elif rd_mode == "cross":
        config = BertConfig()
        bert_mlm = BertForMaskedLM(config)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=10, mode=rd_mode)  # this is something I don't really like...
        rd = ReverseDict.load_from_checkpoint(CROSS_CKPT, bert_mlm=bert_mlm, word2subs=word2subs)
        rd.eval()  # this is necessary
        rd = rd.to(device)
        vocab = VOCAB_CROSS
    else:
        raise ValueError
    fruitifier = Fruitifier(rd, tokenizer, vocab)
    print("### desc: {} ###".format(desc))
    for results in fruitifier.fruitify(descriptions=[desc]):
        for idx, res in enumerate(results):
            print("{}:".format(idx), res)


if __name__ == '__main__':
    main()
