from transformers import BertTokenizer

from fruitify.configs import BERT_MODEL
from fruitify.vocab import build_word2subs


def main():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    word2sub = build_word2subs(tokenizer, 3)
    print(word2sub)  # the masks.


if __name__ == '__main__':
    main()