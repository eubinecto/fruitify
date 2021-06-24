from transformers import BertTokenizer
from fruitify.configs import MBERT_MODEL
from fruitify.vocab import build_word2subs


def main():

    # use the mbert's tokenizer
    tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
    word2sub = build_word2subs(tokenizer, 10, rd_type="cross")
    print(word2sub)  # the masks.


if __name__ == '__main__':
    main()
