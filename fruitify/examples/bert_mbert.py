from transformers import BertTokenizer
from fruitify.configs import MBERT_MODEL, BERT_MODEL


def main():
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    mbert_tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
    print(bert_tokenizer.vocab_size)  # monolingual - english only
    print(mbert_tokenizer.vocab_size)  # multilingual - including korean, english, etc..


if __name__ == '__main__':
    main()
