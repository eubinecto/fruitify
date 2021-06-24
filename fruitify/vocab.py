from torch import Tensor
from transformers import BertTokenizer

# for getting the vocabulary ids.
VOCAB_MONO = ['apple', 'banana', 'strawberry', 'orange', 'grape', 'pineapple', 'avocado']
VOCAB_CROSS = ['apple', 'banana', 'orange', 'grape', 'pineapple', 'avocado',
               '사과', '바나나', '오렌지', '포도', '파인애플', '아보카도']  # exclude strawberry for cross.

# for getting lang_ids
LANGS = ['en', 'kr']


def build_word2subs(tokenizer: BertTokenizer, k: int, rd_type: str) -> Tensor:
    global VOCAB_MONO, VOCAB_CROSS
    if rd_type == "mono":
        vocab = VOCAB_MONO
    elif rd_type == "cross":
        vocab = VOCAB_CROSS
    else:
        raise ValueError
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    encoded = tokenizer(text=vocab,
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == pad_id] = mask_id  # replace them with masks
    return input_ids
