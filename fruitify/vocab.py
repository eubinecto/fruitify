from torch import Tensor
from transformers import BertTokenizer

VOCAB = ['apple', 'banana', 'strawberry', 'orange', 'grape', 'pineapple']


def build_word2subs(tokenizer: BertTokenizer, k: int) -> Tensor:
    global VOCAB
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    encoded = tokenizer(text=VOCAB,
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == pad_id] = mask_id  # replace them with masks
    return input_ids
