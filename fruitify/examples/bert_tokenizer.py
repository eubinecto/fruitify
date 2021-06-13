from transformers import BertTokenizer, AddedToken
from fruitify.configs import BERT_MODEL
from fruitify.vocab import VOCAB

# you can just use special tokens in the sentences:
BATCH = [
    ("I understand Tesla's vision.", "Haha, that's a nice [MASK]."),  # pun
    ("[MASK] are monkey's favorite fruit.", None)  # bananas,
]


def main():
    # encode the batch into input_ids, token_type_ids and attention_mask
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    lefts = [left for left, _ in BATCH]
    rights = [right for _, right in BATCH]
    encoded = tokenizer(text=lefts,
                        text_pair=rights,
                        # return them as pytorch tensors
                        return_tensors="pt",
                        add_special_tokens=True,
                        # align the lengths by padding the short ones with [PAD] tokens.
                        truncation=True,
                        padding=True)
    print(type(encoded))
    print("--- encoded ---")
    print(encoded['input_ids'])  # the id of each subword
    print(encoded['token_type_ids'])  # 0 = first sentence, 1 = second sentence (used for NSP)
    print(encoded['attention_mask'])  # 0 = do not compute attentions for (e.g. auto-regressive decoding)
    # note: positional encodings are optional; if they are not given, BertModel will automatically generate
    # one.
    print("--- decoded ---")
    decoded = [
        [tokenizer.decode(token_id) for token_id in sequence]
        for sequence in encoded['input_ids']
    ]
    for tokens in decoded:
        print(tokens)
    # the model will not attend to the padded tokens, thanks to the attention mask.

    print("---words---")
    k = 5
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    print("mask_id/pad_id:", mask_id, pad_id)

    encoded = tokenizer(text=VOCAB,
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == 0] = mask_id
    decoded = [
        [tokenizer.decode(token_id) for token_id in sequence]
        for sequence in input_ids
    ]
    for tokens in decoded:
        print(tokens)


if __name__ == '__main__':
    main()
