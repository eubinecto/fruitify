from transformers import BertTokenizer

# you can just use special tokens in the sentences:
BATCH = [
    "[CLS] I understand Tesla's vision. [SEP] Haha, that's a nice [MASK].",  # pun
    "[CLS] [MASK] are monkey's favorite fruit."  # bananas
]


def main():
    # encode the batch into input_ids, token_type_ids and attention_mask
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(BATCH,
                        # return as pytorch tensors
                        return_tensors="pt",
                        # set this to True to have all sentences in an equal length
                        padding=True)
    print(encoded['input_ids'])  # the id of each subword
    print(encoded['token_type_ids'])  # 0 = first sentence, 1 = second sentence (used for NSP)
    print(encoded['attention_mask'])  # 0 = do not compute attentions for (e.g. auto-regressive decoding)
    print(encoded['input_ids'].size())


if __name__ == '__main__':
    main()
