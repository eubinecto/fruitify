
from transformers import BertForMaskedLM, BertTokenizer


# you can just use special tokens in the sentences:
BATCH = [
    "[CLS] I understand Tesla's vision. [SEP] Haha, that's a nice [MASK].",  # pun
    "[CLS] [MASK] are monkey's favorite fruit."  # bananas
]


def main():
    global BATCH
    # the pre-trained model (may take a while to download them)
    mlm = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # encode the batch into input_ids, token_type_ids and attention_mask
    encoded = tokenizer(BATCH, return_tensors="pt", padding=True)
    # mlm houses a pretrained bert model
    outputs = mlm.bert(**encoded)
    H = outputs[0]  # the hidden representation of the batch.
    print(H.size())  # (N=2, L) -> (N=2, L, Hidden=768)
    # get the hidden representations for the masked tokens
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    masked_ids = [
        input_id.tolist().index(mask_id)
        for input_id in encoded['input_ids']
    ]
    hidden_masked_1 = H[0, masked_ids[0], :]  # (N, L, 768) -> (1, 768)
    hidden_masked_2 = H[1, masked_ids[1], :]  # (N, L, 768) -> (1, 768)
    # mlm also houses the masked language model (FFN + softmax). here, the outputs are logits.
    logits_1 = mlm.cls(hidden_masked_1)  # (1, 768) -> (1, V)
    logits_2 = mlm.cls(hidden_masked_2)  # (1, 768) -> (1, V)
    print(logits_1.size(), logits_1)
    # predict the masks
    pred_1 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(logits_1.tolist())
    ]
    pred_2 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(logits_2.tolist())
    ]
    print(sorted(pred_1, key=lambda x: x[1], reverse=True)[:20])  #  will "pun" appear in the top 20's?
    print(sorted(pred_2, key=lambda x: x[1], reverse=True)[:20])  # will "bananas" appear in the top 20's?


if __name__ == '__main__':
    main()
