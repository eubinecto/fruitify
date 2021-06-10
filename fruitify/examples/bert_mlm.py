
from transformers import BertForMaskedLM, BertTokenizer
from fruitify.configs import BERT_MODEL

# you can just embed the special tokens in sentences:
BATCH = [
    "I understand Tesla's vision. Haha, that's a nice [MASK].",  # pun
    "[MASK] are monkey's favorite fruit."  # bananas
]


def main():
    global BATCH
    # the pre-trained model and tokenizer (may take a while if they are have not been downloaded yet)
    # the models will be saved to ~/.cache/transformers
    mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    # encode the batch into input_ids, token_type_ids and attention_mask
    encoded = tokenizer(BATCH,
                        add_special_tokens=True,
                        return_tensors="pt",
                        truncation=True,
                        padding=True)
    # mlm houses a pretrained mbert model
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
    # mlm also houses the masked language model (just another FFN layer). here, the outputs are logits.
    logits_1 = mlm.cls(hidden_masked_1)  # (1, 768) * (768, |S|) -> (1, |S|)
    logits_2 = mlm.cls(hidden_masked_2)  # (1, 768) * (768, |S|) -> (1, |S|)
    print(logits_1.size(), logits_1)
    # decode the predictions, the logits.
    pred_1 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(logits_1.tolist())
    ]
    pred_2 = [
        (tokenizer.decode([idx]), logit)
        for idx, logit in enumerate(logits_2.tolist())
    ]
    # sort then in descending order.
    print(sorted(pred_1, key=lambda x: x[1], reverse=True)[:20])  # will "pun" appear in the top 20's?
    print(sorted(pred_2, key=lambda x: x[1], reverse=True)[:20])  # will "bananas" appear in the top 20's?


if __name__ == '__main__':
    main()
