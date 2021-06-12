# Fruitify

## Dependencies
```bash
pip3 install transformers
pip3 install pytorch-lightning
```

## Objective
### Monolingual Reverse Dictionary
Given a description of a fruit, have an English BERT predict the fruits that best match with the description (out of apple, banana, orange, grape and strawberry).
- e.g.1: a red fruit of round shape -> apple / strawberry / orange / grape / banana
- e.g.2: a yellow fruit of round shape  -> orange / banana / apple / grape / strawberry


### Unaligned Cross-lingual Reverse Dictionary
Given a description of a fruit in Korean, have an mBERT predict the fruits in English that best match with the Korean description.
- e.g.1: 동그랗고 빨간 과일 -> apple / strawberry / orange / grape / banana
- e.g.2: 동그랗고 노란 과일 -> orange / banana / apple / grape / strawberry

Note that we attempt to do so with exactly the same training dataset as is used for the monolingual one. This is to explore to what degree mBERT can compensate for unaligned data. 


## Implementation
We follow the same architecture as what is presented in *BERT for Monolingual and Cross-Lingual Reverse Dictionary*(Yan et al., 2020)



## Examples

```
### desc: The fruit that monkeys love ###
0: ('banana', -10.440376281738281)
1: ('grape', -10.463106155395508)
2: ('strawberry', -10.712398529052734)
3: ('orange', -10.870870590209961)
4: ('apple', -11.218637466430664)
5: ('pineapple', -16.276872634887695)
```