# Fruitify

## objective
### Monolingual Dictionary
Given a description of a fruit, have an English BERT predict the fruits that best matches with the description (out of apple, banana, orange, grape and strawberry).
- e.g.1: a red fruit of round shape -> apple / strawberry / orange / grape / banana
- e.g.2: a yellow fruit of round shape  -> apple / strawberry / orange / grape / banana


### Unaligned Cross-lingual Dictionary
Given a description of a fruit in Korean, Have an mBERT predict the fruits in English that best matches with the Korean description.
- e.g.1: 동그랗고 빨간 과일 -> apple / strawberry / orange / grape / banana
- e.g.2: 동그랗고 노란 과일 -> apple / strawberry / orange / grape / banana

Note that we attempt to do so with exactly the same training dataset as is used for the monolingual one. This is to explore to what degree mBERT can compensate for unaligned data. 


## Implementation
We follow the architecture as used in *BERT for Monolingual and Cross-Lingual Reverse Dictionary*(Yan et al., 2020)
