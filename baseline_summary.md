# summary

- train_reg.json: include hand-craft qa pairs from the UIT regulation documents
- train_wiki.json: include hand-craft qa pairs from the Vietnamese Wikipedia.
- train.json: include the combined train_reg.json and train_wiki.json
- train_translated_squad.json: include the translated training SQuAD dataset from English to Vietnamese
- train_translated_squad_25.json: include 25% best translation of the translated SQuAD
- train_translated_squad_50.json: include 50% best translation of the translated SQuAD
- train_translated_squad_75.json: include 75% best translation of the translated SQuAD
- train_translated_squad_100.json: include 100% best translation of the translated SQuAD

```
segment_documents:
max_length=6
stride=3
```

## Base line

```
train_reg
number of docs: 75
number of segments: 122
number of questions: 626
Exact Match: 0.08146964856230032 (51)
F1:  0.40623676953403537
```

```
train_wiki
number of docs: 67
number of segments: 115
number of questions: 599
Exact Match: 0.18864774624373956 (113)
F1:  0.46438375218148154
```

```
train_translated_squad_25
number of docs: 19067
number of segments: 45052
number of questions: 20958
Exact Match: 0.19352991697681077 (4056)
F1:  0.302906754051067
```

## Rerank answer

Test all w0,w1 pair in range (0,1,0.001) and find the best pair

```
w0 * retrieval_score + w1 * predict_score
w0 + w1= 1
```

```
train_reg
number of docs: 75
number of segments: 122
number of questions: 626
best_w0: 0.833
best_w1: 0.167
best_exact_match: 60
best_f1: 0.44819872688227713
```

```
train_wiki
number of docs: 67
number of segments: 115
number of questions: 599
best_w0: 0.048
best_w1: 0.9520000000000001
best_exact_match: 139
best_f1: 0.5212223746067531
```

```
train_translated_squad_25
number of docs: 19067
number of segments: 45052
number of questions: 20958
best_w0: 0.387
best_w1: 0.613
best_exact_match: 6078
best_f1: 0.4407442486094088
```
