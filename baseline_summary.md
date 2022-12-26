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
train_reg
number of sentences: 78
number of questions: 626
Exact Match: 0.08306709265175719 (52)
F1:  0.40422857987548627
```

```
train_wiki
number of sentences: 67
number of questions: 599
Exact Match: 0.18363939899833054 (110)
F1:  0.47736904731553376
```

```
train_translated_squad_25
number of sentences: 20043
number of questions: 20958
Exact Match: 0.18541845595953813 (3886)
F1:  0.2930213403013636
```
