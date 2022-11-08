
import json

import datasets
import pandas as pd
import transformers
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          Trainer, TrainingArguments, default_data_collator,
                          pipeline)
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

model_checkpoint = "deepset/roberta-base-squad2"
batch_size = 16


def main():
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(
                        token_end_index + 1)

        return tokenized_examples

    with open('../data/zac2022_train_merged_final.json', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.json_normalize(data, 'data')

    # 2 trường hợp output là có hoặc không có câu trả lời
    df = df[(df['category'] == 'FULL_ANNOTATION') |
            (df['category'] == 'FALSE_LONG_ANSWER')]
    # độ dài của câu trả lời
    df['short_candidate_length'] = df['short_candidate'].apply(
        lambda x: len(x) if type(x) == str else 0)

    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'answers': {'answer_start': [example['short_candidate_start']] if example['short_candidate_start'] != None else [],
                                                       'text': [example['short_candidate']] if example['short_candidate'] != None else []}})
    dataset = dataset.map(lambda example: {
        'context': example['text']}, batched=True, remove_columns='text')
    dataset = dataset.remove_columns(['answer', 'short_candidate_start',
                                      'short_candidate', 'short_candidate_length', 'is_long_answer', 'category'])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_length = 384
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"

    tokenized_datasets = dataset.map(
        prepare_train_features, batched=True, remove_columns=dataset["train"].column_names
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    args = TrainingArguments(
        f"model_2",

        learning_rate=2e-5,
        weight_decay=0.01,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=50,
        save_total_limit=3,

        push_to_hub=False,
        report_to="wandb",
        run_name="zalo_ai_model_2",

        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",

    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,

    )

    trainer.train()

    trainer.save_model("../model_2/saved")


if __name__ == '__main__':
    main()
