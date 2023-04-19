#!/usr/bin/env python
# coding=utf-8
import torch
import argparse
from itertools import chain
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset

torch.manual_seed(0)
SEED = 42


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--directory", type=str, required=True, help="A path to save model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="A path for initial model.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Size of the batch.")
    parser.add_argument("--token_length", type=int, default=256,
        help="Size of token sequence.")
    parser.add_argument("--train_file", type=str, default="data/process.train.json",
        help="A json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default="data/process.valid.json",
        help="A json file containing the validation data.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
        help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
        help="Weight decay to use.")
    parser.add_argument("--max_steps", type=int, default=100,
        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args(raw_args)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("json", data_files={
        "train": "data/tripadvisor/train/noencoded.json",
        "valid": "data/tripadvisor/valid/noencoded.json"
    })

    datasets = datasets.shuffle(seed=SEED)
    datasets["valid"] = datasets["valid"].select(range(5000))

    def tokenizer_function(examples):
        output = tokenizer(examples["text"])
        return output

    column_names = datasets["train"].column_names

    print("Tokenizing data.")
    datasets = datasets.map(
        tokenizer_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.token_length:
            total_length = (total_length // args.token_length) * args.token_length
        res = {
            k: [t[i : i + args.token_length] for i in range(0, total_length, args.token_length)]
            for k, t in concatenated_examples.items()
        }
        res['labels'] = res['input_ids'].copy()
        return res

    print("Grouping data.")
    datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=8,
    )

    training_args = TrainingArguments(
        f"{args.directory}",
        run_name=f"{args.directory}",
        evaluation_strategy="steps",
        logging_steps=100,
        eval_steps=500,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.num_warmup_steps,
        max_steps=args.max_steps,
        report_to="mlflow",
        load_best_model_at_end=True,
        save_total_limit=5,
        # gradient_checkpointing=True,
        fp16=True,
        optim="adafactor",  # TODO verify diff
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
