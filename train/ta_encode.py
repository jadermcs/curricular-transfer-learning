#!/usr/bin/env python
# coding=utf-8
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

torch.manual_seed(0)

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--directory", type=str, required=True, help="A path to save model.")
    parser.add_argument("--checkpoint", type=str, required=True,  help="A path for initial model.")
    parser.add_argument("--subset", type=str, help="The subset of multiwoz to train.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Size of the batch.")
    parser.add_argument("--train_file", type=str, default="data/process.train.json",
        help="A json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default="data/process.valid.json",
        help="A json file containing the validation data.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
        help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=None,
        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.", choices=["linear", "cosine",
        "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args(raw_args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)

    datasets = load_dataset("parquet", data_files={"train": train_files,
                                                "validation": validation_files},
                            cache_dir="proc_data")
    print("Filtering data.")
    datasets = datasets.filter(
        lambda x: x['text'].endswith((".", "?", "!")) and\
        x['reply'].endswith((".", "?", "!")))

    special_tokens = ["<sos_u>", "<eos_u>",
                    "<sos_b>", "<eos_b>",
                    "<sos_r>", "<eos_r>"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained("models/forums_tokenizer")

    column_names = datasets["train"].column_names

    def tokenizer_function(examples):
        return tokenizer(examples["encoded"])

    column_names = datasets["train"].column_names

    print("Tokenizing data.")
    datasets = datasets.map(
        tokenizer_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
    )

    training_args = TrainingArguments(
        f"{args.checkpoint}-mwozsub",
        run_name=f"{args.checkpoint}-mwozsub",
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=2000,
        num_train_epochs=100,
        report_to="wandb",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    trainer.train()

if __name__ == "__main__":
    main()