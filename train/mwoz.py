#!/usr/bin/env python
# coding=utf-8
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

torch.manual_seed(0)
SEED = 42


def get_special_tokens():
    base = ["<sos_u>", "<eos_u>", "<sos_b>", "<eos_b>",
            "<sos_a>", "<eos_a>", "<sos_r>", "<eos_r>"]
    with open("data/multiwoz/tokens.txt") as fin:
        data = fin.readlines()
    for token in data:
        base.append(token.strip())
    return base


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--directory", type=str, required=True, help="A path to save model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="A path for initial model.")
    parser.add_argument("--percent", type=int, default=100, help="The subset of multiwoz to train.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Size of the batch.")
    parser.add_argument("--token_length", type=int, default=256,
        help="Size of token sequence.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
        help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100,
        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args(raw_args)

    datasets = load_dataset("json", data_files={
        "train": "data/multiwoz/train/encoded.json",
        "valid": "data/multiwoz/dev/encoded.json",
        "test": "data/multiwoz/test/encoded.json",
    })

    datasets = datasets.shuffle(seed=SEED)
    if args.percent:
        size = (len(datasets["train"]) * args.percent) // 100
        datasets["train"] = datasets["train"].select(range(size))

    special_tokens = get_special_tokens()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if args.checkpoint.startswith("models"):
        model = PeftModel(args.checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16,
            lora_alpha=16, lora_dropout=0.1, bias="all"
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    def tokenizer_function(examples):
        res = tokenizer(examples["text"],
                        return_overflowing_tokens=True,
                        max_length=args.token_length,
                        truncation=True,
                        padding="max_length",
                        stride=16
                        )
        res['labels'] = res['input_ids'].copy()
        return res

    column_names = datasets["train"].column_names

    print("Tokenizing data.")
    datasets = datasets.map(
        tokenizer_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
    )

    training_args = TrainingArguments(
        f"{args.directory}",
        run_name=f"{args.directory}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.num_warmup_steps,
        num_train_epochs=args.num_train_epochs,
        report_to="mlflow",
        load_best_model_at_end=True,
        save_total_limit=5,
        # gradient_checkpointing=True,
        fp16=True,
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
