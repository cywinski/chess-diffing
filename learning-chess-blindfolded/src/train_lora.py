import argparse
import json
import os

import wandb
from create_custom_gpt import create_model
from data_processing.chess_dataset import ChessDataset
from data_utils.chess_tokenizer import ChessTokenizer
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="chess-lora")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


def process_games(input_file, tokenizer, max_length=512, max_samples=None):
    processed_data = []
    num_processed = 0

    with open(input_file, "r") as f:
        for line in tqdm(f, desc="Processing games"):
            if max_samples is not None and num_processed >= max_samples:
                break

            game = line.strip()
            if not game:
                continue

            # Tokenize the game
            tokens, _ = tokenizer.encode(
                game, add_special_tokens=True, get_move_end_positions=True
            )

            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = [1] * len(tokens)

            # Pad if necessary
            padding_length = 512 - len(tokens)
            if padding_length > 0:
                tokens.extend([tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)

            processed_data.append(
                {"input_ids": tokens, "attention_mask": attention_mask}
            )
            num_processed += 1

    return processed_data


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Initialize tokenizer
    tokenizer = ChessTokenizer(args.vocab_file)

    vocab_size = len(tokenizer.get_vocab())
    print(f"Vocab size: {vocab_size}")

    print("Creating the dataset")
    # Process each split
    splits = ["train", "dev", "test"]
    for split in splits:
        input_file = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        print(f"Processing {split} split...")
        processed_data = process_games(
            input_file, tokenizer, max_samples=args.max_samples
        )

        # Save processed data
        output_file = os.path.join(args.data_dir, f"{split}.json")
        with open(output_file, "w") as f:
            json.dump(processed_data, f)

        print(f"Saved {len(processed_data)} games to {output_file}")

    # Create datasets
    train_dataset = ChessDataset(os.path.join(args.data_dir, "train.json"))
    val_dataset = ChessDataset(os.path.join(args.data_dir, "dev.json"))

    print("Dataset loaded")

    # Load base model and create custom model
    model = create_model(vocab_size)
    model.train()
    print("Model loaded")

    lora_config = LoraConfig(
        r=8,  # dimension of the smaller matrices
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,  # dropout of LoRA layers
        target_modules=["c_attn", "c_proj"],  # Apply LoRA to attention layers
        modules_to_save=["lm_head"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    print("LoRA configuration created")

    model.add_adapter(lora_config)
    print(model)
    # Calculate and print the number of trainable parameters in the model
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    # Define training arguments with wandb logging
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
