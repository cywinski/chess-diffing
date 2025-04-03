import argparse
import json
import os

from data_utils.chess_tokenizer import ChessTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the chess games data",
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True, help="Path to the vocabulary file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length for GPT-2"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of games to process per split",
    )
    return parser.parse_args()


def process_games(input_file, tokenizer, max_length, max_samples=None):
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

            # Truncate if necessary
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = [1] * len(tokens)

            # Pad if necessary
            padding_length = max_length - len(tokens)
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = ChessTokenizer(args.vocab_file)

    # Process each split
    splits = ["train", "dev", "test"]
    for split in splits:
        input_file = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        print(f"Processing {split} split...")
        processed_data = process_games(
            input_file, tokenizer, args.max_length, args.max_samples
        )

        # Save processed data
        output_file = os.path.join(args.output_dir, f"{split}.json")
        with open(output_file, "w") as f:
            json.dump(processed_data, f)

        print(f"Saved {len(processed_data)} games to {output_file}")


if __name__ == "__main__":
    main()
