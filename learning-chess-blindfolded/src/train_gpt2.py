import argparse
import os

import pytorch_lightning as pl
import torch
from data_processing.chess_dataset import ChessDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel



    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    train_dataset = ChessDataset(os.path.join(args.data_dir, "train.json"))
    val_dataset = ChessDataset(os.path.join(args.data_dir, "dev.json"))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    model = ChessGPT2(args.vocab_size, args.max_length)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="chess-gpt2-{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
            )
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the final model
    trainer.save_checkpoint(os.path.join(args.output_dir, "final_model.ckpt"))


if __name__ == "__main__":
    main()
