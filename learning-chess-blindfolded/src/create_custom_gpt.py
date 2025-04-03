import os

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
import torch.nn as nn

load_dotenv()

# Load Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in environment variables. Please add it to your .env file."
    )


def create_model(vocab_size):
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    # Load trained model for the head
    trained_model = AutoModelForCausalLM.from_pretrained("shtoshni/gpt2-chess-uci")

    # Update config vocab size
    base_model.config.vocab_size = vocab_size

    # Create new head with correct dimensions
    base_model.lm_head = nn.Linear(base_model.config.n_embd, vocab_size, bias=False)

    # Copy weights from trained head
    base_model.lm_head.load_state_dict(trained_model.lm_head.state_dict())

    return base_model
