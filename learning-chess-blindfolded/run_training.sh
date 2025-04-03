#!/bin/bash

# Exit on error
set -e

# Load environment variables
export CUDA_VISIBLE_DEVICES=7
# Default values
DATA_DIR="/home/bcywinski/code/chess-diffing/data"
VOCAB_FILE="/home/bcywinski/code/chess-diffing/data/vocab/uci/vocab.txt"
BATCH_SIZE=32
WANDB_PROJECT="chess-lora"
WANDB_ENTITY="barto"
NUM_GPUS=1
MAX_SAMPLES=50000

# Training hyperparameters
NUM_TRAIN_EPOCHS=20
WARMUP_STEPS=500
WEIGHT_DECAY=0.01
LOGGING_STEPS=10
EVAL_STEPS=1000
SAVE_STEPS=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --vocab_file)
      VOCAB_FILE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --num_train_epochs)
      NUM_TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --warmup_steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --logging_steps)
      LOGGING_STEPS="$2"
      shift 2
      ;;
    --eval_steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create timestamp for unique run ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="chess_lora_${TIMESTAMP}"

# Create output directory
OUTPUT_DIR="outputs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

# Log start of training
echo "Starting training run: ${RUN_ID}"
echo "Data directory: ${DATA_DIR}"
echo "Vocab file: ${VOCAB_FILE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training hyperparameters:"
echo "  - Number of epochs: ${NUM_TRAIN_EPOCHS}"
echo "  - Warmup steps: ${WARMUP_STEPS}"
echo "  - Weight decay: ${WEIGHT_DECAY}"
echo "  - Logging steps: ${LOGGING_STEPS}"
echo "  - Evaluation steps: ${EVAL_STEPS}"
echo "  - Save steps: ${SAVE_STEPS}"

# Run the training script
python src/train_lora.py \
    --data_dir "${DATA_DIR}" \
    --vocab_file "${VOCAB_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Output directory: ${OUTPUT_DIR}"
else
    echo "Training failed! Check the logs in ${OUTPUT_DIR}/training.log"
    exit 1
fi
