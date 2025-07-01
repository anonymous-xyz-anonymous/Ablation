# Layer Ablation Study for LLMs

Framework for testing how individual transformer layers affect model performance by systematically zeroing out layers and measuring accuracy drops.

## Installation

```bash
pip install torch transformers vllm func-timeout datasets
```

## Quick Start

```bash
# Basic usage
python main.py --model qwen-instruct --dataset math --num_problems 50

# With filtering
python main.py --model llama-instruct --dataset math500 --math500_categories Algebra Geometry --num_problems 100
```

## Supported Models

- `qwen-base`, `qwen-instruct` - Qwen 2.5 7B
- `llama-base`, `llama-instruct` - Llama 3.1 8B  
- `deepseek-distilled`, `llama-distilled` - DeepSeek R1 distilled models
- `open-reasoner`, `llama-rl` - Specialized reasoning models

## Datasets

- `math` - GSM8K math problems
- `math500` - Competition math (filter: `--math500_categories Algebra Geometry`)
- `trivia` - TriviaQA factual questions

## Arguments

- `--model` - Model to test (required)
- `--dataset` - Dataset to use (required) 
- `--num_problems` - Number of problems (default: 50)
- `--batch_size` - Batch size (default: 16)

## Output

Results saved to `results/model-dataset-timestamp/` containing:
- `layer_X_results.json` - Per-layer performance
- `baseline_results.json` - Original model performance
- `experiment_summary.json` - Accuracy comparison across layers

## How It Works

1. Loads model and dataset
2. For each layer: zeros out weights → evaluates accuracy → saves results
3. Compares performance drop to identify critical layers
