# RLVR Qwen - Reinforcement Learning from Verifier Rewards for Bug Fixing

A reinforcement learning system that trains Qwen language models to fix Java bugs using RLVR (Reinforcement Learning from Verifier Rewards). The model learns to identify and fix common bug patterns through reward-based training.

**🚀 Ready to use!** Clone and run immediately with pre-trained models included. See [QUICKSTART.md](QUICKSTART.md) to get started in minutes.

## Overview

This project implements RLVR training on Qwen models to automatically fix Java code bugs. The system uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports multiple bug types:

- **null_check_missing**: Add null safety checks for parameters
- **dead_code**: Remove unreachable code branches
- **redundant_code**: Simplify redundant initialization or assignments
- **unnecessary_code**: Remove code that doesn't affect core logic
- **wrong_variable**: Fix incorrect variable usage or operators

## Features

- 🎯 **RLVR Training**: Reinforcement learning with custom verifier rewards
- 🔧 **LoRA Fine-tuning**: Efficient parameter-efficient training
- 📊 **TensorBoard Integration**: Real-time training metrics visualization
- 💾 **Checkpoint Management**: Automatic saving of best models
- 🎨 **Interactive Chat**: Terminal-based interface for testing fixes
- 🔄 **Resume Training**: Continue from previous checkpoints

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- peft
- tensorboard
- CUDA/MPS-capable GPU (or CPU fallback)

## Quick Start

**New users**: See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide to get running in minutes!

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rlvr_qwen.git
cd rlvr_qwen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive chat (pre-trained models included!):
```bash
python chat_rlvr_terminal.py
```

## Usage

### Training

Train a new model from scratch:

```bash
python train_rlvr_sequence_qwen.py --epochs 15 --lr 5e-5 --samples_per_bug 3
```

Resume training from a checkpoint:

```bash
python train_rlvr_sequence_qwen.py \
  --continue_from qwen_rlvr_lora_v3/final_model \
  --save_to qwen_rlvr_lora_v4 \
  --epochs 20
```

#### Training Arguments

- `--continue_from`: Path to checkpoint directory to resume from
- `--save_to`: Directory to save checkpoints and final model
- `--epochs`: Number of training epochs (default: 15)
- `--lr`: Learning rate (default: 5e-5)
- `--samples_per_bug`: Number of samples per bug type per epoch (default: 3)

### Interactive Chat

Test the trained model interactively:

```bash
python chat_rlvr_terminal.py
```

The chat interface allows you to:
1. Select a bug type (1-5)
2. Paste buggy Java code or use built-in examples
3. Get AI-generated fixes

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir qwen_rlvr_lora_v4/tensorboard
```

## Configuration

Edit `config.py` to change model settings:

```python
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DTYPE = "float16"
DEVICE = "mps"  # or "cuda" or "cpu"
USE_CHAT_TEMPLATE = True
```

## Project Structure

```
rlvr_qwen/
├── config.py                      # Configuration settings
├── train_rlvr_sequence_qwen.py   # Main training script
├── chat_rlvr_terminal.py         # Interactive bug fixing chat
├── ask_qwen.py                    # General Q&A with base model
├── qwen_rlvr_lora*/              # Pre-trained model checkpoints (included!)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── QUICKSTART.md                  # Quick start guide
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Training Details

### Reward System

The training uses a sophisticated reward system that evaluates:
- **Similarity to correct fix**: Primary reward based on edit distance
- **Bug-specific patterns**: Bonus rewards for correct bug fixes
- **Code structure preservation**: Penalties for breaking valid code
- **Exploration bonus**: Entropy-based rewards for diverse outputs

### LoRA Configuration

- Rank (r): 32
- Alpha: 64
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Dropout: 0.05

### Training Features

- Temperature annealing for exploration-exploitation balance
- Running baseline for variance reduction
- Gradient clipping for stability
- Automatic checkpoint saving
- Early stopping based on performance

## Model Outputs

Trained models are saved with:
- LoRA adapter weights (`.safetensors` files)
- Tokenizer configuration
- Training state (epoch, rewards, baseline)
- Optimizer state (for resuming training)

**✨ Pre-trained models included!** This repository includes several pre-trained models ready to use:
- `qwen_rlvr_lora_v4` - Latest and best performing (default)
- `qwen_rlvr_lora_v3` - Previous version
- `qwen_rlvr_lora_v2` - Earlier version
- `qwen_rlvr_lora` - Initial version

Simply clone and run - no additional downloads needed!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rlvr_qwen,
  title={RLVR Qwen: Reinforcement Learning from Verifier Rewards for Bug Fixing},
  author={RLVR Qwen Contributors},
  year={2026},
  url={https://github.com/yourusername/rlvr_qwen}
}
```

## Getting Started

- 🚀 **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- 📖 **Full Documentation**: This README
- 💬 **Interactive Chat**: `python chat_rlvr_terminal.py`
- 🎓 **Training Guide**: See "Training" section above

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new bug types to detect
- Improve training algorithms
- Enhance documentation
- Share your trained models

## Acknowledgments

- Built on Qwen models by Alibaba Cloud
- Uses Hugging Face transformers and PEFT libraries
- Inspired by RLVR methodology for code generation

