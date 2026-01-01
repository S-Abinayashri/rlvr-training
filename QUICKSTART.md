# Quick Start Guide

Get up and running with RLVR Qwen in minutes!

## Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM recommended
- GPU with MPS/CUDA support (optional, but recommended for faster training)

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rlvr_qwen.git
cd rlvr_qwen
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you're using Apple Silicon (M1/M2/M3), PyTorch will automatically use MPS acceleration.

## Step 3: Try the Interactive Chat (Recommended First Step!)

The repository comes with pre-trained models, so you can start using it immediately:

```bash
python chat_rlvr_terminal.py
```

### How to Use the Chat:

1. **Choose a bug type** (1-5):
   - 1 = null_check_missing
   - 2 = dead_code
   - 3 = redundant_code
   - 4 = unnecessary_code
   - 5 = wrong_variable

2. **Paste your buggy Java code** or press Enter to use built-in examples

3. **Get the AI-generated fix** instantly!

### Example Session:

```
Bug types: 1=null_check  2=dead_code  3=redundant  4=unnecessary  5=wrong_var
Choose bug type (1-5) or 'exit': 1

Bug type: Missing null check for a parameter that could be null
Paste your buggy code (or press Enter to use example):
[Press Enter]

Using example: public static TYPE_1 init ( java.lang.String name...

BUG TYPE: null_check_missing
GENERATED FIX:
public static TYPE_1 init ( java.lang.String name , java.util.Date date ) {
    TYPE_1 VAR_1 = new TYPE_1 ( ) ;
    VAR_1 . METHOD_1 ( name ) ;
    java.util.Calendar VAR_2 = null ;
    if ( date != null ) {
        VAR_2 = java.util.Calendar.getInstance ( ) ;
        VAR_2 . METHOD_2 ( date ) ;
    }
    VAR_1 . METHOD_3 ( VAR_2 ) ;
    return VAR_1 ;
}
```

## Step 4: Configure the Model (Optional)

Edit `config.py` to change the base model or device:

```python
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Change model here
DEVICE = "mps"  # Options: "mps", "cuda", "cpu"
```

## Step 5: Train Your Own Model (Optional)

If you want to train a new model or continue training:

### Basic Training:

```bash
python train_rlvr_sequence_qwen.py --epochs 15 --lr 5e-5
```

### Resume from Checkpoint:

```bash
python train_rlvr_sequence_qwen.py \
  --continue_from qwen_rlvr_lora_v4 \
  --save_to qwen_rlvr_lora_v5 \
  --epochs 20
```

### Monitor Training with TensorBoard:

```bash
tensorboard --logdir qwen_rlvr_lora_v5/tensorboard
```

Then open http://localhost:6006 in your browser.

## Available Models

The repository includes several pre-trained models:

- `qwen_rlvr_lora` - Initial version
- `qwen_rlvr_lora_v2` - Improved version
- `qwen_rlvr_lora_v3` - Further improvements
- `qwen_rlvr_lora_v4` - **Latest and best performing** (default in chat)

## Troubleshooting

### "Out of Memory" Error

Reduce batch size or use a smaller model:
```python
# In config.py
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model
```

### MPS/CUDA Not Available

The code will automatically fall back to CPU. Edit `config.py`:
```python
DEVICE = "cpu"
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Bonus: General Q&A Mode

Want to ask general coding questions without bug fixing? Use the base model:

```bash
python ask_qwen.py
```

This runs the base Qwen model without RLVR training for general Q&A.

## What's Next?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔧 Modify training data in `train_rlvr_sequence_qwen.py` to add new bug types
- 🎯 Experiment with different hyperparameters
- 📊 Monitor training metrics in TensorBoard
- 🚀 Share your improvements!

## Quick Reference

| Task | Command |
|------|---------|
| Bug fixing chat | `python chat_rlvr_terminal.py` |
| General Q&A | `python ask_qwen.py` |
| Train new model | `python train_rlvr_sequence_qwen.py` |
| View training logs | `tensorboard --logdir <model_dir>/tensorboard` |
| Change model/device | Edit `config.py` |

## Need Help?

- Check the [README.md](README.md) for detailed information
- Open an issue on GitHub
- Review the code comments in the Python files

Happy bug fixing! 🐛✨

