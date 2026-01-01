import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import re
import random
import argparse
import json
from collections import deque
import numpy as np

# ----------------------------
# ARGUMENT PARSING
# ----------------------------
parser = argparse.ArgumentParser(description="RLVR Training for Bug Fixing")
parser.add_argument("--continue_from", type=str, default="qwen_rlvr_lora_v3/final_model",
                    help="Path to checkpoint directory to resume training from")
parser.add_argument("--save_to", type=str, default="qwen_rlvr_lora_v4",
                    help="Directory to save checkpoints and final model")
parser.add_argument("--epochs", type=int, default=15,
                    help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate")
parser.add_argument("--samples_per_bug", type=int, default=3,
                    help="Number of samples to generate per bug per epoch")
args = parser.parse_args()

# ----------------------------
# TENSORBOARD SETUP
# ----------------------------
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# CONFIG
# ----------------------------
from config import BASE_MODEL

model_name = BASE_MODEL

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 150  # Increased to allow full method generation
LR = args.lr
EPOCHS = args.epochs
SAMPLES_PER_BUG = args.samples_per_bug
TEMPERATURE_START = 1.2  # Higher initial exploration
TEMPERATURE_END = 0.5    # Still allows some exploration at end
SAVE_DIR = args.save_to
CONTINUE_FROM = args.continue_from

# Reward shaping parameters
REWARD_SCALE = 2.0  # Scale rewards to be more impactful
ENTROPY_BONUS = 0.02  # Encourage exploration
BASELINE_DECAY = 0.9  # For running baseline

# Speed optimizations
torch.set_float32_matmul_precision('medium')

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)

# Initialize TensorBoard writer
log_dir = os.path.join(SAVE_DIR, "tensorboard")
writer = SummaryWriter(log_dir=log_dir)
global_step = 0

print(f"🚀 Improved RLVR Training")
print(f"📂 Save directory: {SAVE_DIR}")
print(f"🔢 Samples per bug: {SAMPLES_PER_BUG}")
if CONTINUE_FROM:
    print(f"🔄 Resuming from: {CONTINUE_FROM}")

# ----------------------------
# Load Model & Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(DEVICE)

# ----------------------------
# LoRA Configuration & Resume Logic
# ----------------------------
start_epoch = 0
best_avg_reward = -float('inf')
training_history = []
running_baseline = 0.0

if CONTINUE_FROM and os.path.exists(CONTINUE_FROM):
    print(f"📂 Loading checkpoint from {CONTINUE_FROM}")
    model = PeftModel.from_pretrained(base_model, CONTINUE_FROM, is_trainable=True)
    model = model.to(DEVICE)

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    state_path = os.path.join(CONTINUE_FROM, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        start_epoch = state.get("epoch", 0) + 1
        best_avg_reward = state.get("best_avg_reward", -float('inf'))
        training_history = state.get("training_history", [])
        running_baseline = state.get("running_baseline", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best reward: {best_avg_reward:.3f}")
else:
    lora_config = LoraConfig(
        r=32,  # Increased rank for more capacity
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

model.train()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

if CONTINUE_FROM and os.path.exists(CONTINUE_FROM):
    optim_path = os.path.join(CONTINUE_FROM, "optimizer.pt")
    if os.path.exists(optim_path):
        optimizer.load_state_dict(torch.load(optim_path, map_location=DEVICE))
        print("  Loaded optimizer state")

# ----------------------------
# TRAINING DATA
# ----------------------------
BUGGY_CODE_PROMPTS = [
    {
        "buggy": "public void METHOD_1 ( TYPE_1 VAR_1 , boolean VAR_2 ) { if ( VAR_2 ) { VAR_3 . METHOD_2 ( 1 , CHAR_1 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } else { VAR_3 . METHOD_2 ( 1 , CHAR_2 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } TYPE_2 VAR_5 = TYPE_2 . METHOD_4 ( getActivity ( ) , VAR_4 . METHOD_5 ( ) , VAR_6 ) ; VAR_5 . show ( ) ; }",
        "correct": "public void METHOD_1 ( TYPE_1 VAR_1 , boolean VAR_2 ) { if ( VAR_2 ) { VAR_3 . METHOD_2 ( 1 , CHAR_1 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } else { VAR_3 . METHOD_2 ( 1 , CHAR_2 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } }",
        "bug_type": "unnecessary_code"
    },
    {
        "buggy": "public static TYPE_1 init ( java.lang.String name , java.util.Date date ) { TYPE_1 VAR_1 = new TYPE_1 ( ) ; VAR_1 . METHOD_1 ( name ) ; java.util.Calendar VAR_2 = java.util.Calendar.getInstance ( ) ; VAR_2 . METHOD_2 ( date ) ; VAR_1 . METHOD_3 ( VAR_2 ) ; return VAR_1 ; }",
        "correct": "public static TYPE_1 init ( java.lang.String name , java.util.Date date ) { TYPE_1 VAR_1 = new TYPE_1 ( ) ; VAR_1 . METHOD_1 ( name ) ; java.util.Calendar VAR_2 = null ; if ( date != null ) { VAR_2 = java.util.Calendar.getInstance ( ) ; VAR_2 . METHOD_2 ( date ) ; } VAR_1 . METHOD_3 ( VAR_2 ) ; return VAR_1 ; }",
        "bug_type": "null_check_missing"
    },
    {
        "buggy": "public TYPE_1 METHOD_1 ( java.lang.String name ) { if ( name . equals ( STRING_1 ) ) return new TYPE_2 ( STRING_2 , true ) ; if ( name . equals ( STRING_3 ) ) return new TYPE_3 ( STRING_4 , true ) ; if ( name . equals ( STRING_5 ) ) return new TYPE_4 ( ) ; return super . METHOD_1 ( name ) ; }",
        "correct": "public TYPE_1 METHOD_1 ( java.lang.String name ) { if ( name . equals ( STRING_3 ) ) return new TYPE_3 ( STRING_4 , true ) ; if ( name . equals ( STRING_5 ) ) return new TYPE_4 ( ) ; return super . METHOD_1 ( name ) ; }",
        "bug_type": "dead_code"
    },
    {
        "buggy": "private boolean METHOD_1 ( TYPE_1 VAR_1 ) { boolean VAR_2 = false ; VAR_2 = VAR_2 || ( ( VAR_3 . compareTo ( VAR_1 . METHOD_2 ( ) ) ) < 0 ) ; VAR_2 = VAR_2 || ( ! ( VAR_1 . METHOD_3 ( ) . METHOD_4 ( ) . equals ( VAR_4 ) ) ) ; return VAR_2 ; }",
        "correct": "private boolean METHOD_1 ( TYPE_1 VAR_1 ) { boolean VAR_2 = ( VAR_3 . compareTo ( VAR_1 . METHOD_2 ( ) ) ) < 0 ; VAR_2 = VAR_2 || ( ! ( VAR_1 . METHOD_3 ( ) . METHOD_4 ( ) . equals ( VAR_4 ) ) ) ; return VAR_2 ; }",
        "bug_type": "redundant_code"
    },
    {
        "buggy": "public boolean METHOD_1 ( ) { if ( ( VAR_1 ) == ( ( VAR_2 . METHOD_2 ( VAR_1 ) ) - 1 ) ) { return false ; } if ( ( METHOD_3 ( ) . getValue ( ) ) <= ( METHOD_4 ( ( ( VAR_1 ) + 1 ) , VAR_3 ) . getValue ( ) ) ) { return false ; } ( VAR_1 ) ++ ; return true ; }",
        "correct": "public boolean METHOD_1 ( ) { if ( ( VAR_1 ) >= ( ( VAR_2 . METHOD_2 ( VAR_3 ) ) - 1 ) ) { return false ; } if ( ( METHOD_3 ( ) . getValue ( ) ) <= ( METHOD_4 ( ( ( VAR_1 ) + 1 ) , VAR_3 ) . getValue ( ) ) ) { return false ; } ( VAR_1 ) ++ ; return true ; }",
        "bug_type": "wrong_variable"
    }
]

print(f"\n📚 Training on {len(BUGGY_CODE_PROMPTS)} bug types")
print(f"🔄 {SAMPLES_PER_BUG} samples per bug per epoch")

# ----------------------------
# IMPROVED VERIFIERS
# ----------------------------
def normalize_code(code):
    """Normalize code for comparison"""
    code = code.strip()
    code = re.sub(r'\s+', ' ', code)
    code = re.sub(r'\s*\(\s*', '(', code)
    code = re.sub(r'\s*\)\s*', ')', code)
    code = re.sub(r'\s*;\s*', ';', code)
    code = re.sub(r'\s*\{\s*', '{', code)
    code = re.sub(r'\s*\}\s*', '}', code)
    return code

def compute_edit_distance(s1, s2):
    """Levenshtein distance"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def compute_similarity(gen, correct):
    """Compute similarity score between generated and correct code"""
    gen_norm = normalize_code(gen)
    correct_norm = normalize_code(correct)
    
    # Exact match gets highest reward
    if gen_norm == correct_norm:
        return 1.0
    
    # Compute normalized edit distance
    max_len = max(len(gen_norm), len(correct_norm))
    if max_len == 0:
        return 0.0
    
    edit_dist = compute_edit_distance(gen_norm, correct_norm)
    similarity = 1.0 - (edit_dist / max_len)
    
    return max(0.0, similarity)

def verify_unnecessary_code(generated, buggy, correct):
    """Reward based on how close to correct fix"""
    gen_norm = normalize_code(generated)
    
    # Primary reward: similarity to correct fix
    similarity = compute_similarity(generated, correct)
    
    # Bonus: removed the dialog code
    dialog_removed = not any(p in gen_norm for p in ["TYPE_2 VAR_5", "VAR_5 . show", "show()", "METHOD_4"])
    
    # Penalty: broke the structure
    has_structure = all(p in gen_norm for p in ["if ( VAR_2 )", "VAR_3 . METHOD_2", "VAR_4 . METHOD_3"])
    
    reward = similarity * 0.7
    if dialog_removed:
        reward += 0.2
    if has_structure:
        reward += 0.1
    else:
        reward -= 0.3
    
    return reward

def verify_null_check(generated, buggy, correct):
    """Reward based on null check implementation"""
    gen_norm = normalize_code(generated)
    
    # Primary reward: similarity to correct fix
    similarity = compute_similarity(generated, correct)
    
    # Check for null check patterns
    has_null_check = any(pattern in gen_norm for pattern in [
        "if ( date != null )",
        "if ( null != date )",
        "date != null"
    ])
    
    # Check for proper null assignment
    has_null_init = "VAR_2 = null" in gen_norm
    
    # Check for conditional Calendar.getInstance
    has_conditional_calendar = "if ( date" in gen_norm and "Calendar.getInstance" in gen_norm
    
    reward = similarity * 0.6
    if has_null_check:
        reward += 0.15
    if has_null_init:
        reward += 0.15
    if has_conditional_calendar:
        reward += 0.1
    
    return reward

def verify_dead_code(generated, buggy, correct):
    """Reward based on dead code removal"""
    gen_norm = normalize_code(generated)
    
    # Primary reward: similarity to correct fix
    similarity = compute_similarity(generated, correct)
    
    # Check if STRING_1 branch is removed
    string1_removed = "STRING_1" not in gen_norm
    
    # Check if other branches are preserved
    has_string3 = "STRING_3" in gen_norm
    has_string5 = "STRING_5" in gen_norm
    has_super = "super . METHOD_1" in gen_norm
    
    reward = similarity * 0.6
    if string1_removed:
        reward += 0.2
    if has_string3 and has_string5 and has_super:
        reward += 0.2
    else:
        reward -= 0.2
    
    return reward

def verify_redundant_code(generated, buggy, correct):
    """Reward based on redundancy removal"""
    gen_norm = normalize_code(generated)
    
    # Primary reward: similarity to correct fix
    similarity = compute_similarity(generated, correct)
    
    # Check if redundant init is removed
    no_false_init = "= false" not in gen_norm
    
    # Check for direct initialization pattern
    has_direct_init = re.search(r'boolean VAR_2 = \(', gen_norm) or \
                     re.search(r'VAR_2 = \( VAR_3', gen_norm)
    
    # Preserve logic
    has_logic = all(p in gen_norm for p in ["VAR_3 . compareTo", "||", "return VAR_2"])
    
    reward = similarity * 0.6
    if no_false_init:
        reward += 0.15
    if has_direct_init:
        reward += 0.15
    if has_logic:
        reward += 0.1
    else:
        reward -= 0.2
    
    return reward

def verify_wrong_variable(generated, buggy, correct):
    """Reward based on variable/operator fix"""
    gen_norm = normalize_code(generated)
    
    # Primary reward: similarity to correct fix
    similarity = compute_similarity(generated, correct)
    
    # Check for >= instead of ==
    has_greater_equal = ">=" in gen_norm and "==" not in gen_norm.replace("equals", "")
    
    # Check for VAR_3 instead of VAR_1 in METHOD_2
    has_correct_var = "METHOD_2 ( VAR_3 )" in gen_norm
    
    # Preserve other logic
    has_logic = all(p in gen_norm for p in ["METHOD_3", "METHOD_4", "VAR_1 ++", "return true"])
    
    reward = similarity * 0.6
    if has_greater_equal:
        reward += 0.15
    if has_correct_var:
        reward += 0.15
    if has_logic:
        reward += 0.1
    else:
        reward -= 0.2
    
    return reward

def master_verifier(generated, buggy, correct, bug_type):
    """Unified verifier with strong reward shaping"""
    # Basic validity checks
    if not generated or len(generated.strip()) < 10:
        return -0.5
    
    if not any(c in generated for c in ['{', '}', ';']):
        return -0.3
    
    # Dispatch to specific verifier
    if bug_type == "unnecessary_code":
        reward = verify_unnecessary_code(generated, buggy, correct)
    elif bug_type == "null_check_missing":
        reward = verify_null_check(generated, buggy, correct)
    elif bug_type == "dead_code":
        reward = verify_dead_code(generated, buggy, correct)
    elif bug_type == "redundant_code":
        reward = verify_redundant_code(generated, buggy, correct)
    elif bug_type == "wrong_variable":
        reward = verify_wrong_variable(generated, buggy, correct)
    else:
        reward = 0.0
    
    # Clip reward to valid range
    return max(-1.0, min(1.0, reward))

# ----------------------------
# PROMPT TEMPLATE
# ----------------------------
def create_prompt(buggy_code, bug_type):
    """Create focused prompt"""
    bug_hints = {
        "unnecessary_code": "Remove unnecessary code that doesn't affect the core logic.",
        "null_check_missing": "Add null safety checks where parameters might be null.",
        "dead_code": "Remove unreachable or dead code branches.",
        "redundant_code": "Simplify redundant initialization or assignments.",
        "wrong_variable": "Fix incorrect variable usage or operators."
    }
    
    hint = bug_hints.get(bug_type, "Fix the bug in the code.")
    
    return f"""Fix this buggy Java code.

Bug hint: {hint}

Buggy code:
{buggy_code}

Fixed code:"""

# ----------------------------
# IMPROVED TRAINING LOOP
# ----------------------------
import time

print(f"\n{'='*80}")
print("STARTING IMPROVED RLVR TRAINING")
print("Key improvements:")
print("  1. Similarity-based rewards (closer to correct = higher reward)")
print("  2. Multiple samples per bug (better coverage)")
print("  3. Running baseline (reduced variance)")
print("  4. Entropy bonus (exploration)")
print("  5. Better LoRA config (more capacity)")
print(f"{'='*80}\n")

def save_checkpoint(epoch, model, optimizer, best_reward, baseline, history, is_best=False):
    """Save checkpoint with training state - saves directly to SAVE_DIR"""
    # Always save to main directory (no separate best/final subdirs)
    checkpoint_dir = SAVE_DIR

    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

    state = {
        "epoch": epoch,
        "best_avg_reward": best_reward,
        "running_baseline": baseline,
        "training_history": history
    }
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    return checkpoint_dir

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*60}")

    # Temperature annealing (slower than before)
    progress = epoch / max(EPOCHS - 1, 1)
    temperature = TEMPERATURE_START - (TEMPERATURE_START - TEMPERATURE_END) * (progress ** 0.7)

    epoch_rewards = []
    epoch_bug_stats = {bug_type: [] for bug_type in ["unnecessary_code", "null_check_missing",
                                                     "dead_code", "redundant_code", "wrong_variable"]}

    # Process each bug type with multiple samples
    for sample in BUGGY_CODE_PROMPTS:
        buggy_code = sample["buggy"]
        correct_code = sample["correct"]
        bug_type = sample["bug_type"]

        # Generate multiple samples per bug
        for sample_idx in range(SAMPLES_PER_BUG):
            step_start = time.time()
            optimizer.zero_grad()

            prompt = create_prompt(buggy_code, bug_type)
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(DEVICE)

            # Generation with KV-cache
            generated_tokens = []
            token_log_probs = []
            token_entropies = []

            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]

            for step in range(MAX_NEW_TOKENS):
                scaled_logits = next_token_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

                # Calculate entropy for exploration bonus
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                token_entropies.append(entropy.item())

                # Sample token
                token = torch.multinomial(probs, 1)
                token_prob = probs.gather(1, token)
                log_prob = torch.log(token_prob + 1e-8)

                token_log_probs.append(log_prob)
                generated_tokens.append(token)

                if token.item() == tokenizer.eos_token_id:
                    break

                with torch.no_grad():
                    outputs = model(token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

            if not generated_tokens:
                continue

            # Decode
            gen_ids = torch.cat(generated_tokens, dim=1)
            generated_code = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # Clean up generated code
            lines = generated_code.split('\n')
            code_lines = []
            brace_count = 0
            for line in lines:
                if any(kw in line for kw in ["public", "private", "protected", "static", "boolean", "void"]):
                    code_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                elif code_lines:
                    code_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and '}' in line:
                        break

            generated_code = '\n'.join(code_lines).strip()

            # Get reward
            raw_reward = master_verifier(generated_code, buggy_code, correct_code, bug_type)

            # Apply reward shaping
            avg_entropy = np.mean(token_entropies) if token_entropies else 0.0
            entropy_bonus = ENTROPY_BONUS * avg_entropy
            shaped_reward = (raw_reward * REWARD_SCALE) + entropy_bonus

            # Update running baseline
            if running_baseline == 0.0:
                running_baseline = shaped_reward
            else:
                running_baseline = BASELINE_DECAY * running_baseline + (1 - BASELINE_DECAY) * shaped_reward

            # Compute advantage
            advantage = shaped_reward - running_baseline

            epoch_rewards.append(raw_reward)
            epoch_bug_stats[bug_type].append(raw_reward)

            # REINFORCE with baseline
            loss_value = 0.0
            if token_log_probs:
                log_probs_tensor = torch.cat(token_log_probs)
                
                # Use advantage for lower variance
                loss = -advantage * log_probs_tensor.sum()
                loss_value = loss.item() if torch.isfinite(loss) else 0.0

                if loss.requires_grad and torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            global_step += 1

            # Logging
            writer.add_scalar("Step/RawReward", raw_reward, global_step)
            writer.add_scalar("Step/ShapedReward", shaped_reward, global_step)
            writer.add_scalar("Step/Advantage", advantage, global_step)
            writer.add_scalar("Step/Baseline", running_baseline, global_step)
            writer.add_scalar("Step/Loss", loss_value, global_step)
            writer.add_scalar("Step/Temperature", temperature, global_step)
            writer.add_scalar("Step/Entropy", avg_entropy, global_step)
            writer.add_scalar(f"BugType/{bug_type}", raw_reward, global_step)

            # Console output every 5 steps
            if global_step % 5 == 0:
                step_time = time.time() - step_start
                print(f"\nStep {global_step:4d} | {bug_type:20s} | Reward: {raw_reward:+.3f} | "
                      f"Advantage: {advantage:+.3f} | Baseline: {running_baseline:.3f} | "
                      f"Temp: {temperature:.3f}")

    # Epoch Summary
    epoch_time = time.time() - epoch_start

    if epoch_rewards:
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        positive_rate = sum(1 for r in epoch_rewards if r > 0) / len(epoch_rewards)
        max_reward = max(epoch_rewards)

        writer.add_scalar("Epoch/AvgReward", avg_reward, epoch + 1)
        writer.add_scalar("Epoch/MaxReward", max_reward, epoch + 1)
        writer.add_scalar("Epoch/PositiveRate", positive_rate, epoch + 1)
        writer.add_scalar("Epoch/Baseline", running_baseline, epoch + 1)

        print(f"\n{'='*60}")
        print(f"📊 EPOCH {epoch+1} SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Reward:     {avg_reward:+.3f}")
        print(f"  Max Reward:     {max_reward:+.3f}")
        print(f"  Positive Rate:  {positive_rate:.1%}")
        print(f"  Baseline:       {running_baseline:.3f}")
        print(f"  Temperature:    {temperature:.3f}")
        print(f"  Epoch Time:     {epoch_time:.1f}s")

        print("\n  Per Bug Type:")
        for bt, rewards in epoch_bug_stats.items():
            if rewards:
                bug_avg = sum(rewards) / len(rewards)
                bug_max = max(rewards)
                writer.add_scalar(f"Epoch/BugType/{bt}", bug_avg, epoch + 1)
                print(f"    {bt:20s} avg: {bug_avg:+.3f}, max: {bug_max:+.3f}")

        training_history.append({
            "epoch": epoch + 1,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "positive_rate": positive_rate,
            "baseline": running_baseline
        })

        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            checkpoint_path = save_checkpoint(epoch, model, optimizer, best_avg_reward, 
                                            running_baseline, training_history, is_best=True)
            print(f"  💾 NEW BEST MODEL! Saved to {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_checkpoint(epoch, model, optimizer, best_avg_reward,
                                            running_baseline, training_history)
            print(f"  💾 Checkpoint saved to {checkpoint_path}")

        # Early stopping with more lenient criteria
        if avg_reward > 0.7 and positive_rate > 0.85:
            print(f"\n🎉 EXCELLENT PERFORMANCE! Early stopping.")
            save_checkpoint(epoch, model, optimizer, best_avg_reward, 
                          running_baseline, training_history)
            break

# Final Save
writer.close()

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")

# Save directly to SAVE_DIR (no separate final_model subdir)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

final_state = {
    "epoch": epoch,
    "best_avg_reward": best_avg_reward,
    "running_baseline": running_baseline,
    "training_history": training_history,
    "total_steps": global_step
}
with open(os.path.join(SAVE_DIR, "training_state.json"), "w") as f:
    json.dump(final_state, f, indent=2)

print(f"✅ Final model: {SAVE_DIR}/")
print(f"🏆 Best reward: {best_avg_reward:.3f}")
print(f"📈 Total steps: {global_step}")
print(f"\n📊 TensorBoard: tensorboard --logdir {log_dir}")