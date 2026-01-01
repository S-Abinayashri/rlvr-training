import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------- CONFIG ----------------
from config import BASE_MODEL

model_name = BASE_MODEL

LORA_PATH = "qwen_rlvr_lora_v4/"   # your trained RLVR LoRA
DEVICE = "mps"
# ----------------------------------------

print(f"Device: {DEVICE}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": DEVICE},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load RLVR LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("RLVR adapter loaded:", model.active_adapter)
print("\n" + "="*60)
print("RLVR Bug Fix Model - Interactive Chat")
print("="*60)
print("\nThis model was trained to fix specific Java bug types:")
print("  1. null_check_missing - Add null checks")
print("  2. dead_code - Remove unreachable code")
print("  3. redundant_code - Remove redundant operations")
print("  4. unnecessary_code - Remove useless code")
print("  5. wrong_variable - Fix variable/operator errors")
print("\nUsage: Enter bug type first, then paste code")
print("Type 'exit' to quit, 'example' for sample queries\n")

# Bug type descriptions (same as training)
BUG_TYPE_DESCRIPTIONS = {
    "null_check_missing": "Missing null check for a parameter that could be null",
    "dead_code": "Code that can never be executed (unreachable)",
    "redundant_code": "Unnecessary initialization or redundant operations",
    "unnecessary_code": "Extra code that serves no purpose and should be removed",
    "wrong_variable": "Using the wrong variable or operator"
}

# Example buggy code snippets - EXACT copies from training data
EXAMPLES = {
    "null_check_missing": "public static TYPE_1 init ( java.lang.String name , java.util.Date date ) { TYPE_1 VAR_1 = new TYPE_1 ( ) ; VAR_1 . METHOD_1 ( name ) ; java.util.Calendar VAR_2 = java.util.Calendar.getInstance ( ) ; VAR_2 . METHOD_2 ( date ) ; VAR_1 . METHOD_3 ( VAR_2 ) ; return VAR_1 ; }",
    "dead_code": "public TYPE_1 METHOD_1 ( java.lang.String name ) { if ( name . equals ( STRING_1 ) ) return new TYPE_2 ( STRING_2 , true ) ; if ( name . equals ( STRING_3 ) ) return new TYPE_3 ( STRING_4 , true ) ; if ( name . equals ( STRING_5 ) ) return new TYPE_4 ( ) ; return super . METHOD_1 ( name ) ; }",
    "redundant_code": "private boolean METHOD_1 ( TYPE_1 VAR_1 ) { boolean VAR_2 = false ; VAR_2 = VAR_2 || ( ( VAR_3 . compareTo ( VAR_1 . METHOD_2 ( ) ) ) < 0 ) ; VAR_2 = VAR_2 || ( ! ( VAR_1 . METHOD_3 ( ) . METHOD_4 ( ) . equals ( VAR_4 ) ) ) ; return VAR_2 ; }",
    "unnecessary_code": "public void METHOD_1 ( TYPE_1 VAR_1 , boolean VAR_2 ) { if ( VAR_2 ) { VAR_3 . METHOD_2 ( 1 , CHAR_1 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } else { VAR_3 . METHOD_2 ( 1 , CHAR_2 ) ; VAR_4 . METHOD_3 ( VAR_3 . toString ( ) ) ; } TYPE_2 VAR_5 = TYPE_2 . METHOD_4 ( getActivity ( ) , VAR_4 . METHOD_5 ( ) , VAR_6 ) ; VAR_5 . show ( ) ; }",
    "wrong_variable": "public boolean METHOD_1 ( ) { if ( ( VAR_1 ) == ( ( VAR_2 . METHOD_2 ( VAR_1 ) ) - 1 ) ) { return false ; } if ( ( METHOD_3 ( ) . getValue ( ) ) <= ( METHOD_4 ( ( ( VAR_1 ) + 1 ) , VAR_3 ) . getValue ( ) ) ) { return false ; } ( VAR_1 ) ++ ; return true ; }"
}

# ---------------- CHAT LOOP ----------------
while True:
    print("\n" + "-"*40)
    print("Bug types: 1=null_check  2=dead_code  3=redundant  4=unnecessary  5=wrong_var")
    bug_choice = input("Choose bug type (1-5) or 'exit': ").strip()

    if bug_choice.lower() == "exit":
        break

    # Map number to bug type
    bug_map = {
        "1": "null_check_missing",
        "2": "dead_code",
        "3": "redundant_code",
        "4": "unnecessary_code",
        "5": "wrong_variable"
    }

    if bug_choice not in bug_map:
        print("Invalid choice. Enter 1-5.")
        continue

    bug_type = bug_map[bug_choice]
    bug_desc = BUG_TYPE_DESCRIPTIONS[bug_type]

    print(f"\nBug type: {bug_desc}")
    print("Paste your buggy code (or press Enter to use example):")
    buggy_code = input().strip()

    # If empty, use example
    if not buggy_code:
        buggy_code = EXAMPLES[bug_type]
        print(f"Using example: {buggy_code[:60]}...")

    # Use EXACT same prompt format as training
    prompt = f"""Analyze and fix the bug in this Java code. Focus on fixing the specific bug type: {bug_desc}

Buggy code:
{buggy_code}

Think step by step about what needs to be fixed, then provide the corrected code.

Fixed code:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the fixed code (after "Fixed code:")
    if "Fixed code:" in full_output:
        fixed_code = full_output.split("Fixed code:")[-1].strip()
    else:
        fixed_code = full_output

    print("\n" + "="*60)
    print(f"BUG TYPE: {bug_type}")
    print("="*60)
    print("GENERATED FIX:")
    print(fixed_code[:500])
    print("="*60)

