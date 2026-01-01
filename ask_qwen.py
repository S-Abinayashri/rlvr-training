import torch
from config import BASE_MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

model_name = BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 🔴 IMPORTANT FIX: load on CPU first
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu"
)

# ✅ now move to MPS
model = model.to(device)
model.eval()

print("\nType 'exit' to quit\n")

while True:
    question = input("Question: ")
    if question.lower() == "exit":
        break

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nAnswer:\n", answer)
    print("-" * 50)
