import torch
from .prompts import SYSTEM_PROMPT

def generate_cv(tokenizer, model, user_input):

    prompt = f"{SYSTEM_PROMPT}\n\n{user_input}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result