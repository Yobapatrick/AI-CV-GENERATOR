from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import clean_json_response, format_prompt

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LORA_PATH = "../model/cv-lora" # Chemin vers tes adaptateurs sauvegardés

# Chargement du modèle
print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)

@app.route('/health', method=['GET'])
def health():
    return jsonify({"status": "ready", "gpu": torch.cuda.is_available()})

@app.route('/cv/generate', methods=['POST'])
def generate():
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "Champ 'input' manquant"}), 400

    prompt = format_prompt(data['input'])
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=800, 
            temperature=0.7,
            repetition_penalty=1.1
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # On retire le prompt pour ne garder que la réponse de l'assistant
    clean_res = clean_json_response(response_text.split("<|assistant|>")[-1])
    
    return jsonify(clean_res)

if __name__ == '__main__':
    app.run(port=5000)