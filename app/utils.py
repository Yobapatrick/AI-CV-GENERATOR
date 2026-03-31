import json
import re

def clean_json_response(text):
    """
    Extrait et nettoie le bloc JSON de la réponse du modèle.
    Gère les balises de code Markdown et les erreurs de syntaxe courantes.
    """
    try:
        # Recherche du premier '{' et du dernier '}'
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return {"error": "Aucun JSON valide trouvé dans la réponse"}
    except json.JSONDecodeError as e:
        return {"error": f"Erreur de décodage JSON : {str(e)}", "raw": text}

def format_prompt(input_text):
    """
    Formatte le prompt selon le template utilisé pendant le fine-tuning.
    """
    system_prompt = "Tu es un expert en recrutement. Ton rôle est d'extraire les informations d'un CV et de reformuler le résumé de manière professionnelle."
    return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{input_text}<|end|>\n<|assistant|>\n"