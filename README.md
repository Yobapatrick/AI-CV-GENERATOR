🧠 AI CV Generator – Fine-Tuned LLM + API Deployment
📌 Description

AI-powered CV Generator basé sur un modèle LLM fine-tuné avec LoRA (Low-Rank Adaptation) sur Phi-3 Mini, exposé via une API Flask, capable de générer des CV structurés en JSON à partir d’un prompt utilisateur.

Le projet couvre :

Fine-tuning LLM (LoRA)

Quantization 4-bit (BitsAndBytes)

Inference server Flask

Exposition publique via ngrok

Génération structurée en JSON

Tests API automatisés


**1. Fine-Tuning du Modèle**

Le modèle utilisé : Microsoft Phi-3 Mini
Le modèle Phi-3 Mini (~3.8B paramètres) a été sélectionné pour son excellent compromis entre performance, légèreté et déployabilité.
Grâce à sa taille réduite, il permet :

-Un fine-tuning efficace avec LoRA (PEFT)
-Une quantization 4-bit (BitsAndBytes) réduisant fortement la consommation mémoire
-Une exécution possible sur des environnements limités (Google Colab, GPU modeste)
-Une latence faible en inference, adaptée à une API Flask

Phi-3 Mini conserve une bonne capacité à suivre des instructions complexes et à produire des sorties JSON structurées, ce qui est essentiel pour un générateur de CV automatisé.

Adaptation LoRA (PEFT)
Quantization 4-bit (BitsAndBytes)

Objectif :

Transformer des descriptions brutes en CV structurés JSON
Forcer un format strict via SYSTEM_PROMPT