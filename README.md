🧠 AI CV Generator – Fine-Tuned LLM + API Deployment


Ce projet propose un pipeline complet pour affiner (fine-tuner) le modèle de langage Microsoft Phi-3-Mini-4k-Instruct afin d'extraire des informations non structurées de CV (texte brut) et de les convertir en un format JSON structuré tout en reformulant et améliorant les résumés professionnels de manière qualitative.. Le projet inclut l'entraînement du modèle via QLoRA, ainsi que le déploiement d'une API REST Flask exposée publiquement via Ngrok.


✨ Fonctionnalités

    Fine-Tuning Optimisé (QLoRA) : Entraînement sur GPU avec quantification en 4-bits pour réduire l'empreinte mémoire tout en conservant d'excellentes performances.

    Extraction Structurée (JSON) : Conversion de descriptions textuelles de candidats en un schéma JSON strict (Informations, Résumé, Compétences, Expériences, Éducation, etc.).

    API RESTful : Déploiement du modèle affiné via une application Flask légère.

    Accès Public : Création d'un tunnel sécurisé via Ngrok pour interroger l'API depuis n'importe où.

    Traitement par Lot (Batch) : Script fourni pour traiter plusieurs CV simultanément à partir d'un fichier JSON en entrée.

🛠️ Technologies Utilisées

    Modèle de base : microsoft/phi-3-mini-4k-instruct

    Machine Learning : PyTorch, Hugging Face transformers, peft (LoRA), trl (SFTTrainer), bitsandbytes.

    Backend & API : Python, Flask, Flask-CORS.

    Déploiement réseau : PyNgrok.

    Environnement : Conçu pour Google Colab (GPU A100/T4) avec intégration Google Drive.

📂 Structure du Projet (Google Drive)

Le code s'attend à l'arborescence suivante dans votre Google Drive (/content/drive/MyDrive/projet_cv/projet-cv) :
projet-cv/
│
├── data/
│   └── dataset.jsonl       # Dataset d'entraînement et d'évaluation
├── model/
│   └── cv-lora/            # Dossier généré contenant les poids LoRA (Adapters) sauvegardés
├── cache/                  # Dossier de cache pour les modèles Hugging Face
├── input_cv.json           # Fichier de test contenant les candidats en entrée
└── output_cvs.json         # Fichier généré contenant les résultats d'extraction

🚀 Installation et Prérequis

    Environnement : Ouvrez le notebook projet_cv.ipynb dans Google Colab. Un GPU (ex: T4 ou A100) est requis.

    Google Drive : Montez votre Google Drive et assurez-vous que le chemin PATH_PROJET correspond à votre arborescence.

    Compte Ngrok : Créez un compte gratuit sur Ngrok pour obtenir un Auth Token. Remplacez le token présent dans le code par le vôtre.

⚙️ Utilisation
1. Entraînement du Modèle (Fine-Tuning)

Exécutez la première cellule du notebook pour lancer le processus d'entraînement. Le script va :

    Charger et diviser le dataset (80% train / 20% test).

    Quantifier le modèle Phi-3 en 4-bits.

    Entraîner les adaptateurs LoRA sur 4 époques.

    Sauvegarder les poids dans le dossier model/cv-lora.

2. Lancement de l'API Flask

Exécutez les cellules de configuration et de démarrage du serveur Flask. L'API démarrera en arrière-plan sur le port 5000.
3. Exposition Publique avec Ngrok

Exécutez la cellule contenant la configuration Ngrok. Une URL publique (ex: https://xxx.ngrok-free.dev) sera générée et affichée dans la console.

📡 Documentation de l'API

L'API expose deux routes principales :
GET /health

Vérifie l'état du serveur et la disponibilité du GPU.
POST /cv/generate

Extrait les informations d'un CV textuel.

Requête (Body JSON) :
{
  "input": "Patrick Yoba. Étudiant ingénieur data/IA. Résumé: recherche stage data science. Email: patrick.yoba@email.com. Compétences: Python, SQL, ML. École: 3iL Limoges (2024-)."
}

{
  "informations": {
    "prenom": "Patrick",
    "nom": "Yoba",
    "titre": "Étudiant ingénieur data/IA",
    "email": "patrick.yoba@email.com",
    "telephone": null,
    "adresse": "France",
    "liens": []
  },
  "resume": "Enquête active stages data science.",
  "competences": {
    "techniques": ["Python", "SQL", "ML"],
    "outils": [],
    "soft_skills": []
  },
  "experience": [],
  "education": [
    {
      "ecole": "3iL Limoges",
      "diplome": "Ingénieur data/IA",
      "annee": "2024"
    }
  ],
  "projets": [],
  "langues": []
}

Le Fine-Tuning a démontré d'excellentes capacités à imposer un format de sortie JSON strict au modèle Phi-3 et à classifier correctement les entités (Nom, Compétences, Diplômes).

Axes d'amélioration connus :

    Hallucinations : Le modèle peut parfois inventer des noms d'écoles, d'entreprises ou de diplômes s'ils ne sont pas explicites dans le texte.

    Instabilité syntaxique : De rares variations dans les clés JSON (ex: experece au lieu de experience) ont été observées, nécessitant potentiellement une validation post-traitement (ex: Pydantic) ou un ajustement de la pénalité de répétition (repetition_penalty).

👨‍💻 Auteur

Patrick Yoba
Etudiants 3IL INGENIEURS