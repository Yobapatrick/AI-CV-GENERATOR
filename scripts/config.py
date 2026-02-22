import os

PATH_PROJET = os.getenv("PATH_PROJET", "./")
DOSSIER_CACHE = os.path.join(PATH_PROJET, "cache")

BASE_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_DIR = os.path.join(PATH_PROJET, "lora_adapter")

PORT = 5000