# ============================
# 8. TEST DE L'API
# ============================
import requests, time, json

BASE = public_url.strip()

# --- Health check ---
print("=== Health Check ===")
r_health = requests.get(BASE + "/health", timeout=15)
print("Status :", r_health.status_code)
print("Body   :", r_health.json())

# --- Génération CV ---
print("\n=== Test Génération CV ===")
test_inputs = [
    "Patrick Yoba. Étudiant ingénieur data/IA. Résumé: recherche stage data science. "
    "Email: patrick.yoba@email.com. Compétences: Python, SQL, ML. "
    "École: 3iL Limoges (2024-).",

    "Inès Wagner. Maître-Nageur Sauveteur. Tel: 0467112233. Adresse: Montpellier. "
    "Exp: Piscine Municipale (2022-2024). Compétences: Secourisme, Aquagym. "
    "Diplôme: BPJEPS AAN (2021).",
]

for i, text in enumerate(test_inputs, 1):
    print(f"\n--- Test {i} ---")
    print("Input:", text[:80], "...")
    try:
        r = requests.post(
            BASE + "/cv/generate",
            json={"input": text},
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        print("Status:", r.status_code)
        if r.status_code == 200:
            cv = r.json()
            print("Nom   :", cv.get("informations", {}).get("nom", "N/A"))
            print("Titre :", cv.get("informations", {}).get("titre", "N/A"))
            print("Résumé:", str(cv.get("resume", ""))[:120])
            print("✅ CV généré avec succès")
        else:
            print("❌ Erreur:", r.json())
    except Exception as e:
        print("❌ Exception:", e)
