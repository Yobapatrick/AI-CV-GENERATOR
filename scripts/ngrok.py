# ============================
from pyngrok import ngrok

#   Remplace par ton propre token : https://dashboard.ngrok.com/get-started/your _token
NGROK_AUTH_TOKEN = "***"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Ferme les tunnels existants avant d'en ouvrir un nouveau
ngrok.kill()

public_url = ngrok.connect(PORT).public_url
print(" URL publique :", public_url)
print("   Health  :", public_url + "/health")
print("   Generate:", public_url + "/cv/generate")