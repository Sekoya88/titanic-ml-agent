#!/bin/bash
# Script pour lancer l'application Streamlit

echo "🚢 Lancement de l'application Titanic ML Agent Dashboard..."

# Vérifier si streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit non trouvé. Installation en cours..."
    pip3 install streamlit plotly --break-system-packages
fi

echo "🌐 Ouverture de l'application dans le navigateur..."
echo "📡 URL: http://localhost:8501"
echo ""
echo "Pour arrêter l'application: Ctrl+C"
echo ""

# Lancer streamlit
streamlit run streamlit_app.py 