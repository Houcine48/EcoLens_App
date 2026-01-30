import streamlit as st
import subprocess
import sys

def run_script(script_name):
    """Lance un script python et capture la sortie"""
    try:
        # Exécute la commande : python script_name.py
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def show_retraining_page():
    st.title("⚙️ Centre de Commande (MLOps)")
    st.write("Lancez les pipelines manuellement depuis cette interface.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Pipeline de Données")
        st.info("Scan du Drive, nettoyage et génération du rapport CSV.")
        if st.button("🚀 Lancer ETL & Audit"):
            with st.spinner("Exécution de data_eng.py..."):
                success, logs = run_script("data_eng.py")
                if success:
                    st.success("ETL Terminé !")
                    st.expander("Voir les logs").code(logs)
                else:
                    st.error("Erreur durant l'ETL")
                    st.error(logs)

    with col2:
        st.subheader("2. Pipeline d'Entraînement")
        st.warning("⚠️ Attention : Nécessite un GPU. Peut être lent en local.")
        if st.button("🔥 Lancer Entraînement (Train)"):
            with st.spinner("Entraînement en cours... (Patience)"):
                success, logs = run_script("models/train.py")
                if success:
                    st.success("Modèle mis à jour !")
                    st.expander("Voir les logs").code(logs)
                else:
                    st.error("Erreur durant l'entraînement")
                    st.error(logs)