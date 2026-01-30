import streamlit as st

def check_password():
    """Renvoie True si le mot de passe est bon."""
    
    # 1. Si déjà connecté
    if st.session_state.get("password_correct", False):
        return True

    # 2. Formulaire de connexion
    st.header("🔒 Accès Administrateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if password == "admin123":  # Mot de passe en dur pour le prototype
            st.session_state["password_correct"] = True
            st.rerun() # Recharge la page
        else:
            st.error("Mot de passe incorrect")
            
    return False