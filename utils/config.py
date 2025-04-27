import os
import streamlit as st

# Configuration de l'application
class Config:
    PAGE_TITLE = "PDF AI Assistant"
    PAGE_ICON = "🤖"
    MAX_FILE_SIZE = 10_000_000  # 10MB
    DEFAULT_CONVERSATION_NAME = "Nouvelle conversation"
    
    @staticmethod
    def get_openai_key():
        try:
            return st.secrets["OPENAI_API_KEY"]
        except Exception as e:
            st.error(f"Erreur de configuration : {str(e)}")
            st.error("Veuillez configurer votre clé API dans le fichier .streamlit/secrets.toml")
            st.stop()