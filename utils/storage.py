import json
import os
from datetime import datetime
import streamlit as st

class ConversationStorage:
    @staticmethod
    def save_conversations():
        """Sauvegarde les conversations et gère les fichiers PDF"""
        if 'conversations' not in st.session_state:
            return

        # Créer le dossier temporaire s'il n'existe pas
        os.makedirs("temp_pdfs", exist_ok=True)

        # Préparer les données à sauvegarder
        conversations_to_save = {}
        
        for conv_id, conv in st.session_state.conversations.items():
            # Créer une copie sérialisable sans le vector_store
            conv_copy = {
                "id": conv["id"],
                "title": conv["title"],
                "messages": conv["messages"],
                "documents": []
            }

            # Sauvegarder les métadonnées des documents
            for doc in conv.get("documents", []):
                doc_copy = {
                    "name": doc["name"],
                    "size": doc["size"],
                    "uploaded_at": doc["uploaded_at"],
                    "file_path": doc.get("file_path", "")
                }
                conv_copy["documents"].append(doc_copy)

            conversations_to_save[conv_id] = conv_copy

        # Sauvegarder dans le fichier JSON
        with open('conversations.json', 'w') as f:
            json.dump(conversations_to_save, f, default=str)

    @staticmethod
    def load_conversations():
        """Charge les conversations et prépare la reconstruction des vector stores"""
        if not os.path.exists('conversations.json'):
            return None

        with open('conversations.json', 'r') as f:
            conversations = json.load(f)
            
            # Convertir les dates et préparer les vector stores
            for conv in conversations.values():
                # Conversion des timestamps
                for msg in conv['messages']:
                    if isinstance(msg['timestamp'], str):
                        try:
                            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                        except ValueError:
                            msg['timestamp'] = datetime.now()
                
                # Initialiser le vector_store pour reconstruction ultérieure
                conv['vector_store'] = None
                
                # Vérifier les chemins des fichiers PDF
                for doc in conv.get('documents', []):
                    if 'file_path' not in doc:
                        doc['file_path'] = os.path.join("temp_pdfs", doc["name"])

            return conversations

    @staticmethod
    def cleanup_old_files():
        """Nettoie les fichiers PDF orphelins"""
        if not os.path.exists('conversations.json') or not os.path.exists('temp_pdfs'):
            return

        # Récupérer tous les fichiers PDF référencés
        referenced_files = set()
        with open('conversations.json', 'r') as f:
            conversations = json.load(f)
            for conv in conversations.values():
                for doc in conv.get('documents', []):
                    if 'file_path' in doc:
                        referenced_files.add(doc['file_path'])

        # Supprimer les fichiers non référencés
        for filename in os.listdir('temp_pdfs'):
            filepath = os.path.join('temp_pdfs', filename)
            if filepath not in referenced_files and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    st.error(f"Erreur lors de la suppression de {filename}: {str(e)}")