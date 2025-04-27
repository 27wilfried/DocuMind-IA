from datetime import datetime
import streamlit as st
import os
from utils.ui import UI
from utils.pdf_processor import PDFProcessor
from utils.chat_manager import ChatManager
from utils.storage import ConversationStorage


def handle_file_uploads(uploaded_files):
    """Gère l'upload et le traitement des fichiers PDF"""
    if not uploaded_files:
        return
    
    current_conv = st.session_state.conversations[st.session_state.current_conversation]
    
    for file in uploaded_files:
        # Vérification que c'est bien un objet fichier Streamlit
        if not hasattr(file, 'name') or not hasattr(file, 'read'):
            st.error(f"Format de fichier invalide pour {file}")
            continue
            
        try:
            # Lire le contenu du fichier une seule fois
            file_bytes = file.getvalue()
            file_size = len(file_bytes)
            
            # Vérifier si le document existe déjà
            if any(doc["name"] == file.name for doc in current_conv["documents"]):
                st.warning(f"{file.name} existe déjà")
                continue
                
            with st.spinner(f"Traitement de {file.name}..."):
                # Sauvegarder temporairement
                os.makedirs("temp_pdfs", exist_ok=True)
                file_path = os.path.join("temp_pdfs", file.name)
                
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                
                # Traiter le PDF
                new_vector_store = PDFProcessor.process_pdf(file_path)
                
                if new_vector_store:
                    # Mettre à jour le vector store
                    if current_conv["vector_store"]:
                        current_conv["vector_store"].merge_from(new_vector_store)
                    else:
                        current_conv["vector_store"] = new_vector_store
                    
                    # Enregistrer les métadonnées
                    current_conv["documents"].append({
                        "name": file.name,
                        "size": file_size,
                        "uploaded_at": datetime.now().isoformat(),
                        "file_path": file_path
                    })
                    
                    st.success(f"✅ {file.name} prêt à l'utilisation")
                    ConversationStorage.save_conversations()
                else:
                    st.error(f"Échec du traitement pour {file.name}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
        except Exception as e:
            st.error(f"Erreur avec {file.name}: {str(e)}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

                
def handle_user_message(user_input):
    """Gère les interactions de chat"""
    if not user_input:
        return
    
    current_conv = st.session_state.conversations[st.session_state.current_conversation]
    
    # Ajouter le message utilisateur
    current_conv["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Vérifier la disponibilité des documents
    if not current_conv.get("vector_store"):
        # Tenter de reconstruire le vector store si des documents existent
        if current_conv.get("documents"):
            with st.spinner("Chargement des documents..."):
                try:
                    for doc in current_conv["documents"]:
                        if os.path.exists(doc.get("file_path", "")):
                            vector_store = PDFProcessor.process_pdf(doc["file_path"])
                            if vector_store:
                                if current_conv["vector_store"]:
                                    current_conv["vector_store"].merge_from(vector_store)
                                else:
                                    current_conv["vector_store"] = vector_store
                except Exception as e:
                    st.error(f"Erreur de chargement: {str(e)}")
    
    # Générer la réponse
    with st.spinner("L'Assistant analyse..."):
        try:
            if current_conv.get("vector_store"):
                ai_response = ChatManager.generate_response(
                    user_input, 
                    current_conv["vector_store"]
                )
            else:
                ai_response = "Aucun document valide n'a pu être chargé. Veuillez vérifier vos fichiers PDF."
            
            current_conv["messages"].append({
                "role": "ai",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            ConversationStorage.save_conversations()
            
        except Exception as e:
            current_conv["messages"].append({
                "role": "ai",
                "content": f"Erreur: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    st.rerun()

def main():
    # Initialisation
    UI.setup_page()
    UI.init_session_state()
    ConversationStorage.cleanup_old_files()
    
    # Interface
    uploaded_files = UI.render_sidebar()
    user_input = UI.render_chat()
    
    # Interactions
    handle_file_uploads(uploaded_files)
    handle_user_message(user_input)

if __name__ == "__main__":
    main()