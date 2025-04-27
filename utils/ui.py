import streamlit as st
import uuid
import os
from datetime import datetime
from utils.config import Config
from utils.storage import ConversationStorage
from utils.pdf_processor import PDFProcessor

class UI:
    @staticmethod
    def setup_page():
        """Configure la page Streamlit avec persistance des données"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout="centered"
        )
        
        # Chargement du CSS
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    @staticmethod
    def init_session_state():
        """Initialise l'état de session avec chargement des conversations sauvegardées"""
        if "conversations" not in st.session_state:
            saved_conversations = ConversationStorage.load_conversations()
            
            if saved_conversations:
                st.session_state.conversations = saved_conversations
                # Reconstruire les vector stores à partir des PDFs sauvegardés
                for conv in st.session_state.conversations.values():
                    if conv["documents"]:
                        conv["vector_store"] = None
                        for doc in conv["documents"]:
                            file_path = os.path.join("temp_pdfs", doc["name"])
                            if os.path.exists(file_path):
                                try:
                                    vector_store = PDFProcessor.process_pdf(file_path)
                                    if vector_store:
                                        if conv["vector_store"]:
                                            conv["vector_store"].merge_from(vector_store)
                                        else:
                                            conv["vector_store"] = vector_store
                                except Exception as e:
                                    st.error(f"Erreur lors du rechargement de {doc['name']}: {str(e)}")
            else:
                st.session_state.conversations = {
                    "default": {
                        "id": str(uuid.uuid4()),
                        "title": Config.DEFAULT_CONVERSATION_NAME,
                        "messages": [],
                        "documents": [],
                        "vector_store": None
                    }
                }

        if "current_conversation" not in st.session_state:
            st.session_state.current_conversation = "default"

        # Créer le dossier temp_pdfs s'il n'existe pas
        os.makedirs("temp_pdfs", exist_ok=True)


    @staticmethod
    def render_sidebar():
        """Affiche la sidebar avec gestion persistante des conversations"""
        with st.sidebar:
            st.title("📚 Conversations")
            
            if st.button("➕ Nouvelle conversation", use_container_width=True, key="new_chat"):
                conv_id = str(uuid.uuid4())
                conv_title = f"Conversation {len(st.session_state.conversations)}"
                st.session_state.conversations[conv_id] = {
                    "id": conv_id,
                    "title": conv_title,
                    "messages": [],
                    "documents": [],
                    "vector_store": None
                }
                st.session_state.current_conversation = conv_id
                ConversationStorage.save_conversations()
                st.rerun()
            
            st.divider()
            for conv_id, conv in st.session_state.conversations.items():
                is_active = conv_id == st.session_state.current_conversation
                btn_label = f"🗨️ {conv['title']}" if is_active else f"💬 {conv['title']}"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        btn_label,
                        key=f"conv_{conv_id}",
                        help=f"{len(conv['messages'])} messages | {len(conv['documents'])} docs",
                        use_container_width=True
                    ):
                        st.session_state.current_conversation = conv_id
                        st.rerun()
                
                with col2:
                    if st.button(
                        "×", 
                        key=f"delete_{conv_id}",
                        help="Supprimer cette conversation",
                        type="secondary"
                    ):
                        if len(st.session_state.conversations) > 1:
                            # Supprimer les fichiers PDF associés
                            for doc in conv["documents"]:
                                file_path = os.path.join("temp_pdfs", doc["name"])
                                if os.path.exists(file_path):
                                    try:
                                        os.remove(file_path)
                                    except Exception as e:
                                        st.error(f"Erreur suppression {doc['name']}: {str(e)}")
                            
                            del st.session_state.conversations[conv_id]
                            if st.session_state.current_conversation == conv_id:
                                st.session_state.current_conversation = next(iter(st.session_state.conversations))
                            ConversationStorage.save_conversations()
                            st.rerun()
                        else:
                            st.warning("Vous ne pouvez pas supprimer la dernière conversation")
            
            st.divider()
            st.title("📁 Documents")
            uploaded_files = st.file_uploader(
                "Ajouter des PDF",
                type="pdf",
                accept_multiple_files=True,
                key="file_uploader"
            )
            
            return uploaded_files

    @staticmethod
    def render_chat():
        """Affiche la zone de chat principale avec historique persisté"""
        current_conv = st.session_state.conversations[st.session_state.current_conversation]

        # En-tête avec titre et bouton de renommage
        col_title, col_rename = st.columns([4, 1])
        with col_title:
            st.title(current_conv["title"])
        with col_rename:
            if st.button("✏️ Renommer"):
                new_title = st.text_input("Nouveau nom", value=current_conv["title"])
                if new_title and new_title != current_conv["title"]:
                    current_conv["title"] = new_title
                    ConversationStorage.save_conversations()
                    st.rerun()

        st.caption(f"📝 {len(current_conv['messages'])} messages | 📄 {len(current_conv['documents'])} documents")

        # Affichage des messages
        chat_container = st.container()
        with chat_container:
            for msg in current_conv["messages"]:
                with st.container():
                    role_class = "user-message" if msg["role"] == "user" else "ai-message"
                    role_name = "Vous" if msg["role"] == "user" else "Assistant"
                    role_color = "#4d90fe" if msg["role"] == "user" else "#34a853"
                    
                    # Gestion robuste du timestamp
                    try:
                        if isinstance(msg["timestamp"], datetime):
                            timestamp = msg["timestamp"].strftime("%H:%M")
                        elif isinstance(msg["timestamp"], str):
                            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M")
                        else:
                            timestamp = datetime.now().strftime("%H:%M")
                    except (ValueError, TypeError, AttributeError):
                        timestamp = datetime.now().strftime("%H:%M")
                    
                    st.markdown(
                        f'<div class="message-container">'
                        f'<div class="chat-message {role_class}">'
                        f'<div style="display: flex; justify-content: space-between;">'
                        f'<strong style="color: {role_color};">{role_name}</strong>'
                        f'<small style="color: #666;">{timestamp}</small>'
                        f'</div>'
                        f'<div style="margin-top: 8px;">{msg["content"]}</div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="height: 16px;"></div>',
                        unsafe_allow_html=True
                    )

        # Zone de saisie
        st.markdown('<div class="chat-input-container"><div class="chat-input-box">', unsafe_allow_html=True)
        user_input = st.chat_input("Écrivez votre message ici...", key="chat_input")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        return user_input