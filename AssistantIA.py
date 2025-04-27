import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import uuid
from datetime import datetime

# Configuration initiale
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# 🔑 Récupération de la clé API depuis secrets.toml
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Erreur de configuration : {str(e)}")
    st.error("Veuillez configurer votre clé API dans le fichier .streamlit/secrets.toml")
    st.stop()

# Style CSS personnalisé (thème noir)
st.markdown("""
<style>
    /* Style général */
    html, body, .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Zone de chat */
    .message-container {
        margin-bottom: 24px;
    }
    .chat-message {
        padding: 16px 20px;
        border-radius: 12px;
        line-height: 1.6;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        margin-top: 8px;
    }
    .user-message {
        background-color: #1a1a1a;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        border: 1px solid #333333;
    }
    .ai-message {
        background-color: #0d0d0d;
        color: #ffffff !important;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #333333;
    }
    
    /* Zone de saisie fixe */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1.2rem;
        background: #000000;
        border-top: 1px solid #333333;
        z-index: 100;
    }
    .chat-input-box {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #333333 !important;
    }
    .conversation-btn {
        width: 100%;
        text-align: left;
        margin: 8px 0;
        padding: 12px 16px;
        border-radius: 8px;
        background: #1a1a1a;
        border: 1px solid #333333;
        color: white !important;
        transition: all 0.2s;
    }
    .conversation-btn:hover {
        background-color: #262626 !important;
    }
    .conversation-btn.active {
        background-color: #333333 !important;
        font-weight: bold;
    }
    
    /* Textes */
    h1, h2, h3, .stMarkdown {
        color: white !important;
    }
    
    /* Boutons */
    .stButton>button {
        border: 1px solid #333333 !important;
        background-color: #1a1a1a !important;
        color: white !important;
    }
    .stButton>button:hover {
        background-color: #262626 !important;
        border-color: #444444 !important;
    }
    
    /* Uploader */
    .stFileUploader>div {
        border: 1px dashed #333333 !important;
        background: #0a0a0a !important;
        color: white !important;
    }
    
    /* Spinner */
    .stSpinner>div {
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if "conversations" not in st.session_state:
    st.session_state.conversations = {
        "default": {
            "id": str(uuid.uuid4()),
            "title": "Nouvelle conversation",
            "messages": [],
            "documents": [],
            "vector_store": None
        }
    }

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = "default"

# Fonctions utilitaires
def process_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Erreur lors du traitement du PDF: {str(e)}")
        return None
    

    
def generate_response(question, vector_store):
    if not vector_store:
        return "Aucun document n'a été chargé pour cette conversation. Veuillez d'abord uploader un document PDF."
    
    try:
        # Recherche des passages pertinents
        docs = vector_store.similarity_search(question, k=5)  # Augmenter à 5 passages
        
        if not docs:
            return "Je n'ai pas trouvé d'information pertinente dans le document pour répondre à votre demande."
            
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3,  # Réduire pour plus de précision
            model_name="gpt-3.5-turbo"
        )
        
        # Prompt amélioré pour les résumés
        if "résumé" in question.lower() or "résume" in question.lower():
            chain = load_qa_chain(llm, chain_type="map_reduce")  # Meilleur pour les résumés longs
        else:
            chain = load_qa_chain(llm, chain_type="stuff")
            
        response = chain.run(input_documents=docs, question=question)
        return response
        
    except Exception as e:
        return f"Une erreur est survenue lors de l'analyse du document: {str(e)}"
    
# Sidebar - Gestion des conversations
with st.sidebar:
    st.title("📚 Conversations")
    
    # Bouton nouvelle conversation
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
        st.rerun()
    
    # Liste des conversations
    st.divider()
    for conv_id, conv in st.session_state.conversations.items():
        is_active = conv_id == st.session_state.current_conversation
        btn_label = f"🗨️ {conv['title']}" if is_active else f"💬 {conv['title']}"
        if st.button(
            btn_label,
            key=f"conv_{conv_id}",
            help=f"{len(conv['messages'])} messages",
            type="primary" if is_active else "secondary",
            use_container_width=True
        ):
            st.session_state.current_conversation = conv_id
            st.rerun()
    
    # Gestion des documents
    st.divider()
    st.title("📁 Documents")
    uploaded_files = st.file_uploader(
        "Ajouter des PDF",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        current_conv = st.session_state.conversations[st.session_state.current_conversation]
        for file in uploaded_files:
            if file.name not in [doc["name"] for doc in current_conv["documents"]]:
                with st.spinner(f"Traitement de {file.name}..."):
                    vector_store = process_pdf(file)
                    if vector_store:
                        current_conv["documents"].append({
                            "name": file.name,
                            "size": file.size,
                            "uploaded_at": datetime.now().isoformat()
                        })
                        if current_conv["vector_store"]:
                            current_conv["vector_store"].merge_from(vector_store)
                        else:
                            current_conv["vector_store"] = vector_store
                        st.success(f"{file.name} ajouté !")
                    else:
                        st.error(f"Échec du traitement de {file.name}")

# Zone de chat principale
current_conv = st.session_state.conversations[st.session_state.current_conversation]

st.title(current_conv["title"])
st.caption(f"📝 {len(current_conv['messages'])} messages | 📄 {len(current_conv['documents'])} documents")

# Affichage des messages avec espacement
chat_container = st.container()
with chat_container:
    for msg in current_conv["messages"]:
        with st.container():
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="message-container">'
                    f'<div class="chat-message user-message">'
                    f'<strong style="color: #4d90fe;">Vous:</strong><br>{msg["content"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="message-container">'
                    f'<div class="chat-message ai-message">'
                    f'<strong style="color: #34a853;">Assistant:</strong><br>{msg["content"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)  # Espacement

# Zone de saisie fixe
st.markdown('<div class="chat-input-container"><div class="chat-input-box">', unsafe_allow_html=True)
user_input = st.chat_input(
    "Écrivez votre message ici...",
    key="chat_input"
)
st.markdown('</div></div>', unsafe_allow_html=True)

# Traitement du message utilisateur
if user_input:
    # Ajout du message utilisateur
    current_conv["messages"].append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Génération de la réponse
    with st.spinner("L'Assistant réfléchit..."):
        ai_response = generate_response(
            user_input,
            current_conv["vector_store"]
        )
        
        # Ajout de la réponse de l'IA
        current_conv["messages"].append({
            "role": "ai",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
    
    st.rerun()