from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from utils.config import Config
import streamlit as st

class ChatManager:
    @staticmethod
    def generate_response(question, vector_store):
        """Génère une réponse à partir d'une question et d'un vector store"""
        if not vector_store:
            return "Aucun document chargé. Veuillez uploader un PDF."
            
        try:
            # Recherche des passages pertinents
            docs = vector_store.similarity_search(question, k=5)
            
            if not docs:
                return "Aucune information pertinente trouvée."
                
            llm = ChatOpenAI(
                openai_api_key=Config.get_openai_key(),
                temperature=0.3,
                model_name="gpt-3.5-turbo"
            )
            
            # Sélection du type de chain
            chain_type = "map_reduce" if "résumé" in question.lower() or "résume" in question.lower() else "stuff"
            chain = load_qa_chain(llm, chain_type=chain_type)
            
            return chain.run(input_documents=docs, question=question)
            
        except Exception as e:
            return f"Erreur lors de l'analyse: {str(e)}"