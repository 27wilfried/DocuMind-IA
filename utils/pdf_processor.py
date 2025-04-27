import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

import streamlit as st
from utils.config import Config

class PDFProcessor:
    @staticmethod
    @st.cache_resource(show_spinner=False, max_entries=5)
    def process_pdf(file_path):
        """
        Traite un fichier PDF et retourne un vector store FAISS
        
        Args:
            file_path (str): Chemin vers le fichier PDF
            
        Returns:
            FAISS: Vector store contenant les embeddings du PDF
            None: En cas d'erreur
        """
        try:
            # Validation du fichier
            if not isinstance(file_path, str):
                raise ValueError("Le chemin du fichier doit être une chaîne de caractères")
                
            if not os.path.isfile(file_path):
                raise ValueError(f"Fichier {file_path} introuvable")
                
            # Extraction du texte
            with st.spinner(f"Extraction du texte depuis {os.path.basename(file_path)}..."):
                with open(file_path, "rb") as f:
                    pdf_reader = PdfReader(f)
                    
                    if not pdf_reader.pages:
                        raise ValueError("PDF vide ou corrompu")
                        
                    text = "\n".join(
                        page.extract_text() or "" 
                        for page in pdf_reader.pages
                    )
                    
            if not text.strip():
                raise ValueError("Aucun texte extrait - le PDF est peut-être une image scannée")
                
            # Découpage du texte
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Création des embeddings
            with st.spinner("Création des embeddings..."):
                embeddings = OpenAIEmbeddings(
                    openai_api_key=Config.get_openai_key(),
                    model="text-embedding-3-small"  # Plus rapide et économique
                )
                vector_store = FAISS.from_texts(
                    chunks, 
                    embeddings,
                    metadatas=[{"source": file_path} for _ in chunks]
                )
                
            return vector_store
            
        except Exception as e:
            st.error(f"Erreur lors du traitement du PDF: {str(e)}")
            # Nettoyer le cache en cas d'erreur
            st.cache_resource.clear()
            return None