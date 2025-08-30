# backend/ingest.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from database import clear_documents, insert_document

load_dotenv()

# --- Paths and Connection Details ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "documents")

def ingest_documents_to_postgresql():
    """Loads documents, splits them, creates embeddings, and uploads them to PostgreSQL with pgvector."""
    
    print("Starting document ingestion to PostgreSQL...")
    
    print(f"Loading documents from: {DOCUMENTS_PATH}")
    
    # Create documents directory if it doesn't exist
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        print(f"Created documents directory: {DOCUMENTS_PATH}")
        print("Please add PDF files to the 'documents' folder and run the script again.")
        return
    
    try:
        loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        documents = loader.load()

        if not documents:
            print("No documents found. Please add PDF files to the 'documents' folder.")
            return

        print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

        print("Creating embeddings and uploading to PostgreSQL... (This may take a moment)")
        
        # Initialize OpenAI embeddings with error handling
        try:
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
            print("OpenAI embeddings initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI embeddings: {e}")
            print("Please ensure your OpenAI API key is set in the environment")
            return

        # Clear existing documents
        print("Clearing existing documents...")
        if not clear_documents():
            print("Failed to clear existing documents. Continuing anyway...")

        # Process and store each document chunk
        successful_inserts = 0
        for i, doc in enumerate(docs):
            try:
                # Create embedding for the document
                embedding = embeddings_model.embed_query(doc.page_content)
                
                # Prepare metadata
                metadata = {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": i
                }
                
                # Insert into PostgreSQL
                doc_id = insert_document(
                    content=doc.page_content,
                    metadata=metadata,
                    embedding=embedding
                )
                
                if doc_id:
                    successful_inserts += 1
                    if i % 10 == 0:  # Progress update every 10 documents
                        print(f"Processed {i+1}/{len(docs)} chunks...")
                else:
                    print(f"Failed to insert document chunk {i+1}")
                    
            except Exception as e:
                print(f"Error processing document chunk {i+1}: {e}")
                continue
        
        print(f"Ingestion complete! Successfully uploaded {successful_inserts}/{len(docs)} document chunks to PostgreSQL.")
        
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        return

if __name__ == "__main__":
    ingest_documents_to_postgresql()