import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import hashlib

load_dotenv(override=True)

class RAGEngine:
    def __init__(self):
        # Configuration
        self.google_api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyAlzaj6p8s-VTtH5dHEjsq1MpoEba6Qzls"
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Determine if we are using Cloud or Local
        if not self.qdrant_url:
            self.qdrant_url = "http://localhost:6333"
            print("Using Local Qdrant (Env var QDRANT_URL not found)")
        else:
            print(f"Using Qdrant Cloud: {self.qdrant_url}")

        # Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        
        # Initialize LLM Client (Gemini via OpenAI compat)
        self.client = OpenAI(
            api_key=self.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        self.collection_name = "real_world_rag_collection"

    def get_vector_store(self):
        """Get the Qdrant Vector Store instance."""
        try:
            return QdrantVectorStore.from_existing_collection(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                prefer_grpc=bool(self.qdrant_api_key) # Use GRPC if using Cloud/Key
            )
        except Exception as e:
            # If collection doesn't exist, this might fail, handled in indexing usually
            print(f"Error getting vector store: {e}")
            return None

    def index_file(self, file_path: str):
        """Index a PDF file into the vector store."""
        print(f"Indexing file: {file_path}")
        
        # 0. Check for duplicates
        file_hash = self._compute_md5(file_path)
        if self._file_already_indexed(file_hash):
             return {"status": "skipped", "message": "File already indexed!"}

        # 1. Load PDF
        loader = PyPDFLoader(file_path=file_path)
        docs = loader.load()
        
        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Add metadata
        for doc in split_docs:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["source"] = os.path.basename(file_path)

        # 3. Add to Qdrant (creates collection if needed)
        try:
            QdrantVectorStore.from_documents(
                documents=split_docs,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                prefer_grpc=bool(self.qdrant_api_key)
            )
            return {"status": "success", "chunks": len(split_docs)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _compute_md5(self, file_path: str):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _file_already_indexed(self, file_hash: str):
        # We need a qdrant client to perform filter search
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=bool(self.qdrant_api_key)
            )
            
            # Check if any point exists with this file_hash
            try:
                res, _ = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.file_hash",
                                match=models.MatchValue(value=file_hash)
                            )
                        ]
                    ),
                    limit=1
                )
                return len(res) > 0
            except:
                return False # Collection might not exist
        except Exception as e:
            print(f"Error checking duplicate: {e}")
            return False

    def chat(self, query: str):
        """Chat with the document context."""
        vector_store = self.get_vector_store()
        if not vector_store:
            return "Error: Vector store not initialized or collection not found. Please upload a file first."
            
        # Search for context
        results = vector_store.similarity_search(query, k=5)
        
        context = "\n\n".join([
            f"Page {doc.metadata.get('page_label', '?')}: {doc.page_content}" 
            for doc in results
        ])
        
        system_prompt = f"""You are a helpful assistant. Answer based on the context below.
        If you find the answer, mention the page number.
        
        Context:
        {context}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    def chat_stream(self, query: str):
        """Chat with streaming response and citations."""
        vector_store = self.get_vector_store()
        if not vector_store:
            yield {"type": "error", "content": "Vector store not initialized."}
            return

        # 1. Retrieve
        results = vector_store.similarity_search(query, k=5)
        
        # Yield Sources first
        sources = [
            {"page": doc.metadata.get("page_label", "?"), "text": doc.page_content[:200] + "...", "source": doc.metadata.get("source", "doc")} 
            for doc in results
        ]
        yield {"type": "sources", "content": sources}
        
        context = "\n\n".join([
            f"Page {doc.metadata.get('page_label', '?')}: {doc.page_content}" 
            for doc in results
        ])
        
        system_prompt = f"""You are a helpful assistant. Answer based on the context below.
        If you find the answer, mention the page number.
        
        Context:
        {context}
        """
        
        # 2. Generate Stream
        try:
            stream = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {"type": "chunk", "content": chunk.choices[0].delta.content}
                    
        except Exception as e:
            yield {"type": "error", "content": f"Error generating response: {str(e)}"}

    def clear_database(self):
        """Delete the collection to free up space."""
        try:
             # Initialize client if needed (QdrantVectorStore doesn't expose client easily, so we use qdrant_client directly)
            from qdrant_client import QdrantClient
            
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=bool(self.qdrant_api_key)
            )
            
            client.delete_collection(self.collection_name)
            return "Database cleared successfully!"
        except Exception as e:
            return f"Error clearing database: {str(e)}"

    def get_collection_stats(self):
        """Get stats about the collection (e.g. vector count)."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=bool(self.qdrant_api_key)
            )
            # Check if collection exists first
            collections = client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                return {"status": "Empty", "vectors": 0}
            
            info = client.get_collection(self.collection_name)
            return {"status": "Active", "vectors": info.points_count}
        except Exception as e:
            return {"status": "Error", "error": str(e)}

    def get_db_connection(self):
        """Get a connection to the PostgreSQL database."""
        try:
            import psycopg2
            
            conn_str = os.getenv("POSTGRES_URL")
            if not conn_str:
                return None
                
            conn = psycopg2.connect(conn_str)
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
