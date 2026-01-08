from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Loading
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()  # Read PDF File

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

split_docs = text_splitter.split_documents(documents=docs)

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key="AIzaSyAlzaj6p8s-VTtH5dHEjsq1MpoEba6Qzls"
)

# Using [embedding_model] create embeddings of [split_docs] and store in DB

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_vectors_gemini_native",
    embedding=embedding_model
)

print("Indexing of Documents Done...")
