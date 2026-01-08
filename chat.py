from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

# client = OpenAI()
client = OpenAI(
    api_key="AIzaSyAlzaj6p8s-VTtH5dHEjsq1MpoEba6Qzls",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key="AIzaSyAlzaj6p8s-VTtH5dHEjsq1MpoEba6Qzls"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors_gemini_native",
    embedding=embedding_model
)

# Take User Query
query = input("> ")

# Vector Similarity Search [query] in DB
search_results = vector_db.similarity_search(
    query=query
)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
"""

chat_completion = client.chat.completions.create(
    model="gemini-2.5-flash",
    # model="gpt-4.1",
    messages=[
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": query },
    ]
)

print(f"🤖: {chat_completion.choices[0].message.content}")

