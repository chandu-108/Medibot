import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"

print("Loading embeddings and database...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# Setup HuggingFace client
client = InferenceClient(token=HF_TOKEN)

print("\nReady to answer questions!")
user_query = input("Write Query Here: ")

# Get relevant documents
print("\nRetrieving relevant documents...")
docs = retriever.invoke(user_query)

# Format context
context = "\n\n".join([doc.page_content[:500] for doc in docs])  # Limit context size

# Create prompt
prompt = f"""Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {user_query}

Start the answer directly. No small talk please."""

# Get response from HuggingFace using chat completion
print("Generating answer...")
try:
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(
        messages=messages,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_tokens=512,
        temperature=0.5
    )
    
    answer = response.choices[0].message.content
    
    print("\n" + "="*50)
    print("RESULT:", answer)
    print("="*50)
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nThe HuggingFace Inference API is having issues.")
    print("Please consider:")
    print("1. Installing Ollama (https://ollama.com) for local inference")
    print("2. Using OpenAI API instead")
    print("3. Trying again later")
    print("\nHowever, here are the relevant documents retrieved:")

# Show source documents
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:400])
    print("...")
