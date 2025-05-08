from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import ollama  # ✅ Ollama local model call

app = Flask(__name__)
load_dotenv()

# Load API key if needed
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download local embeddings
embeddings = download_hugging_face_embeddings()

# Set up vector store
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Optional: template for logging/debugging
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    print("🧑 User Input:", msg)

    try:
        # Step 1: Retrieve top 3 docs from vector store
        retrieved_docs = retriever.invoke(msg)
        print("📚 Retrieved Docs:", retrieved_docs)

        if not retrieved_docs:
            return "Sorry, I couldn't find any relevant medical information."

        # Step 2: Build context from documents (internal only)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        print("📄 Context:\n", context)

        # Step 3: Create final prompt (internal context included)
        final_prompt = f"""
You are a helpful and accurate medical assistant.

Context:
{context}

Question: {msg}
Answer:
        """.strip()

        print("🧠 Final Prompt Sent to Ollama:\n", final_prompt)

        # Step 4: Query Ollama locally
        response = ollama.chat(model="tinyllama", messages=[
            {"role": "user", "content": final_prompt}
        ])

        print("📝 Ollama Response:", response)

        # Step 5: Return only the actual answer content
        content = response["message"]["content"]
        if "Answer:" in content:
            content = content.split("Answer:")[-1].strip()

        return content

    except Exception as e:
        print("❌ Error generating response:", e)
        return "Sorry, something went wrong."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5151, debug=True)

