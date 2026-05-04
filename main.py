import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment logic reliably
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_FILE = "data.txt"

# --- TASK 6: INTEGRATION ---
# Core logic bridging the local database into an operational Groq deployment for backend endpoints
def build_deployment_rag():
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant", 
            api_key=GROQ_API_KEY or "dummy_key" 
        )
    except Exception:
        return None
        
    if not os.path.exists(DATA_FILE):
        return None
        
    loader = TextLoader(DATA_FILE)
    docs = loader.load()
    split_docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN,
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Prompt template for Groq RAG specifically tuned for integration 
    rag_prompt = ChatPromptTemplate.from_template("""
    Answer the following question based ONLY on the provided context.
    Context: {context}
    
    Question: {question}
    """)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = None  # loaded lazily on first /chat request

# --- TASK 5: CREATING A FAST API APPLICATION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

# Utilizing the app structure specifically engineered for serving AI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Gen AI - Task 5 Server Initialization", lifespan=lifespan)

@app.get("/")
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Groq RAG Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; }
            #chatbox { width: 100%; height: 300px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
            #question { width: 80%; padding: 10px; }
            button { padding: 10px; width: 15%; }
            .user-msg { color: blue; margin-bottom: 5px; }
            .bot-msg { color: green; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <h2>Groq RAG Chatbot (Task 7 Integration)</h2>
        <div id="chatbox"></div>
        <input type="text" id="question" placeholder="Ask a question about the document..."/>
        <button onclick="askQuestion()">Send</button>

        <script>
            async function askQuestion() {
                const questionInput = document.getElementById("question");
                const chatbox = document.getElementById("chatbox");
                
                const q = questionInput.value;
                if (!q) return;
                
                chatbox.innerHTML += `<div class="user-msg"><b>You:</b> ${q}</div>`;
                questionInput.value = "";
                
                chatbox.innerHTML += `<div class="bot-msg" id="loading"><b>Bot:</b> Thinking...</div>`;
                
                try {
                    const response = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: q })
                    });
                    
                    const data = await response.json();
                    document.getElementById("loading").remove();
                    
                    if (response.ok) {
                        chatbox.innerHTML += `<div class="bot-msg"><b>Bot:</b> ${data.answer}</div>`;
                    } else {
                        chatbox.innerHTML += `<div class="bot-msg" style="color:red;"><b>Error:</b> ${data.detail}</div>`;
                    }
                } catch (error) {
                    document.getElementById("loading").remove();
                    chatbox.innerHTML += `<div class="bot-msg" style="color:red;"><b>Error:</b> Failed to reach server.</div>`;
                }
                
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# --- TASK 6: CONTINUED (INTEGRATING POST ENDPOINT) ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global rag_chain
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise HTTPException(status_code=500, detail="WARNING: GROQ_API_KEY is missing! Follow the README to add your key to the .env file.")

    if rag_chain is None:
        rag_chain = build_deployment_rag()
    if not rag_chain:
        raise HTTPException(status_code=500, detail="WARNING: RAG Integration failed to build. Ensure data.txt is configured.")
        
    try:
        # Pushing integration directly through dynamic LCEL execution mapped to FastApi return objects
        answer = rag_chain.invoke(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- TASK 5 and TASK 6 (Deployment App Logic) ---
if __name__ == "__main__":
    import uvicorn
    # Initializing deployment listener
    uvicorn.run(app, host="0.0.0.0", port=8000)
