import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_FILE = "data.txt"

rag_chain = None  # loaded lazily on first /chat request


def build_rag_chain():
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    loader = TextLoader(DATA_FILE)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN,
    )
    retriever = FAISS.from_documents(chunks, embeddings).as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based ONLY on the provided context.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Groq FastAPI RAG Server",
    description=(
        "Production-grade REST API serving a Retrieval-Augmented Generation pipeline. "
        "POST a question to /chat and get a grounded answer from the AI knowledge base, "
        "powered by Groq LPU inference (llama-3.1-8b-instant) and FAISS vector search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Groq RAG Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 640px; margin: 50px auto; }
            h2 { margin-bottom: 4px; }
            p.sub { color: #666; font-size: 0.9em; margin-top: 0; }
            #chatbox { width: 100%; height: 320px; border: 1px solid #ccc; padding: 12px;
                       overflow-y: scroll; margin-bottom: 10px; border-radius: 4px; box-sizing: border-box; }
            #question { width: 82%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 14px; background: #4f46e5; color: #fff; border: none;
                     border-radius: 4px; cursor: pointer; }
            .user-msg { color: #1d4ed8; margin-bottom: 4px; }
            .bot-msg  { color: #166534; margin-bottom: 14px; }
            .err-msg  { color: #dc2626; margin-bottom: 14px; }
        </style>
    </head>
    <body>
        <h2>Groq RAG Server</h2>
        <p class="sub">Ask anything about AI, ML, NLP, LLMs, RAG, or computer vision.</p>
        <div id="chatbox"></div>
        <input type="text" id="question" placeholder="e.g. How does RAG reduce hallucinations?" />
        <button onclick="ask()">Send</button>
        <script>
            async function ask() {
                const input = document.getElementById("question");
                const box   = document.getElementById("chatbox");
                const q = input.value.trim();
                if (!q) return;
                input.value = "";
                box.innerHTML += `<div class="user-msg"><b>You:</b> ${q}</div>`;
                box.innerHTML += `<div class="bot-msg" id="loading"><b>Bot:</b> Thinking…</div>`;
                box.scrollTop = box.scrollHeight;
                try {
                    const res  = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: q })
                    });
                    const data = await res.json();
                    document.getElementById("loading").remove();
                    if (res.ok) {
                        box.innerHTML += `<div class="bot-msg"><b>Bot:</b> ${data.answer}</div>`;
                    } else {
                        box.innerHTML += `<div class="err-msg"><b>Error:</b> ${data.detail}</div>`;
                    }
                } catch {
                    document.getElementById("loading").remove();
                    box.innerHTML += `<div class="err-msg"><b>Error:</b> Could not reach server.</div>`;
                }
                box.scrollTop = box.scrollHeight;
            }
            document.getElementById("question").addEventListener("keydown", e => {
                if (e.key === "Enter") ask();
            });
        </script>
    </body>
    </html>
    """


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse, summary="Ask a question against the knowledge base")
async def chat_endpoint(request: ChatRequest):
    """
    Retrieve relevant chunks from the FAISS index and generate a grounded answer via Groq.
    First call initialises the RAG chain (embeddings + vector index) — expect ~5s latency.
    """
    global rag_chain
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on this server.")
    if rag_chain is None:
        rag_chain = build_rag_chain()
    try:
        answer = rag_chain.invoke(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
