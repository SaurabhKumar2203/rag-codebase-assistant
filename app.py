import os
import subprocess
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser.text.code import CodeSplitter

# Page Configuration
st.set_page_config(
    page_title="Multi-Repo Codebase Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Repo Codebase Assistant")
st.write("Chat with any public Python GitHub repository on the fly with lightning-fast shallow cloning, core AST chunking, local embeddings, reranking, and Gemini chat memory!")

# Check for API Keys
if "PINECONE_API_KEY" not in os.environ or "GOOGLE_API_KEY" not in os.environ:
    st.error("Missing API Keys! Please ensure PINECONE_API_KEY and GOOGLE_API_KEY are set.")
    st.stop()

INDEX_NAME = "fastapi-codebase-local"

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc = get_pinecone_client()

# --- SIDEBAR: REPOSITORY MANAGER ---
st.sidebar.header("📂 Repository Manager")
repo_url = st.sidebar.text_input(
    "GitHub Repo URL", 
    value="https://github.com/fastapi/fastapi.git",
    help="Paste any public Python GitHub repository URL here."
)

def get_namespace_from_url(url: str):
    clean_url = url.rstrip(".git").strip("/")
    repo_name = clean_url.split("/")[-1]
    return repo_name.lower().replace("-", "_")

current_namespace = get_namespace_from_url(repo_url)
st.sidebar.info(f"Target Namespace: **{current_namespace}**")

def clone_and_index_repo(url: str, namespace: str):
    repo_dir = f"repo_{namespace}"
    
    # 1. Shallow clone (--depth 1) to bypass heavy git commit history downloads
    if not os.path.exists(repo_dir):
        with st.spinner(f"Cloning latest snapshot from {url}..."):
            subprocess.run(["git", "clone", "--depth", "1", url, repo_dir], check=True)
            
    # 2. Read Python files and filter out heavy test/doc/example paths for high performance
    with st.spinner("Scanning and parsing core AST code chunks..."):
        reader = SimpleDirectoryReader(
            input_dir=repo_dir, 
            required_exts=[".py"], 
            recursive=True,
            exclude_hidden=True
        )
        all_docs = reader.load_data()
        
        # Exclude tests, docs, examples, and benchmarks to make loading instantaneous
        excluded_keywords = ["/tests/", "\\tests\\", "/docs/", "\\docs\\", "/examples/", "\\examples\\", "/benchmarks/", "\\benchmarks\\"]
        docs = [
            doc for doc in all_docs 
            if not any(kw in doc.metadata.get("file_path", "") for kw in excluded_keywords)
        ]
        
        splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
        nodes = splitter.get_nodes_from_documents(docs)
        
    # 3. Setup local embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 1024
    
    # Ensure base index exists
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    with st.spinner(f"Embedding {len(nodes)} core chunks and pushing to namespace '{namespace}'..."):
        # Batch size 200 for faster network uploads
        VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False, insert_batch_size=200)
        
    st.sidebar.success(f"Successfully indexed {len(nodes)} core chunks!")

if st.sidebar.button("Load & Index Repository"):
    try:
        clone_and_index_repo(repo_url, current_namespace)
        st.success(f"Repository '{current_namespace}' loaded successfully! You can now start chatting below.")
    except Exception as e:
        st.sidebar.error(f"Error during ingestion: {e}")

# --- CHAT ENGINE INITIALIZATION ---
@st.cache_resource
def get_chat_engine(namespace: str):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = GoogleGenAI(
        model="gemini-3.5-flash", 
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3
    )
    
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
        top_n=3
    )
    
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=10,
        node_postprocessors=[reranker],
        verbose=True
    )

try:
    chat_engine = get_chat_engine(current_namespace)
except Exception:
    st.warning("👈 Please click **'Load & Index Repository'** in the sidebar to initialize this codebase.")
    st.stop()

# --- SESSION STATE & CHAT UI ---
if "active_namespace" not in st.session_state or st.session_state.active_namespace != current_namespace:
    st.session_state.active_namespace = current_namespace
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am ready to help you explore the **{current_namespace}** repository. What would you like to know?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask a question about {current_namespace}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing codebase and thinking..."):
            response = chat_engine.chat(prompt)
            assistant_response = response.response
            
            st.markdown(assistant_response)
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                with st.expander("📁 View Source Files Used"):
                    for node in response.source_nodes:
                        score = getattr(node, 'score', 0.0)
                        file_path = node.metadata.get('file_path', 'Unknown')
                        st.markdown(f"- **{file_path}** *(Rerank Score: {score:.4f})*")

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})