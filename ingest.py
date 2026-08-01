import os
import subprocess
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser.text.code import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings

REPO_URL = "https://github.com/fastapi/fastapi.git"
REPO_DIR = "fastapi_repo"
INDEX_NAME = "fastapi-codebase-local" # New index name

def clone_repository():
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL} into {REPO_DIR}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    else:
        print(f"Directory '{REPO_DIR}' already exists. Skipping clone.")

def load_python_files():
    print("Scanning and loading Python files...")
    reader = SimpleDirectoryReader(
        input_dir=REPO_DIR,
        required_exts=[".py"],
        recursive=True
    )
    return reader.load_data()

def chunk_code(documents):
    print("Parsing AST and chunking code files...")
    splitter = CodeSplitter(
        language="python",
        chunk_lines=40,
        chunk_lines_overlap=15,
        max_chars=1500
    )
    return splitter.get_nodes_from_documents(documents)

def store_in_pinecone(nodes):
    print("Initializing Local Embeddings and Pinecone...")
    
    # 1. Use a local embedding model (NO API LIMITS!)
    # all-MiniLM-L6-v2 is small, fast, and outputs 384 dimensions
    print("Downloading local embedding model (first time only)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Setting a higher chunk size since we process locally
    Settings.chunk_size = 1024 
    
    # 2. Setup Pinecone Client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # 3. Handle Index Creation (384 Dimensions for local model)
    if INDEX_NAME in pc.list_indexes().names():
        index_info = pc.describe_index(INDEX_NAME)
        if index_info.dimension != 384:
            print(f"Deleting old Pinecone index '{INDEX_NAME}' with wrong dimension...")
            pc.delete_index(INDEX_NAME)
            
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{INDEX_NAME}' with 384 dimensions...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"Total chunks to upload: {len(nodes)}")
    print("Embedding locally and uploading to Pinecone. (No rate limits!)")
    
    # 4. Generate embeddings and upload
    # Since we are local, we can let LlamaIndex handle the batching again
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=100
    )
    print("Upload complete! The database is populated.")

if __name__ == "__main__":
    print("--- Starting Phase 3: Local Vector Storage ---")
    
    if "PINECONE_API_KEY" not in os.environ:
        print("ERROR: Please set the PINECONE_API_KEY environment variable.")
        exit(1)
        
    clone_repository()
    docs = load_python_files()
    chunks = chunk_code(docs)
    
    store_in_pinecone(chunks)
    print("Phase 3 is complete. The ingestion pipeline is finished!")