import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

INDEX_NAME = "fastapi-codebase-local"

def test_retrieval(query: str):
    print(f"--- Querying Pinecone for: '{query}' ---")
    
    # 1. Use the exact same local embedding model to vectorize the question
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Connect to the existing Pinecone Index
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(INDEX_NAME)
    
    # 3. Load the index into LlamaIndex
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 4. Create a Retriever (Fetching the top 3 most relevant chunks)
    retriever = index.as_retriever(similarity_top_k=3)
    
    # 5. Execute the query
    nodes = retriever.retrieve(query)
    
    # 6. Display the results
    print(f"\nFound {len(nodes)} relevant code chunks:\n")
    for idx, node in enumerate(nodes, 1):
        print(f"=" * 50)
        print(f"Result #{idx} (Score: {node.get_score():.4f})")
        print(f"File Path: {node.metadata.get('file_path', 'Unknown')}")
        print(f"-" * 50)
        # Display the first 400 characters of the retrieved chunk
        print(f"{node.text[:400]}...\n")

if __name__ == "__main__":
    if "PINECONE_API_KEY" not in os.environ:
        print("ERROR: Please set the PINECONE_API_KEY environment variable.")
        exit(1)
        
    sample_query = "Where is the APIRouter defined and how does routing work?"
    test_retrieval(sample_query)