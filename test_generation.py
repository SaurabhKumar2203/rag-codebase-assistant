import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.postprocessor import SentenceTransformerRerank # <-- NEW IMPORT

INDEX_NAME = "fastapi-codebase-local"

def query_assistant_with_rerank(query: str):
    print(f"--- Asking Gemini (with Reranker): '{query}' ---")
    
    # 1. Local Embeddings for Initial Retrieval
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Gemini LLM for Generation
    Settings.llm = GoogleGenAI(
        model="gemini-3.5-flash", 
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3 
    )
    
    # 3. Connect to Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # 4. Initialize the Index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # 5. Configure the Reranker
    # This grabs a wider net (top 10) from Pinecone, then cross-encoders them down to the best 3
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
        top_n=3
    )
    
    # 6. Create Query Engine with the Reranker attached
    query_engine = index.as_query_engine(
        similarity_top_k=10, # Cast a wider net initially
        node_postprocessors=[reranker] # Refine and sort using the reranker
    )
    
    print("\nRetrieving chunks, reranking for precision, and generating answer...\n")
    
    # 7. Execute the query
    response = query_engine.query(query)
    
    print("=" * 60)
    print("GEMINI'S ANSWER (WITH RERANKING):")
    print("-" * 60)
    print(response.response)
    print("=" * 60)
    
    # Print the reranked source nodes
    print("\nTop Reranked Sources Used:")
    for node in response.source_nodes:
         score = getattr(node, 'score', 0.0)
         print(f"- {node.metadata.get('file_path', 'Unknown')} (Rerank Score: {score:.4f})")

if __name__ == "__main__":
    if "PINECONE_API_KEY" not in os.environ or "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: Please set both PINECONE_API_KEY and GOOGLE_API_KEY.")
        exit(1)
        
    sample_query = "How do I implement basic HTTP Basic Auth in FastAPI based on this codebase?"
    query_assistant_with_rerank(sample_query)