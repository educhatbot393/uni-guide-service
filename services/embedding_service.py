import os
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import Config
from services.document_service import process_documents

# Global vector store instance
_vectorstore = None

def initialize_pinecone():
    """Initialize Pinecone vector database connection"""
    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not set")
        
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if Config.PINECONE_INDEX_NAME not in existing_indexes:
        try:
            # Create index if it doesn't exist
            pc.create_index(
                name=Config.PINECONE_INDEX_NAME,
                dimension=Config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Pinecone index created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return False
    
    return True

def initialize_vectorstore(callback=None):
    """
    Load and process data with Pinecone integration
    
    Args:
        callback: Optional callback function for status updates
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _vectorstore
    
    # Return immediately if vectorstore is already initialized
    if _vectorstore:
        print("Vector store already initialized in memory.")
        return True
        
    # Initialize Pinecone connection first
    if not initialize_pinecone():
        print("Failed to initialize Pinecone connection.")
        return False
    
    # Set up embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if vectors already exist in Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    
    try:
        # Check if vectors exist in the namespace
        stats = index.describe_index_stats()
        vector_count = stats.namespaces.get(Config.PINECONE_NAMESPACE, {}).get("vector_count", 0)
        
        if vector_count > 0:
            # Vectors already exist in Pinecone - just connect to them
            print(f"Found {vector_count} existing vectors in Pinecone. Loading...")
            if callback:
                callback(f"Found {vector_count} existing vectors in Pinecone. Loading...")
                
            # Connect to existing vectors without processing documents
            _vectorstore = PineconeVectorStore(
                index_name=Config.PINECONE_INDEX_NAME,
                embedding=embeddings,
                namespace=Config.PINECONE_NAMESPACE
            )
            print("Successfully connected to existing vectors.")
            return True
        else:
            print("No existing vectors found in Pinecone. Need to process documents.")
    except Exception as e:
        print(f"Error checking Pinecone stats: {e}")
        # Continue with data processing if there's an error
    
    # If we get here, we need to process the data and create embeddings
    print("Starting document processing...")
    chunks, _ = process_documents(callback)
    
    if not chunks:
        print("No document chunks were generated.")
        return False
    
    # Create embeddings and build a vector store using Pinecone
    print(f"Creating embeddings for {len(chunks)} chunks and uploading to Pinecone...")
    if callback:
        callback(f"Creating embeddings for {len(chunks)} chunks and uploading to Pinecone...")
        
    _vectorstore = PineconeVectorStore.from_texts(
        chunks,
        embeddings,
        index_name=Config.PINECONE_INDEX_NAME,
        namespace=Config.PINECONE_NAMESPACE
    )
    
    print("Embeddings successfully created and stored in Pinecone!")
    if callback:
        callback("Embeddings successfully created and stored in Pinecone!")
    
    return True

def get_vectorstore():
    """Get the current vector store instance"""
    global _vectorstore
    return _vectorstore