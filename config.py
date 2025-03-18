# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings"""
    # API Keys
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    
    # Pinecone settings
    PINECONE_INDEX_NAME = 'uconn-course-catalog'
    PINECONE_NAMESPACE = 'course-catalog-data'
    EMBEDDING_DIMENSION = 768  # Dimension for Google's embedding model
    
    # API settings
    MAX_TOKENS = 2048
    TEMPERATURE = 0.3
    TOP_P = 0.9
    
    @staticmethod
    def check_environment_variables():
        """Check if required environment variables are set"""
        required_vars = ['GROQ_API_KEY', 'GOOGLE_API_KEY', 'PINECONE_API_KEY']
        return all(os.environ.get(var) for var in required_vars)

def setup_environment():
    """Set up environment variables for development"""
    load_dotenv()