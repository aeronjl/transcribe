"""This module stores the OpenAI client instance. It is a singleton class that ensures only one instance of the OpenAI client is created."""
import openai
import os
from dotenv import load_dotenv

class OpenAIClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClient, cls).__new__(cls)
            cls._instance._initialize_openai()
        return cls._instance
    
    def _initialize_openai(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        
        openai.api_key = api_key
        
    def get_openai(self):
        return openai