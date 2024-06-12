"""This module stores the OpenAI client instance. It is a singleton class that ensures only one instance of the OpenAI client is created."""
import openai

class OpenAIClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_openai(self):
        return openai