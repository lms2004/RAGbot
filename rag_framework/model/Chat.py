import os
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEndpoint
import os


def createHuggingFaceChat(repo_id="google/flan-t5-large"):
    # Ensure max_new_tokens is set to a value within the limit
    return HuggingFaceEndpoint(repo_id=repo_id)



def createOpenAIChat():
    return ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0.8, max_tokens=100)
