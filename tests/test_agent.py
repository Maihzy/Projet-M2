from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
agent_id = "ag:f81b839b:20250724:untitled-agent:1747b6e9"

client = Mistral(api_key=api_key)
response = client.beta.conversations.start(
    agent_id=agent_id,
    inputs="Bonjour, quel est le meilleur fromage français ?"
)
print(response.outputs)


