import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(
    model_name=os.getenv("OPEN_AI_CHAT_MODEL"),
    openai_api_key=os.getenv("OPEN_AI_API_KEY"),
    openai_api_base=os.getenv("OPEN_AI_BASE_URL"),
    verbose=True
)

print(chat.invoke("你是谁?"))
