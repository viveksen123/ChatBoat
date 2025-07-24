import os
from dotenv import load_dotenv
import gradio as gr
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load Hugging Face API key
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise ValueError("API Key not found! Add it to .env as HUGGINGFACEHUB_API_TOKEN")

# Load HuggingFace chat model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    api_key=api_key
)
model = ChatHuggingFace(llm=llm)

# Chat function
def chat_with_bot(message, history):
    response = model.invoke(message)
    return response.content

# Gradio interface
chat_interface = gr.ChatInterface(fn=chat_with_bot, title="🧠 LangChain ChatBot (Hugging Face)")

# Launch app
if __name__ == "__main__":
    chat_interface.launch(share=True)
