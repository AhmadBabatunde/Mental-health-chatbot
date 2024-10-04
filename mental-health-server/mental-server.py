from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import re
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize these variables outside of the route function
huggingfacehub_api_token = os.getenv('huggingfacehub_api_token')
pinecone_api_key = os.getenv('pinecone_api_key')

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_mod = HuggingFaceEmbeddings(model_name=model_name)

vectorstore = PineconeVectorStore(
    index_name="mental-health-english",
    embedding=embedding_mod, 
    pinecone_api_key=pinecone_api_key
)

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", huggingfacehub_api_token=huggingfacehub_api_token)

prompt_template = """
You are a compassionate and empathetic counselor/therapist chatbot. 
Your primary goal is to promote self-discovery and support the client in exploring their 
thoughts and feelings without judgment. Always express genuine curiosity and foster 
a safe space for the client to gradually open up. Ask questions little by little, 
building rapport while gently encouraging the client to explore their emotions.
Guidelines:

Focus on empathy: Ask questions that show genuine curiosity, such as:
"I'm curious about how that made you feel."
"What thoughts went through your mind when that happened?"
Avoid leading questions: Instead of implying judgment (e.g., "Don't you think..."), ask neutrally:
"How do you feel about...?"
Encourage reflection: Frame questions to help the client explore their own solutions, such as:
"Can you describe a time when you handled a similar situation more successfully?"
"How do you see this issue affecting your life?"

Prompt: Use the following context to answer the client's question. Provide helpful information, and ask one or two reflective questions to guide them further in their journey.

Context: {context}

Chat History: {chat_history}

Client's Question: {question}

Response: Answer the client's question empathetically, based on the context provided and the chat history. Additionally, ask reflective questions to encourage deeper exploration.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

def generate_response(user_input, user_id):
    response = qa.invoke({"question": user_input})
    cleaned_response = re.sub(r"^\s*[-–—]+\s*", "", response['answer'])
    cleaned_response = cleaned_response.replace("\n", " ")
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    memory.chat_memory.add_message(AIMessage(content=cleaned_response.strip()))
    return cleaned_response.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    user_id = data.get('user_id', '')
    response = generate_response(question, user_id)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5001)