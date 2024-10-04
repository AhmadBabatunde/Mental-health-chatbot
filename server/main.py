import eventlet
eventlet.monkey_patch()


import os
from flask import Flask, request, jsonify, session
from pinecone import Pinecone, ServerlessSpec
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex,Settings
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import shutil
from llama_index.readers.file import PDFReader



# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
CORS(app)

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# pinecone_client=Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
pinecone_client=Pinecone(api_key=pinecone_api_key)
print("Pinecone initialized successfully.")

def list_index():
    index_data = pinecone_client.list_indexes()
    # Extract the names of the indexes
    index_names = [index['name'] for index in index_data]
    return index_names
        

# Initialize LLM and embedding model
llm = Gemini()
embed_model = GeminiEmbedding(model_name="models/embedding-001")


def create_pinecone_index(index_name):
    try:
        if index_name in list_index():
            pinecone_client.delete_index(index_name)
        pinecone_client.create_index(index_name, dimension=768,spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) )  

        index = pinecone_client.Index(index_name)
        return index
    except Exception as e:
        print(f"Failed to create or load Pinecone index '{index_name}': {str(e)}")
        return None

UPLOAD_FOLDER = 'uploads/'
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
print(f"Directory '{UPLOAD_FOLDER}' has been deleted.")
os.makedirs(UPLOAD_FOLDER)
print(f"Directory '{UPLOAD_FOLDER}' has been created.")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


@app.route('/upload', methods=['POST'])
@cross_origin(origin='*')

def upload_pdf():
    # Retrieve file from the request
    pdf_file = request.files['pdf']
    user_id = request.form.get('user_id')

    if not user_id:
        return jsonify({"error": "No user_id provided"}), 400
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400

    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(pdf_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(filepath)

    # Load the PDF document using PDFReader
    loader = PDFReader()
    documents  = loader.load_data(file=filepath)



    index_name = f"rag-experiment-{user_id}"

    pinecone_index = create_pinecone_index(index_name)

    # Create a PineconeVectorStore
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Create a StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index the documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    shutil.rmtree(UPLOAD_FOLDER)


   
    return jsonify({"message": "PDF uploaded and indexed successfully"}), 200

@app.route('/ask', methods=['POST'])
@cross_origin(origin='*')

def ask_question():
    # Retrieve question from the request
    question = request.json['question']
    user_id = request.json.get('user_id', '')
    index_name = f"rag-experiment-{user_id}"
    
    
    # Reinitialize the Pinecone index and vector store using the index name
    pinecone_index = pinecone_client.Index(index_name)
    if not pinecone_index:
        return jsonify({"error": "Index not found. Please upload a PDF first."}), 400

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Reinitialize the storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Reinitialize the index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Create a query engine
    query_engine = index.as_query_engine()

    # Query the index using the provided question
    gemini_response = query_engine.query(question)

    return jsonify({"answer": str(gemini_response)}), 200


@app.route('/end_session', methods=['POST'])
@cross_origin(origin='*')
def end_session():
    user_id = request.json.get('user_id', '')
    index_name = f"rag-experiment-{user_id}"
    if index_name:
        if index_name in list_index():
            try:
                pinecone_client.delete_index(index_name)
                return jsonify({"status": "Session ended"})

            except Exception as e:
                print(f"An error occurred while deleting the index: {str(e)}")
                return jsonify({"error": "Failed to delete the index"}), 500
            
        else:
                return jsonify({"status": "Session ended no currrnt index"})



        # session.pop('index_name', None)



if __name__ == '__main__':
    app.run(debug=True)
