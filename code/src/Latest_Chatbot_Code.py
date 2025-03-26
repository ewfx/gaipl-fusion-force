import os
import shutil  # For copying files
import pandas as pd
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
import logging
import traceback

app = Flask(__name__,
            template_folder=os.path.join(r'C:/Users/snrah/PycharmProjects/gaipl-fusion-force/code/src/Templates'))

# Configure Logging for debugging and better error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBRVUM2GAIpbSjCnplnhhBiuynudl9XmJo")
model = genai.GenerativeModel("gemini-2.0-flash")  # Updated to use gemini-2.0-flash

# Initialize ChromaDB and Sentence Transformer
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="excel_docs")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Few-shot prompt examples for better accuracy
FEW_SHOT_EXAMPLES = [
    {"query": "For Job Name - Daily_Sales_Ingestion failed due missing input data from SalesDB, which is assigned to Data Integration Team. Provide resolution notes?",
     "answer": "Copy file to reload missing data in SalesDB and re-trigger the job."},
    {"query": "For Job Name - ReportGenerationJob failed due invalid file format in SalesReportOutput.xlsx Provide resolution notes?",
     "answer": "Copy file from source path and re-run the report generation job."},
    {"query": "For Job Name - Weekly_Inventory_Update failed due to incorrect file format. Provide resolution notes?",
     "answer": "Correct the file format and restart the job."},
    {"query": "What are the upstream and downstream dependencies for the job Daily_Sales_Ingestion?",
     "answer": "Upstream: POS Data Extraction; Downstream: BI Dashboards."},
    {"query": "What is the business impact for the job Operational_Efficiency_Metrics?",
     "answer": "Increased costs and reduced efficiency hurt profitability."},
    {"query": "Is there telemetry details related to Server crash due to high CPU usage in Server-3 on 25-03-2025",
     "answer": "On 25-03-2025, the CPU usage was 95.4%, Memory usage was 80.2% and Disk Usage was 60.5%"},
    {"query": "What is the CPU usage percentage when Server crash due to high CPU usage in Server-3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, the CPU usage was 95.4%"},
    {"query": "What is the Memory usage percentage when Server crash due to high CPU usage in Server-3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, Memory usage was 80.2%"},
    {"query": "What is the Disk usage percentage when Server crash due to high CPU usage in Server-3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, Disk Usage was 60.5%"},
    {"query": "What could be the action or recommendation for Server crash due to high CPU usage",
     "answer": "Action - Investigate running processes consuming excessive CPU. Optimize processes and monitor the system regularly. Recommendation- Identify resource-hogging processes and either optimize them or distribute the load across additional servers. Implement alerting thresholds for CPU usage to catch high usage early."}
]

# Function to copy a file
def copy_file(source_path, destination_path):
    try:
        if not os.path.exists(source_path):
            return f"Error: The source file at {source_path} does not exist."

        if os.path.exists(destination_path):
            return f"Error: A file already exists at the destination path: {destination_path}"

        shutil.copy2(source_path, destination_path)
        return "Successfully completed: File copied successfully."
    except Exception as e:
        logger.error(f"Error copying file from {source_path} to {destination_path}")
        logger.error(traceback.format_exc())
        return f"Error occurred while copying the file: {e}"

# Extracts text from all Excel files in a given directory
def extract_excel_content(directory):
    """Extracts text from all Excel files in a given directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                for sheet, data in df.items():
                    text = data.astype(str).values.flatten()  # Convert to string
                    content = " ".join(text)
                    documents.append((file, sheet, content))
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                logger.error(traceback.format_exc())  # Log full stack trace
    return documents

# Indexes general document content into ChromaDB
def index_documents(directory):
    """Indexes general document content into ChromaDB."""
    documents = extract_excel_content(directory)

    for i, (filename, sheet, content) in enumerate(documents):
        try:
            embedding = embed_model.encode(content).tolist()
            collection.add(documents=[content], embeddings=[embedding], ids=[f"doc-{i}"])
            logger.info(f"Indexed document: {filename} - {sheet}")
        except Exception as e:
            logger.error(f"Error indexing document: {filename} - {sheet}")
            logger.error(traceback.format_exc())

# Retrieves the closest matching document content based on the query
def retrieve_document(query):
    """Finds the closest matching document content."""
    try:
        query_embedding = embed_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=3)  # Fetch 3 documents for better relevance

        if results and results.get("documents"):
            return results["documents"]
        return None  # Return None if no relevant information is found
    except Exception as e:
        logger.error(f"Error retrieving document for query: {query}")
        logger.error(traceback.format_exc())
        return None

# Generate response: first searches local documents; if no match, queries Gemini API
def generate_response(query):
    """Generate a response by first searching local documents, and if no match, querying Gemini API."""
    retrieved_content = retrieve_document(query)

    few_shot_prompt = "".join(
        [f"Example Query: {ex['query']}\nExample Answer: {ex['answer']}\n\n" for ex in FEW_SHOT_EXAMPLES])

    if retrieved_content:
        prompt = f"{few_shot_prompt}User Query: {query}\n\nRelevant Information from Documents: {retrieved_content}"
    else:
        prompt = f"{few_shot_prompt}User Query: {query}\n\nNo relevant information found in documents. Please generate a comprehensive answer using external knowledge."

    try:
        response = model.generate_content(prompt)
        # Check if the response includes file copying instruction
        if "Copy file" in response.text:
            return f"{response.text} Please provide the source and destination file paths."

        return response.text if hasattr(response, 'text') else "Error generating response."
    except Exception as e:
        logger.error(f"Error generating response from Gemini: {e}")
        logger.error(traceback.format_exc())
        return "An error occurred while generating the response."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('user_input')
    if not user_query:
        return jsonify({"error": "Missing user input"}), 400  # Return 400 Bad Request if no input provided

    try:
        answer = generate_response(user_query)
        return jsonify({"bot_response": answer})
    except Exception as e:
        logger.error(f"Error processing query: {user_query}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An error occurred while processing the request"}), 500  # Internal Server Error

@app.route('/copy-file', methods=['POST'])
def copy_file_route():
    source_file = request.form.get('source_file')
    destination_file = request.form.get('destination_file')

    if not source_file or not destination_file:
        return jsonify({"error": "Both source and destination file paths are required."}), 400

    result = copy_file(source_file, destination_file)
    return jsonify({"message": result})

if __name__ == "__main__":
    excel_directory = r"C://Users//snrah//PycharmProjects//gaipl-fusion-force//code//src//Dataset"  # Change to your directory
    try:
        index_documents(excel_directory)
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error initializing the app: {e}")
        logger.error(traceback.format_exc())
