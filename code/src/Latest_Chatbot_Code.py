import os
import pandas as pd
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from configparser import ConfigParser
import asyncio
import logging
from threading import Thread
import shutil
import subprocess

# Setup logging for debugging and error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__, template_folder=os.path.join(r'C:/Users/snrah/PycharmProjects/gaipl-fusion-force/code/src/Templates'))

# Configure Gemini API (Google Generative AI)
genai.configure(api_key="ADD API KEY here")
model = genai.GenerativeModel("gemini-2.0-flash")  # Updated to use gemini-2.0-flash

# Initialize ChromaDB and Sentence Transformer for embeddings
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
    {"query": "Is there telemetry details related to Server crash due to high CPU usage in Server3 on 25-03-2025",
     "answer": "On 25-03-2025, the CPU usage was 95.4%, Memory usage was 80.2% and Disk Usage was 60.5%"},
    {"query": "What is the CPU usage percentage when Server crash due to high CPU usage in Server3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, the CPU usage was 95.4%"},
    {"query": "What is the Memory usage percentage when Server crash due to high CPU usage in Server3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, Memory usage was 80.2%"},
    {"query": "What is the Disk usage percentage when Server crash due to high CPU usage in Server3 occurred on 25-03-2025",
     "answer": "On 25-03-2025, Disk Usage was 60.5%"},
    {"query": "What could be the action or recommendation for Server crash due to high CPU usage",
     "answer": "Action - Investigate running processes consuming excessive CPU. Optimize processes and monitor the system regularly. Recommendation- Identify resource-hogging processes and either optimize them or distribute the load across additional servers. Implement alerting thresholds for CPU usage to catch high usage early."}
]

# Read directory paths from config file
def get_config_path():
    """Read the configuration file and return the directory path for Excel files."""
    config = ConfigParser()
    config.read('config.ini')  # Ensure the config.ini file exists in the same folder
    excel_directory = config.get('FilePaths', 'EXCEL_DIRECTORY')
    return excel_directory

# Extracts content from all Excel files in a given directory
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
    return documents

# Index content into ChromaDB
def index_documents(directory):
    """Indexes general document content into ChromaDB."""
    documents = extract_excel_content(directory)

    for i, (filename, sheet, content) in enumerate(documents):
        embedding = embed_model.encode(content).tolist()
        collection.add(documents=[content], embeddings=[embedding], ids=[f"doc-{i}"])
        logger.info(f"Indexed document: {filename} - {sheet}")

# Retrieve the closest matching document content based on the query
def retrieve_document(query):
    """Finds the closest matching document content."""
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results and results.get("documents") and results["documents"][0]:
        return results["documents"][0][0]

    return None  # Return None if no relevant information is found

# Generate a concise, actionable response by searching local documents and using few-shot learning if necessary
def generate_response(query):
    """Generate a concise, actionable response by searching local documents and using few-shot learning if necessary."""
    retrieved_content = retrieve_document(query)

    # Create a prompt focused on getting a short, direct answer
    few_shot_prompt = "".join([f"Q: {ex['query']}\nA: {ex['answer']}\n" for ex in FEW_SHOT_EXAMPLES])

    # If relevant content is retrieved from the document, use it in the prompt
    if retrieved_content:
        prompt = f"{few_shot_prompt}Q: {query}\nA: {retrieved_content.strip()}"
    else:
        # If no document found, instruct LLM to generate a concise response
        prompt = f"{few_shot_prompt}Q: {query}\nA: Provide a brief, direct response."

    # Generate the response asynchronously to improve performance
    response = asyncio.run(generate_async_response(prompt))

    # Post-process the response to ensure it's concise and actionable
    action_buttons = False
    if "copy file" in response.lower() or "restart service" in response.lower():
        action_buttons = True

    return {"response": format_crisp_response(response), "enable_buttons": action_buttons}

# Asynchronously generate the response from the model
async def generate_async_response(prompt):
    """Generate response asynchronously."""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
    return response.text

# Trims and simplifies the response to ensure it's concise and actionable
def format_crisp_response(response_text):
    """Trims and simplifies the response to ensure it's concise and actionable."""
    lines = response_text.split("\n")
    crisp_response = " ".join(line.strip() for line in lines if line.strip())  # Clean up extra spaces

    # Further trimming for length or excess info, e.g., only return the first few sentences
    crisp_response = " ".join(crisp_response.split(".")[:2]) + "."  # Take the first 2 sentences

    return crisp_response

# Action handler for file operations
def copy_file(source_path, destination_path):
    """Function to copy a file from source to destination."""
    try:
        shutil.copy(source_path, destination_path)
        logger.info(f"File copied from {source_path} to {destination_path}")
        return f"File copied from {source_path} to {destination_path}"
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return f"Error copying file: {e}"

def restart_service(service_name):
    """Function to restart a service using a batch file or system command."""
    try:
        # For Windows: Use a batch file or service management commands
        command = f"net stop {service_name} && net start {service_name}"
        subprocess.run(command, shell=True, check=True)
        logger.info(f"Service {service_name} restarted successfully.")
        return f"Service {service_name} restarted successfully."
    except subprocess.CalledProcessError as e:
        logger.error(f"Error restarting service: {e}")
        return f"Error restarting service: {e}"

# Route to handle user queries
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('user_input')  # Use .get() to avoid KeyError
    if not user_query:
        return jsonify({"error": "Missing user input"}), 400  # Return 400 Bad Request

    response_data = generate_response(user_query)
    return jsonify({"bot_response": response_data["response"], "enable_buttons": response_data["enable_buttons"]})

# Route to perform action like copy file or restart service
@app.route('/perform_action', methods=['POST'])
def perform_action():
    action_type = request.form.get('action_type')  # e.g., 'copy_file' or 'restart_service'
    if action_type == 'copy_file':
        source = request.form.get('source')
        destination = request.form.get('destination')
        response = copy_file(source, destination)
    elif action_type == 'restart_service':
        service_name = request.form.get('service_name')
        response = restart_service(service_name)
    else:
        return jsonify({"error": "Invalid action type"}), 400  # Return 400 Bad Request if action is not valid

    return jsonify({"bot_response": response})

if __name__ == "__main__":
    try:
        excel_directory = get_config_path()  # Get the directory from the config file
        if not os.path.exists(excel_directory):
            raise FileNotFoundError(f"Directory {excel_directory} not found!")
        logger.info(f"Using dataset directory: {excel_directory}")
        index_documents(excel_directory)
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
    app.run(debug=True)
