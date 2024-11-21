from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Cassandra
from watchdog.events import FileSystemEventHandler
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA
from watchdog.observers import Observer
from dotenv import load_dotenv
from astrapy import DataAPIClient
from docx2pdf import convert
import pdfplumber
import logging
import threading
import shutil
import time
import cassio
import os

app = Flask(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")            
FOLDER_TO_MONITOR = os.getenv("FOLDER_TO_MONITOR")
BACKUP_FOLDER = os.getenv("BACKUP_FOLDER")
ASTRA_DB_APPLICATION_TOKEN =  os.getenv("ASTRA_DEFAULT_DB_APPLICATION_TOKEN")
ASTRA_DB_ID =  os.getenv("ASTRA_DEFAULT_DB_ID")

logging.info("Initializing Astra DB client.")
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database_by_api_endpoint(f"https://{ASTRA_DB_ID}-us-east-2.apps.astra.datastax.com")
logging.info(f"Connected to Astra DB: {db.list_collection_names()}")
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

MODEL_EMBEDDING = "models/text-embedding-004"
UPLOAD_FOLDER = 'C:\OLD LAPY\AutoaReaume\Pdfs'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processed_files = set()
PROCESSED_FILES_PATH = os.getenv("DEFAULT_PROCESSED_FILES_PATH")

def reload_astra_db(selected_token, selected_db_id):
    global ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID
    ASTRA_DB_APPLICATION_TOKEN = selected_token
    ASTRA_DB_ID = selected_db_id
    try:
        logging.info("Re-initializing Astra DB client with selected database.")
        client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
        db = client.get_database_by_api_endpoint(f"https://{ASTRA_DB_ID}-us-east-2.apps.astra.datastax.com")
        logging.info(f"Connected to Astra DB: {db.list_collection_names()}")
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    except Exception as e:
        logging.error(f"Error reloading Astra DB: {e}")
        raise

def restart_observer(new_folder):
    global FOLDER_TO_MONITOR, observer_thread
    FOLDER_TO_MONITOR = new_folder

    if observer_thread and observer_thread.is_alive():
        observer.stop()
        observer.join()
        logging.info(f"Stopped the existing observer thread for folder: {FOLDER_TO_MONITOR}")

    start_observer_thread()  
    logging.info(f"Started observer for new folder: {FOLDER_TO_MONITOR}")


def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as file:
            return set(line.strip() for line in file)
    return set()

def save_processed_files():
    with open(PROCESSED_FILES_PATH, 'w') as file:
        for file_name in processed_files:
            file.write(file_name + '\n')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(file_path, resume_no):
    text = ""
    filename = os.path.basename(file_path)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING, google_api_key=GOOGLE_API_KEY)
        astra_vector_store = Cassandra(
            embedding=embeddings,
            table_name="pp_mini_demo",
            session=None,
            keyspace=None
        )

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                page_data = f"This is the content of Document or Resume {resume_no}: {page_text}"
                print("extraded")
                if page_text:
                    text += page_text
                    try:
                        astra_vector_store.add_texts([page_data])
                        logging.info(f"Inserted text of Document {resume_no}, page {page.page_number} into Astra Cassandra VectorStore.")
                    except Exception as e:
                        logging.error(f"Error inserting text of Document {resume_no}, page {page.page_number} into vectorstore: {e}")
                         
        print(f"extraction done {file_path}")
        if not text:
            logging.warning(f"No valid text extracted from {file_path}.")
        else:
            logging.info(f"Extracted text from {file_path}")
        
        processed_files.add(filename)
        save_processed_files()

        try:
            os.remove(file_path)
            logging.info(f"Deleted processed file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
    return text



def handle_file_conversion_and_processing(file_path, resume_no):
    filename = os.path.basename(file_path)

    if filename in processed_files:
        logging.info(f"File {filename} has already been processed. Skipping.")
        return  
    
    if file_path.endswith('.pdf'):
        process_file(file_path, resume_no)
    elif file_path.endswith('.docx'):
        pdf_path = convert_word_to_pdf(file_path)
        if pdf_path:
            process_file(pdf_path, resume_no)

def convert_word_to_pdf(word_path):
    pdf_path = word_path.replace('.docx', '.pdf')
    try:
        convert(word_path, pdf_path)
        logging.info(f"Converted {word_path} to {pdf_path}")
        os.remove(word_path) 
        return pdf_path
    except Exception as e:
        logging.error(f"Error converting {word_path} to PDF: {e}")
        return None

def process_existing_files_in_folder():
    for filename in os.listdir(FOLDER_TO_MONITOR):
        file_path = os.path.join(FOLDER_TO_MONITOR, filename)
        if filename in processed_files:
            logging.info(f"File {filename} has already been processed. Skipping.")
            continue 

        if allowed_file(filename):
            resume_no = len(processed_files) + 1
            handle_file_conversion_and_processing(file_path, resume_no)

def load_astra_vectorstore():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING, google_api_key=GOOGLE_API_KEY)
        astra_vector_store = Cassandra(
            embedding=embeddings,
            table_name="pp_mini_demo",  
            session=None,
            keyspace=None  
        )
        logging.info("Astra Cassandra VectorStore loaded successfully.")
        return astra_vector_store
    except Exception as e:
        logging.error(f"Error loading Astra Cassandra VectorStore: {e}")
        return None

def get_conversation_chain(astra_vector_store):
    try:
        retriever = astra_vector_store.as_retriever(search_kwargs={"k":20})
        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        logging.info("Conversation chain created successfully.")
        return conversation_chain
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        return None

PROMPT_TEMPLATE = PromptTemplate(
    template=(
        "Group pages by Document Number (found at the beginning of each page)."
        "If multiple pages have the same Document Number, combine them to form a complete document."
        "Search within the grouped pages for any missing information. If something is missing on one page, check other pages with the same Document Number."
        "After gathering all relevant information for each document, answer the {query} in bullet points."
    ),
    input_variables=["query"]
)

@app.route('/select_database', methods=['POST'])
def select_database():
    global selected_folder, processed_files, PROCESSED_FILES_PATH, PROMPT_TEMPLATE
    messages = []
    selected_db = request.form.get('database')
    folder_location = request.form.get('folderLocation')

    if selected_db == "resume":
        selected_token = os.getenv("ASTRA_RESUME_DB_APPLICATION_TOKEN")
        selected_db_id = os.getenv("ASTRA_RESUME_DB_ID")
        PROCESSED_FILES_PATH = os.getenv("RESUME_PROCESSED_FILES_PATH")
        prompt_template_string = os.getenv("RESUME_PROMPT_TEMPLATE")
    elif selected_db == "document":
        selected_token = os.getenv("ASTRA_DOCUMENT_DB_APPLICATION_TOKEN")
        selected_db_id = os.getenv("ASTRA_DOCUMENT_DB_ID")
        PROCESSED_FILES_PATH = os.getenv("DOCUMENT_PROCESSED_FILES_PATH")
        prompt_template_string = os.getenv("DOCUMENT_PROMPT_TEMPLATE")
    elif selected_db == "other":
        selected_token = os.getenv("ASTRA_OTHER_DB_APPLICATION_TOKEN")
        selected_db_id = os.getenv("ASTRA_OTHER_DB_ID")
        PROCESSED_FILES_PATH = os.getenv("OTHER_PROCESSED_FILES_PATH")
        prompt_template_string = os.getenv("OTHER_PROMPT_TEMPLATE")
    elif selected_db == "default":
        selected_token = os.getenv("ASTRA_DEFAULT_DB_APPLICATION_TOKEN")
        selected_db_id = os.getenv("ASTRA_DEFAULT_DB_ID")
        PROCESSED_FILES_PATH = os.getenv("DEFAULT_PROCESSED_FILES_PATH")
        prompt_template_string = os.getenv("DEFAULT_PROMPT_TEMPLATE")
    else:
        messages.append('Invalid database option selected. Loading default template.')
        selected_token = os.getenv("ASTRA_DEFAULT_DB_APPLICATION_TOKEN")  
        selected_db_id = os.getenv("ASTRA_DEFAULT_DB_ID")  
        PROCESSED_FILES_PATH = os.getenv("DEFAULT_PROCESSED_FILES_PATH")  
        prompt_template_string = os.getenv("PROMPT_TEMPLATE_DEFAULT")  

    try:
        logging.info("Disabling the previous database and folder...")

        if observer.is_alive():
            observer.stop() 
            observer.join() 
            logging.info(f"Stopped monitoring the previous folder: {selected_folder}")

        if selected_db and selected_token and selected_db_id:
            reload_astra_db(selected_token, selected_db_id)
            logging.info(f"Reloaded Astra DB with new database: {selected_db}")

        if folder_location:
            restart_observer(folder_location)
            selected_folder = folder_location
            processed_files = load_processed_files()  
            logging.info(f"Started monitoring the new folder: {folder_location}")
                      
            process_existing_files_in_folder()

        if prompt_template_string:
            PROMPT_TEMPLATE = PromptTemplate(
                template=prompt_template_string,
                input_variables=["query"]
            )
        else:
            PROMPT_TEMPLATE = PromptTemplate(
                template=(
                    "Group pages by Document Number(you will found at the beginning of each page)."
                    "If multiple pages have the same Document Number, combine them to form a complete document."
                    "Search within the grouped pages for any missing information. If something is missing on one page, check other pages with the same Document Number."
                    "After gathering all relevant information for each document, answer the {query} in bullet points."
                ),
                input_variables=["query"]
            )

        messages.append(f"Database, folder location, and prompt template updated to {selected_db if selected_db else 'default'} and {folder_location}.")
        return jsonify({'messages': messages})

    except Exception as e:
        logging.error(f"Error updating database and folder: {e}")
        messages.append(f"Error occurred: {e}")
        return jsonify({'messages': messages}), 500



@app.route('/query', methods=['POST'])
def query():
    user_query = request.data.decode('utf-8').strip()
    if not user_query:
        return 'No query provided', 400
    
    try:
        astra_vector_store = load_astra_vectorstore()
        if astra_vector_store is None:
            return 'Failed to load vector store.', 500

        conversation_chain = get_conversation_chain(astra_vector_store)
        if conversation_chain is None:
            return 'Failed to initialize conversation chain.', 500
        
        formatted_query = PROMPT_TEMPLATE.format(query=user_query)
        response = conversation_chain.invoke({"query": formatted_query})
        
        logging.info(f"Using the following prompt template: {PROMPT_TEMPLATE.template}")

        unique_answer = response["result"].strip() if response and "result" in response else "No valid answer found."
        return unique_answer
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f'Error occurred: {e}', 500
selected_folder = FOLDER_TO_MONITOR

@app.route('/upload', methods=['POST'])
def upload_file():
    global selected_folder 

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        if filename in processed_files:
            return jsonify({'message': f'File {filename} has already been processed. Skipping.'}), 400
        
       
        file_path = os.path.join(selected_folder, filename)
        file.save(file_path)

      
        return jsonify({'message': 'File uploaded successfully'})

    return jsonify({'message': 'Invalid file type'}), 400


class PDFFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf") or event.src_path.endswith(".docx"):
            logging.info(f"New file detected: {event.src_path}")
            
            try:
                shutil.copy(event.src_path, BACKUP_FOLDER)
                logging.info(f"File copied to backup folder: {BACKUP_FOLDER}")
            except Exception as e:
                logging.error(f"Failed to copy file to backup folder: {e}")
            
            time.sleep(5)  
            process_existing_files_in_folder()

def start_observer_thread():
    global observer_thread, observer
    observer = Observer()
    event_handler = PDFFileHandler()
    observer.schedule(event_handler, FOLDER_TO_MONITOR, recursive=True)
    observer.start()

    observer_thread = threading.Thread(target=observer.join)
    observer_thread.daemon = True
    observer_thread.start()
    logging.info(f"Started observer thread to monitor folder: {FOLDER_TO_MONITOR}")
    
@app.route('/')
def index():
    return render_template('index3.html')

if __name__ == "__main__":
    processed_files = load_processed_files()  
    process_existing_files_in_folder()
    start_observer_thread()  
    app.run(debug=True, port=5001)
