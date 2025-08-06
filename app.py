import os
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

from functools import wraps
from flask import request, Response

APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")

def check_auth(username, password):
    return username == APP_USERNAME and password == APP_PASSWORD

def authenticate():
    return Response(
        "Unauthorized. Please provide valid credentials.\n",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Load environment variables
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
AZURE_FORMRECOGNIZER_ENDPOINT = os.getenv("AZURE_FORMRECOGNIZER_ENDPOINT")
AZURE_FORMRECOGNIZER_KEY = os.getenv("AZURE_FORMRECOGNIZER_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Azure clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
form_recognizer_client = DocumentAnalysisClient(
    endpoint=AZURE_FORMRECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORMRECOGNIZER_KEY)
)
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

def convert_docx_to_pdf(docx_path, pdf_path):
    doc = Document(docx_path)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - inch  # Start 1 inch from the top

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines = split_text(text, 90)  # wrap at 90 chars
            for line in lines:
                c.drawString(inch, y, line)
                y -= 14  # move down line by line
                if y < inch:
                    c.showPage()
                    y = height - inch

    c.save()

def split_text(text, max_length):
    # Simple word wrapping
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= max_length:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def upload_to_blob(file_path, filename):
    blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=filename)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=AZURE_BLOB_CONTAINER,
        blob_name=filename,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(minutes=10)
    )
    return f"https://{blob_client.account_name}.blob.core.windows.net/{AZURE_BLOB_CONTAINER}/{filename}?{sas_token}"

@app.route("/")
@requires_auth
def index():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
@requires_auth
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No valid file uploaded"}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    # Enforce allowed extensions
    if file_ext not in [".pdf", ".doc", ".docx"]:
        return jsonify({"error": "Unsupported file type"}), 400

    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(temp_path)

    # Convert .docx to .pdf if needed
    if file_ext == ".docx":
        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
        convert_docx_to_pdf(temp_path, pdf_path)
        upload_name = pdf_filename
        file_path_to_upload = pdf_path
    else:
        upload_name = f"{uuid.uuid4()}{file_ext}"
        file_path_to_upload = temp_path

    try:
        # Upload to Azure Blob
        blob_url = upload_to_blob(file_path_to_upload, upload_name)

        # Analyze with Form Recognizer
        poller = form_recognizer_client.begin_analyze_document_from_url("prebuilt-document", blob_url)
        result = poller.result()

        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"

        # Summarize with Azure OpenAI
        prompt = (
            "You are a legal assistant. Summarize the following legal document and extract key clauses, "
            "dates, parties involved, and obligations:\n\n" + extracted_text
        )

        chat_response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        summary_text = chat_response.choices[0].message.content.strip()
        return jsonify({"summary": summary_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
