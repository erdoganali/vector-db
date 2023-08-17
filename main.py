from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pdf2image import convert_from_bytes
from io import BytesIO
import os
import psycopg2
from pgvector import VectorClient
from transformers import pipeline
import torch
import numpy as np
from PIL import Image

app = FastAPI()

# Uploads folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Basic HTML form for uploading PDFs
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Document QA</title>
</head>
<body>
    <h1>Upload PDF</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="pdf_file">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
"""

# Database connection and vector client initialization
conn = psycopg2.connect(
    host="localhost",
    database="document_qa",
    user="postgres",
    password="mysecretpassword"
)
vector_client = VectorClient(
    psycopg2.connect(
        host="localhost",
        database="document_qa",
        user="postgres",
        password="mysecretpassword"
    ),
    "vectors",
    "document_id"
)

# Hugging Face pipeline initialization
nlp = pipeline("question-answering")

# Load the model and set it to evaluation mode
model = torch.load("model.pth")
model.eval()


@app.get("/", response_class=HTMLResponse)
async def get_upload_form():
    return html_form


@app.post("/upload/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.file.read())

    # Process PDF and save it to the database
    process_pdf(pdf_path)

    return {"status": "success", "filename": pdf_file.filename}


def process_pdf(pdf_path):
    pdf_images = convert_from_bytes(open(pdf_path, "rb").read())
    for page_num, image in enumerate(pdf_images):
        # Extract text from the image
        text = extract_text(image)

        # Add the text to the database and get the document ID
        document_id = add_to_database(text)

        # Get the vectors for the text and save them to the database
        vectors = get_vectors(text)
        save_vectors(vectors, document_id)

        # Answer some example questions
        answer_questions(document_id)


def extract_text(image):
    # Replace this with your actual OCR logic
    return "This is some example text."


def add_to_database(text):
    # Replace this with your actual database logic
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (text) VALUES (%s) RETURNING id", (text,))
    document_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    return document_id


def get_vectors(text):
    # Use the pgvector library to get the vectors for the text
    return vector_client.select("to_tsvector('english', text)", "text_vector")


def save_vectors(vectors, document_id):
    # Replace this with your actual database logic
    cursor = conn.cursor()
    for page_num, vector in enumerate(vectors):
        cursor.execute(
            "INSERT INTO vectors (document_id, page_num, vector) VALUES (%s, %s, %s)",
            (document_id, page_num, vector)
        )
    conn.commit()
    cursor.close()


def answer_questions(document_id):
    # Replace this with your actual question-answering logic
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM documents WHERE id = %s", (document_id,))
    text = cursor.fetchone()[0]
    cursor.execute("SELECT vector, page_num FROM vectors WHERE document_id = %s", (document_id,))
    rows = cursor.fetchall()
    vectors = [row[0] for row in rows]
    page_nums = [row[1] for row in rows]
    query_vector = vector_client.search("to_tsvector('english', 'What is the answer to this question?')")[0]
    similarities = np.array(vector_client.cosine_similarity(query_vector, vectors))
    top_page_num = page_nums[np.argmax(similarities)]
    top_page_index = page_nums.index(top_page_num)
    top_vector = vectors[top_page_index]
    top_text = extract_text_from_vector(top_vector, text)
    answer = nlp(question="What is the answer to this question?", context=top_text)["answer"]
    cursor.execute(
        "INSERT INTO answer_log (document_id, question, answer, page_num) VALUES (%s, %s, %s, %s)",
        (document_id, "What is the answer to this question?", answer, top_page_num)
    )
    conn.commit()
    cursor.close()


def extract_text_from_vector(vector, text):
    # Replace this with your actual vector-to-text logic
    return text


@app.get("/answer/")
async def answer_question(question: str):
    # Replace this with your actual question-answering logic
    query_vector = vector_client.search(f"to_tsvector('english', '{question}')")[0]
    cursor = conn.cursor()
    cursor.execute("SELECT document_id, page_num, vector FROM vectors")
    rows = cursor.fetchall()
    vectors = [row[2] for row in rows]
    document_ids = [row[0] for row in rows]
    page_nums = [row[1] for row in rows]
    similarities = np.array(vector_client.cosine_similarity(query_vector, vectors))
    top_document_id = document_ids[np.argmax(similarities)]
    top_page_num = page_nums[np.argmax(similarities)]
    top_page_index = page_nums.index(top_page_num)
    top_vector = vectors[top_page_index]
    cursor.execute("SELECT text FROM documents WHERE id = %s", (top_document_id,))
    text = cursor.fetchone()[0]
    top_text = extract_text_from_vector(top_vector, text)
    answer = nlp(question=question, context=top_text)["answer"]
    cursor.execute(
        "INSERT INTO answer_log (document_id, question, answer, page_num) VALUES (%s, %s, %s, %s)",
        (top_document_id, question, answer, top_page_num)
    )
    conn.commit()
    cursor.close()
    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
