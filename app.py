import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def handle_file_upload(file, upload_path="uploads"):
    
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)

    file_path = os.path.join(upload_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.read())

    if file.filename.endswith(('.pdf', '.docx', '.txt', '.md')):
        return file_path
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, TXT, and Markdown are allowed.")


# 2. File Parsing and Content Extraction
import fitz  
from docx import Document 

def extract_content(file_path):
    content = ""
    images = []

    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                content += page.get_text()
                # Extract images
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    images.append(base_image["image"])
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            content += para.text + "\n"
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    return content, images



# 3. Preprocessing

import re

def preprocess_text(text):
    # Remove unwanted characters
    text = re.sub(r"\s+", " ", text).strip()
    # Split into chunks 
    chunks = text.split("\n\n")
    return chunks



# 4. Image Processing
import cv2
import numpy as np

def process_images(images):
    processed_images = []
    for img_data in images:
        # Decode image data
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        # Resize or preprocess (e.g., grayscale conversion)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray_image)
    return processed_images

# 5. Chatbot Integration
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline

def integrate_chatbot(chunks):
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]

    # Store in FAISS vector database
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Initialize retrieval-based QA
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_pipeline = RetrievalQA.from_chain_type(llm=pipeline("text2text-generation", model=os.getenv('L_MODEL')),
                                              retriever=retriever)
    return qa_pipeline


def main(file):
    # Step 1: Handle file upload
    file_path = handle_file_upload(file)
    
    # Step 2: Extract content and images
    content, images = extract_content(file_path)
    
    # Step 3: Preprocess text
    chunks = preprocess_text(content)
    
    # Step 4: Process images
    processed_images = process_images(images)
    
    # Step 5: Integrate chatbot
    chatbot = integrate_chatbot(chunks)
    
    return chatbot, processed_images


