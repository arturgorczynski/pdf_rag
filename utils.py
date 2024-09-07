import logging
import streamlit as st
import os
from vectordb_handling import prepare_PDF_document, prepare_collection_db, retreive_documents
from glob import glob
from langchain.document_loaders.pdf import PyPDFLoader
from PyPDF2 import PdfReader
import pdf2image
from io import BytesIO
from PIL import Image
import pandas as pd
import re


log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.handlers.RotatingFileHandler('logs/app.log', maxBytes=5 * 1024 * 1024, backupCount=5)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def call_oracle( model, tokenizer, context:str, question_text:str):
    system_text = """You will be provided with question and context.
    Use context information to answer question as accurately as you can.
    Keep your answer short and cosise. You must not exceed 1000 characters..
    Give yourself time to read question and context carefully."""
    context = {context}
    question_text = {question_text}


    final_question = f'''
    {system_text}

    Context: {context}

    [INST]Question: {question_text}[/INST]
    '''

    prompt_tokens  = tokenizer.encode(final_question, return_tensors="pt").to(model.device)
    start_index = prompt_tokens.shape[-1]

    output_ids = model.generate(prompt_tokens, max_length=5000)
    generation_output = output_ids[0][start_index:]
    generated_text = tokenizer.decode(generation_output, skip_special_tokens=True)

    return generated_text


def handle_question(question, collection, model, tokenizer):
    if question:
        similar_documents = retreive_documents(collection, question)
        if len(similar_documents['documents']) == 0:
            logger.warning(f'For question -------- {question} -------- no context was found.')
            return {}, 'Sorry, no relevant context was found in the current document for your question.'
        else:
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(similar_documents['documents'])])
            answer = call_oracle(model, tokenizer, question_text=question, context=context)
            answer = f"<div style='word-wrap: break-word;'>{answer}</div>"
            return similar_documents, answer
    else:
        return "Please enter a question."

def save_and_process_pdf(uploaded_file, DATA_DIR):
    pdf_path = DATA_DIR / uploaded_file.name
    pdf_name = pdf_path.name
    # Ensure the directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Delete any existing PDF files in DATA_DIR
    existing_pdfs = glob(str(DATA_DIR.resolve()) + '/*')
    for existing_pdf in existing_pdfs:
        os.remove(existing_pdf)
        st.warning("PDF replaced with new one.")

    # Save new PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully.")

    # Process PDF
    pdf_text = PyPDFLoader(str(pdf_path)).load()
    new_chunks = prepare_PDF_document(document=pdf_text)
    chroma_client, collection = prepare_collection_db(new_chunks)


    st.session_state['collection'] = collection
    st.session_state['pdf_name'] = uploaded_file.name
    st.session_state['pdf_data'] = uploaded_file.getvalue()
    st.session_state['pdf_reader'] = PdfReader(BytesIO(st.session_state['pdf_data']))
    st.session_state['total_pages'] = len(st.session_state['pdf_reader'].pages)

    st.success(f"PDF document {pdf_name} has been processed and database prepared.")
    
def display_pdf_page(page_number):
    pdf_data = st.session_state['pdf_data']
    images = pdf2image.convert_from_bytes(pdf_data, first_page=page_number, last_page=page_number)
    return images[0]


def results_into_table(results):
    df = pd.DataFrame.from_dict(results)
    if df.shape[0] > 1:
        df['distances'] = df['distances'].round(2)
        df.sort_values(by='distances', ascending=True)
        df[['Document name', 'page number']] = df['ids'].str.split(':', expand=True)[[0, 1]]
        df = df[['Document name', 'page number', 'distances', 'documents']]
        df.rename(columns={'distances': 'subset similarity', 'documents': 'document subset'}, inplace=True)
        
    else:
        df = pd.DataFrame()

    return df