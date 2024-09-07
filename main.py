
from utils import logger, call_oracle
from model_handling import get_model_and_tokenizer
from vectordb_handling import prepare_PDF_document, prepare_collection_db, retreive_documents
import sys

from pathlib import Path
from glob import glob
from datetime import datetime

from langchain.document_loaders.pdf import PyPDFLoader

logger.info(f'New Run starts at {datetime.now().strftime("%d/%m/%Y -- %H:%M")}')


dir_path = Path('')
MODEL_DIR = dir_path / 'model'
DATA_DIR = dir_path / 'data'
MODEL_ID ='meta-llama/Llama-2-13b-chat-hf'

# Get model along with tokenizer
model, tokenizer = get_model_and_tokenizer(MODEL_ID, MODEL_DIR)
logger.info(f'Model and tokenizer set ready')

# Get PDF file along with its path
pdf_file  = glob(str(DATA_DIR.resolve())+'/*.pdf')[0]
if not pdf_file:
    logger.error('No PDF files found in the data directory.')
    sys.exit('Error: No PDF files found.')
pdf_text = PyPDFLoader(pdf_file).load()

# Prepare chroma DB database 
new_chunks = prepare_PDF_document(document = pdf_text)
chroma_client, collection = prepare_collection_db(new_chunks)

# Question answering logic
pdf_name = pdf_file.split("\\")[-1]
while True:
    question = input(f'Ask a question about {pdf_name} (or type "exit" to leave): ')

    if question.lower() == 'exit':
        logger.info('User chose to exit the application.')
        print('Exiting the application. Goodbye!')
        break

    similar_documents = retreive_documents(collection, question)

    if len(similar_documents['documents']) == 0:
        logger.warning(f'For question -------- {question} -------- no context was found.')
        print('Sorry, no relevant context was found in the current document for your question.')
    else:
        context = "\n\n".join([f"Document {i+1}:\n{string}" for i, string in enumerate(similar_documents['documents'])])
        answer = call_oracle(model, tokenizer, question_text=question, context=context)
        print(f"Answer: {answer}")
