markdown
# PDF Retrieval-Augmented Generation (RAG) System

This repository provides a system for uploading PDFs and utilizing a Retrieval-Augmented Generation (RAG) approach to extract and retrieve information from them. 
The system is powered by state-of-the-art models and vectorization techniques to ensure accurate and efficient information retrieval.

## Features
- **PDF Upload**: Upload your PDF files to the system for processing.
- **RAG-based Information Retrieval**: Employ a Retrieval-Augmented Generation system to query and find relevant information within the uploaded PDFs.
- **Pre-trained LLaMA Model**: Uses the `meta-llama/Llama-2-13b-chat-hf` model for natural language processing tasks, including both the model and tokenizer.
- **ChromaDB Vectorization**: Utilizes ChromaDB for embedding and vectorizing text from PDFs to facilitate efficient search and retrieval.
- **Streamlit UI**: Provides a simple and intuitive user interface using Streamlit, served on `localhost` by default.

## Installation

1. **Clone the Repository**
2. **CD repository location**
3. **Instal requirements.txt**
4. **Run with command streamlit run app.py**
5. **In case of GPU configuration error run application with streamlit run app.py --server.enableXsrfProtection false**


**Please ensure you have installed poppler, otherwise pdf reader wont run**
