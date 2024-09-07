
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings



def prepare_PDF_document(document='', chunk_size=700, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(document)

    new_chunks = []
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source').split('\\')[-1]
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{int(page)+1}"

        if current_page_id == last_page_id:
            current_chunk_index += 1  
        else:
            current_chunk_index = 0  
        
        last_page_id = current_page_id

        full_page_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = full_page_id

        new_chunks.append([chunk.page_content, full_page_id])
    
    return new_chunks


def prepare_collection_db(chunks):
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.create_collection(name="current_pdf", metadata={"hnsw:space": "cosine"})
    except ValueError:
        chroma_client.delete_collection('current_pdf')
        collection = chroma_client.create_collection(name="current_pdf", metadata={"hnsw:space": "cosine"})
    collection.add(documents=[c[0] for c in chunks], ids= [c[1] for c in chunks])

    return chroma_client, collection

def retreive_documents(collection, question):
    results = collection.query(
        query_texts=[question], 
        n_results=8,
        include=["documents", 'distances']
    )
    del results['metadatas']
    del results['embeddings']

    index = next((i for i, v in enumerate(results['distances'][0]) if v > 0.7), 10)
    for k,v in results.items():
        results[k] = results[k][0][:index]
    
    return results