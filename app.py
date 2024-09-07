# streamlit run app.py --server.enableXsrfProtection false
import streamlit as st
from utils import logger, handle_question, save_and_process_pdf, display_pdf_page, results_into_table
from model_handling import get_model_and_tokenizer

from pathlib import Path
from datetime import datetime




st.set_page_config(layout="wide")

# Define directories
dir_path = Path('')
MODEL_DIR = dir_path / 'model'
DATA_DIR = dir_path / 'pdfs'
MODEL_ID = 'meta-llama/Llama-2-13b-chat-hf'


# Load model and tokenizer only if not already loaded
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    logger.info(f'Model and tokenizer were initialized')
    st.session_state['model'], st.session_state['tokenizer'] = get_model_and_tokenizer(MODEL_ID, MODEL_DIR)
    logger.info(f'Model and tokenizer initialization succesfull')
else:
    logger.info(f'Model and tokenizer are already loaded')


# Retrieve model and tokenizer from session state
model = st.session_state['model']
tokenizer = st.session_state['tokenizer']



'''

Initialize Streamlit Application

'''

st.title("PDF Document Question Answering")
st.write("Upload a PDF file and ask questions about its content.")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key='pdf_uploader')


col1, col2 = st.columns(2)
logger.info(f'New Run starts at {datetime.now().strftime("%d/%m/%Y -- %H:%M")}')

# Column 1: PDF upload and question handling
with col1:
       
    # Clear everything on new PDF upload
    if uploaded_file is not None:
        if 'pdf_name' not in st.session_state or st.session_state['pdf_name'] != uploaded_file.name:
            st.session_state['pdf_name'] = uploaded_file.name
            st.session_state['question'] = ""  
            st.session_state['answer'] = None  
            st.session_state['page_number'] = 1  
            st.session_state['page_image'] = None  
            st.session_state['documents'] = None  
            save_and_process_pdf(uploaded_file, DATA_DIR)

    # If a PDF is already uploaded and stored in session state
    if 'pdf_name' in st.session_state:
        question = st.text_input(f"Ask a question based on {st.session_state['pdf_name']}:", value=st.session_state.get('question', ""))
        
        # Handle question submission
        if st.button("Submit Question", key='submit_question'):
            st.session_state['question'] = question 
            st.session_state['documents'], st.session_state['answer'] = handle_question(question, st.session_state['collection'], model, tokenizer)
            print(st.session_state['documents'])

        # Always display the stored answer if it exists
        if 'answer' in st.session_state and st.session_state['answer']:
            st.markdown(f"<div style='word-wrap: break-word;'>{st.session_state['answer']}</div>", unsafe_allow_html=True)


        sim_df = results_into_table(st.session_state['documents'])

        # Check if similarity results exist
        if not sim_df.empty:
            # Section Title
            st.subheader("Documents used for answer")
            
            # Display a list of the most similar pages
            st.write(f"Document pages ranked by similarity: {(', ').join([str(s) for s in list(sim_df['page number'])])}", bool=True)
    
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Now display the detailed information for each document
            for index, row in sim_df.iterrows():
                document_subset = row['document subset'].replace('\n', ' ')
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                        <b>Document name:</b> {row['Document name']}<br>
                        <b>Page number:</b> {row['page number']}<br>
                        <b>Subset similarity:</b> {row['subset similarity']}<br>
                        <b>Document subset:</b><br>
                        <div style="white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow: auto;">
                            {document_subset}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please upload a PDF file to begin.")

# Column 2: PDF page display
with col2:
    if 'pdf_name' in st.session_state:


        page_number = st.number_input(
            "Enter the page number to view", 
            min_value=1, 
            max_value=st.session_state['total_pages'], 
            value=st.session_state.get('page_number', 1), step=1, key='page_number_input')
        
        # Handle page display
        if st.button("Display Page", key='display_page'):
            st.session_state['page_number'] = page_number  # Store the current page number in session state
            st.session_state['page_image'] = display_pdf_page(page_number)  # Store the image in session state
        
        # Display the page if it has already been rendered
        if st.session_state.get('page_image'):
            st.image(st.session_state['page_image'], caption=f"Page {st.session_state['page_number']} of {st.session_state['pdf_name']}")

    else:
        st.info("Please upload a PDF file to view its pages.")

# streamlit run app.py --server.enableXsrfProtection false


