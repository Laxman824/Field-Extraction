import streamlit as st
from PIL import Image
import json
import pandas as pd
from document_handler import DocumentHandler
from models.model_loader import load_model

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = {}
    if 'current_doc_index' not in st.session_state:
        st.session_state.current_doc_index = 0

def display_batch_results(results: Dict):
    """Display results for multiple documents"""
    st.header("Processing Results")
    
    # Create tabs for different views
    summary_tab, details_tab, download_tab = st.tabs([
        "Summary", "Detailed View", "Download Options"
    ])
    
    with summary_tab:
        # Display summary statistics
        total_docs = len(results)
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = total_docs - successful
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", total_docs)
        col2.metric("Successfully Processed", successful)
        col3.metric("Failed", failed)
        
        # Display summary table
        summary_data = []
        for file_name, result in results.items():
            summary_data.append({
                "File Name": file_name,
                "Status": result.get('status', 'unknown'),
                "Pages": result.get('total_pages', 0) if result.get('status') == 'success' else 0,
                "Error": result.get('error', '') if result.get('status') == 'error' else ''
            })
        
        st.dataframe(pd.DataFrame(summary_data))
    
    with details_tab:
        # Detailed view for each document
        for file_name, result in results.items():
            with st.expander(f"Details for {file_name}"):
                if result.get('status') == 'success':
                    for page in result['pages']:
                        st.subheader(f"Page {page['page_number']}")
                        display_page_results(page['results'])
                else:
                    st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
                    if 'traceback' in result:
                        with st.expander("Show Error Details"):
                            st.code(result['traceback'])
    
    with download_tab:
        # Prepare download options
        st.subheader("Download Results")
        
        # JSON download
        json_str = json.dumps(results, indent=2)
        st.download_button(
            "Download Full Results (JSON)",
            json_str,
            file_name="batch_results.json",
            mime="application/json"
        )
        
        # CSV download
        csv_data = prepare_csv_export(results)
        st.download_button(
            "Download Summary (CSV)",
            csv_data,
            file_name="batch_summary.csv",
            mime="text/csv"
        )

def prepare_csv_export(results: Dict) -> str:
    """Prepare results for CSV export"""
    csv_rows = []
    for file_name, result in results.items():
        if result.get('status') == 'success':
            for page in result['pages']:
                page_data = {
                    'File Name': file_name,
                    'Page Number': page['page_number']
                }
                # Add extracted fields
                if 'extracted_info' in page['results']:
                    for field, value in page['results']['extracted_info'].items():
                        page_data[field] = value
                csv_rows.append(page_data)
    
    return pd.DataFrame(csv_rows).to_csv(index=False)

def main():
    st.set_page_config(page_title="Multi-Document OCR", layout="wide")
    initialize_session_state()
    
    st.title("Multi-Document OCR Processing")
    st.write("Upload multiple documents for batch processing")
    
    # Initialize document handler
    doc_handler = DocumentHandler()
    
    # Create sidebar and get selected options
    selected_fields, confidence_threshold = create_sidebar()
    
    # File upload - multiple files
    uploaded_files = st.file_uploader(
        "Choose documents to process",
        type=doc_handler.SUPPORTED_FORMATS['images'] + 
             doc_handler.SUPPORTED_FORMATS['documents'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Load model
        model = load_model()
        
        # Process button
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Prepare files for processing
                files_to_process = [
                    (file.read(), file.name) for file in uploaded_files
                ]
                
                # Process batch
                results = doc_handler.process_batch(
                    files_to_process,
                    model,
                    selected_fields,
                    confidence_threshold
                )
                
                # Store results in session state
                st.session_state.processed_docs = results
                
                # Display results
                display_batch_results(results)
        
        # Cleanup
        doc_handler.cleanup()

if __name__ == "__main__":
    main()