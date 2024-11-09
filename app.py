import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import re
from doctr.models import ocr_predictor
import io
import traceback
import pandas as pd
import sys
print(sys.path)  # This will print the list of paths Python is searching

from typing import Dict, List, Tuple, Optional
from utils.document_handler import DocumentHandler


# Define field categories and their definitions
FIELD_CATEGORIES = {
    "Personal Information": {
        "PAN": {
            "pattern": r"\b(PAN|Permanent\s+Account\s+Number)\b",
            "description": "Permanent Account Number for tax identification",
            "example": "ABCDE1234F",
            "validation": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
        },
        "Tax Status": {
            "pattern": r"\bTax\s+Status\b",
            "description": "Tax category of the investor",
            "example": "Individual, NRI, Company",
            "validation": None
        }
    },
    "Investment Details": {
        "Scheme Name": {
            "pattern": r"\b(Scheme|Plan)\s+Name\b",
            "description": "Name of the investment scheme or plan",
            "example": "Growth Fund, Equity Fund",
            "validation": None
        },
        "Folio Number": {
            "pattern": r"\b(Folio|Account)\s+(Number|No\.?)\b",
            "description": "Unique identifier for your investment account",
            "example": "1234567890",
            "validation": r"\b[A-Za-z0-9]+\b"
        },
        "Number of Units": {
            "pattern": r"\b(Number\s+of\s+Units|Units|Quantity)\b",
            "description": "Number of units held in the investment",
            "example": "100.50",
            "validation": r"\b\d+(\.\d+)?\b"
        }
    },
    "Contact Information": {
        "Mobile Number": {
            "pattern": r"\b(Mobile|Phone|Cell)\s+(Number|No\.?)\b",
            "description": "Contact phone number",
            "example": "+91 9876543210",
            "validation": r"\b(\+\d{1,3}[-.\s]?)?\d{10,14}\b"
        },
        "Email": {
            "pattern": r"\b(Email|E-mail)\b",
            "description": "Email address for correspondence",
            "example": "investor@example.com",
            "validation": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        },
        "Address": {
            "pattern": r"\bAddress\b",
            "description": "Physical address for correspondence",
            "example": "123 Main St, City",
            "validation": None
        }
    },
    "Banking Information": {
        "Bank Account Details": {
            "pattern": r"\b(Bank\s+Account|Account)\s+(Details|Information)\b",
            "description": "Bank account and IFSC information",
            "example": "A/C: 1234567890, IFSC: ABCD0123456",
            "validation": None
        }
    },
    "Date Information": {
        "Date": {
            "pattern": r"\b[Dd]ate\b",
            "description": "Document or transaction date",
            "example": "2024-01-01",
            "validation": None
        }
    }
}

@st.cache_resource
def load_model():
    """Load and cache the OCR model"""
    try:
        return ocr_predictor(pretrained=True)
    except Exception as e:
        st.error(f"Error loading OCR model: {str(e)}")
        st.stop()

def preprocess_image(image: Image.Image, max_size: int = 1000) -> Image.Image:
    """Preprocess the input image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image

def create_sidebar() -> Tuple[Dict[str, bool], float]:
    """Create sidebar with field selection and processing options"""
    st.sidebar.title("Processing Options")
    
    selected_fields = {}
    
    # Field selection by category
    st.sidebar.header("Select Fields to Extract")
    for category, fields in FIELD_CATEGORIES.items():
        st.sidebar.subheader(category)
        for field_name, field_info in fields.items():
            help_text = f"""
            Description: {field_info['description']}
            Example: {field_info['example']}
            """
            selected_fields[field_name] = st.sidebar.checkbox(
                field_name,
                value=True,
                help=help_text
            )
    
    # Processing options
    st.sidebar.header("Processing Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence score for field extraction"
    )
    
    return selected_fields, confidence_threshold

def extract_value(text: str, field_name: str, field_info: dict) -> Tuple[Optional[str], float]:
    """Extract value for a specific field with validation"""
    if not text:
        return None, 0.0
    
    validation_pattern = field_info.get('validation')
    if validation_pattern:
        match = re.search(validation_pattern, text)
        if match:
            return match.group(0), 1.0
    
    # Default text processing if no validation pattern or no match
    words = text.split()
    if words:
        return ' '.join(words[:5]), 0.6
    
    return None, 0.0

def process_image_with_fields(
    image: Image.Image,
    model,
    selected_fields: Dict[str, bool],
    confidence_threshold: float
) -> Tuple[np.ndarray, np.ndarray, Dict, List]:
    """Process image with selected fields"""
    try:
        image = preprocess_image(image)
        img_np = np.array(image)
        result = model([img_np])

        height, width = img_np.shape[:2]
        img_all_text = img_np.copy()
        img_fields = img_np.copy()

        # Extract text and create bounding boxes
        extracted_info = {}
        bounding_boxes = []
        word_id = 0

        # Process each word in the document
        for block in result.pages[0].blocks:
            for line in block.lines:
                line_text = ' '.join([word.value for word in line.words])
                
                # Check each selected field
                for field_name, is_selected in selected_fields.items():
                    if not is_selected:
                        continue
                        
                    for category, fields in FIELD_CATEGORIES.items():
                        if field_name in fields:
                            field_info = fields[field_name]
                            if re.search(field_info['pattern'], line_text, re.IGNORECASE):
                                value_text = re.sub(field_info['pattern'], '', line_text, flags=re.IGNORECASE).strip()
                                value, confidence = extract_value(value_text, field_name, field_info)
                                
                                if value and confidence >= confidence_threshold:
                                    extracted_info[field_name] = {
                                        'value': value,
                                        'confidence': confidence,
                                        'category': category
                                    }

                # Create bounding boxes
                for word in line.words:
                    box = create_bounding_box(word, width, height, word_id)
                    bounding_boxes.append(box)
                    word_id += 1

        # Draw bounding boxes
        draw_bounding_boxes(img_all_text, img_fields, bounding_boxes, extracted_info)

        return img_all_text, img_fields, extracted_info, bounding_boxes

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None

def create_bounding_box(word, width: int, height: int, word_id: int) -> dict:
    """Create a bounding box dictionary for a word"""
    x, y = word.geometry[0]
    w, h = word.geometry[1][0] - x, word.geometry[1][1] - y
    return {
        "text": word.value,
        "box": [
            int(x * width),
            int(y * height),
            int((x + w) * width),
            int((y + h) * height)
        ],
        "id": word_id,
        "label": "other",
        "linking": [],
        "words": [{
            "text": word.value,
            "box": [
                int(x * width),
                int(y * height),
                int((x + w) * width),
                int((y + h) * height)
            ]
        }]
    }

def draw_bounding_boxes(img_all_text: np.ndarray, img_fields: np.ndarray, 
                       boxes: List[dict], extracted_info: Dict):
    """Draw bounding boxes on images"""
    for box in boxes:
        x1, y1, x2, y2 = box['box']
        cv2.rectangle(img_all_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_all_text, box['text'], (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        if box['label'] in ['question', 'answer']:
            cv2.rectangle(img_fields, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_fields, f"{box['text']} ({box['label']})",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

def display_results(extracted_info: Dict):
    """Display extracted information in an organized manner"""
    if not extracted_info:
        st.warning("No fields were extracted. Try adjusting the confidence threshold or selecting different fields.")
        return

    # Group results by category
    categorized_results = {}
    for field_name, info in extracted_info.items():
        category = info['category']
        if category not in categorized_results:
            categorized_results[category] = []
        categorized_results[category].append((field_name, info))

    # Display results by category
    for category, fields in categorized_results.items():
        st.subheader(category)
        for field_name, info in fields:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{field_name}:**")
            with col2:
                st.write(info['value'])
            with col3:
                st.progress(info['confidence'])


################
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

    if uploaded_file is not None:
        # Process image
        image = Image.open(uploaded_file)
        results = process_image_with_fields(
            image, model, selected_fields, confidence_threshold
        )

        if all(result is not None for result in results):
            img_all_text, img_fields, extracted_info, bounding_boxes = results

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Document with Detection")
                st.image(img_all_text)

            with col2:
                st.subheader("Extracted Fields")
                st.image(img_fields)

            # Display extracted information
            st.header("Extracted Information")
            display_results(extracted_info)

            # Download buttons
            st.subheader("Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "Download Field Values",
                    data=json.dumps(extracted_info, indent=4),
                    file_name="extracted_fields.json",
                    mime="application/json"
                )
            
            with col2:
                st.download_button(
                    "Download Bounding Boxes",
                    data=json.dumps({"form": bounding_boxes}, indent=4),
                    file_name="bounding_boxes.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()