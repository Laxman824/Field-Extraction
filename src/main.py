# src/main.py
import streamlit as st
from utils.image_processing import preprocess_image
from utils.ocr_utils import process_image
from models.model_loader import load_model
from config.field_patterns import FIELDS
import yaml

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

def main():
    st.title(config["app"]["title"])
    st.write(config["app"]["description"])

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=config["app"]["allowed_extensions"]
    )

    if uploaded_file is not None:
        # Process image and display results
        process_and_display_results(uploaded_file, model)

if __name__ == "__main__":
    main()
