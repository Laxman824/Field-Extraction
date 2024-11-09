import streamlit as st
from doctr.models import ocr_predictor

@st.cache_resource
def load_model():
    return ocr_predictor(pretrained=True)
