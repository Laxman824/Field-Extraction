import streamlit as st
from PIL import Image
import pdf2image
import pytesseract
from pathlib import Path
import tempfile
import os
from typing import List, Dict, Union, Tuple
import fitz  # PyMuPDF for PDF handling
from concurrent.futures import ThreadPoolExecutor
import traceback

class DocumentHandler:
    SUPPORTED_FORMATS = {
        'images': ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'],
        'documents': ['.pdf']
    }
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.temp_dir = tempfile.mkdtemp()
        
    def is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS['images'] + self.SUPPORTED_FORMATS['documents']
    
    def convert_to_images(self, file: Union[str, bytes], file_name: str) -> List[Image.Image]:
        """Convert various file formats to PIL Images"""
        try:
            ext = Path(file_name).suffix.lower()
            
            # Handle PDFs
            if ext == '.pdf':
                return self._convert_pdf_to_images(file)
            
            # Handle TIFF (multiple pages)
            elif ext in ['.tiff', '.tif']:
                return self._convert_tiff_to_images(file)
            
            # Handle regular image formats
            elif ext in self.SUPPORTED_FORMATS['images']:
                image = Image.open(file)
                return [image.convert('RGB')]
            
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            st.error(f"Error converting {file_name}: {str(e)}")
            traceback.print_exc()
            return []

    def _convert_pdf_to_images(self, file) -> List[Image.Image]:
        """Convert PDF to list of images"""
        try:
            # Save PDF to temporary file if it's bytes
            if isinstance(file, bytes):
                temp_pdf = os.path.join(self.temp_dir, 'temp.pdf')
                with open(temp_pdf, 'wb') as f:
                    f.write(file)
                file = temp_pdf

            # Convert PDF to images
            pdf_document = fitz.open(file)
            images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
            return images
            
        except Exception as e:
            st.error(f"Error converting PDF: {str(e)}")
            return []

    def _convert_tiff_to_images(self, file) -> List[Image.Image]:
        """Convert TIFF to list of images"""
        try:
            tiff_image = Image.open(file)
            images = []
            
            for i in range(tiff_image.n_frames):
                tiff_image.seek(i)
                images.append(tiff_image.copy().convert('RGB'))
                
            return images
            
        except Exception as e:
            st.error(f"Error converting TIFF: {str(e)}")
            return []

    def process_batch(self, 
                     files: List[Tuple[bytes, str]], 
                     ocr_model,
                     selected_fields: Dict[str, bool],
                     confidence_threshold: float) -> Dict:
        """Process multiple documents in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for file_content, file_name in files:
                if self.is_supported_format(file_name):
                    future = executor.submit(
                        self.process_single_document,
                        file_content,
                        file_name,
                        ocr_model,
                        selected_fields,
                        confidence_threshold
                    )
                    futures.append((file_name, future))
                else:
                    st.warning(f"Unsupported format for file: {file_name}")
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Process results as they complete
            for idx, (file_name, future) in enumerate(futures):
                try:
                    result = future.result()
                    results[file_name] = result
                    progress_bar.progress((idx + 1) / len(futures))
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                    results[file_name] = {"error": str(e)}
            
            progress_bar.empty()
        
        return results

    def process_single_document(self,
                              file_content: bytes,
                              file_name: str,
                              ocr_model,
                              selected_fields: Dict[str, bool],
                              confidence_threshold: float) -> Dict:
        """Process a single document"""
        try:
            # Convert document to images
            images = self.convert_to_images(file_content, file_name)
            
            if not images:
                raise ValueError("No images extracted from document")
            
            # Process each image
            page_results = []
            for idx, image in enumerate(images):
                result = process_image_with_fields(
                    image,
                    ocr_model,
                    selected_fields,
                    confidence_threshold
                )
                page_results.append({
                    "page_number": idx + 1,
                    "results": result
                })
            
            return {
                "status": "success",
                "pages": page_results,
                "total_pages": len(images)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"Error cleaning up temporary files: {str(e)}")