from setuptools import setup, find_packages

setup(
    name="ocr-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.22.0",
        "python-doctr[torch]>=0.6.0",
        "pillow>=9.5.0",
        "opencv-python-headless>=4.7.0.72",
        "numpy>=1.23.5",
    ],
    author="Laxman",
    author_email="your.email@example.com",
    description="A Streamlit application for OCR processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
