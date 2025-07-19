"""
Global settings and configuration variables for bank statement processing
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"
MASTER_CSV_DIR = BASE_DIR / "master_csv"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
MASTER_CSV_DIR.mkdir(exist_ok=True)

# OCR settings
OCR_OUTPUT_DIR = DATA_DIR / "ocr_outputs"
OCR_OUTPUT_DIR.mkdir(exist_ok=True)

# File patterns
VALID_BANK_STATEMENT_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']

# Bank statement specific settings
BANK_KEYWORDS_FILE = MASTER_CSV_DIR / "bankstmt_allkeys.csv"
IFSC_MASTER_FILE = MASTER_CSV_DIR / "IFSC_master.csv"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = bool(os.getenv("API_DEBUG", "True"))

# Streamlit settings
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Authentication settings
AUTH_CREDENTIALS = {
    "admin": os.getenv("ADMIN_PASSWORD", "admin123"),
    "user": os.getenv("USER_PASSWORD", "user123"),
    "demo": os.getenv("DEMO_PASSWORD", "demo123")
} 