#!/usr/bin/env python3
"""
FastAPI application for bank statement processing
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
import json
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import fitz  # PyMuPDF
import pandas as pd

from utils.extraction.table_extractor import DocumentTableExtractor
from bank_statements.extractors.header_extractor import BankStatementHeaderExtractor
from bank_statements.extractors.table_extractor import BankStatementExtractor

class PDFProcessor:
    """
    Handles initial PDF/Image processing including readability checks and conversion.
    
    This class is responsible for:
    1. Checking if a file is readable PDF or image
    2. Converting scanned PDFs/images to readable format
    3. Providing a clean interface for downstream processing
    """
    
    def __init__(self, output_dir: str = "temp"):
        """
        Initialize the PDF processor.
        
        Args:
            output_dir (str): Base directory for all outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize extractors
        self.table_extractor = DocumentTableExtractor(
            output_dir=os.path.join(output_dir, "tables"),
            save_reconstructed_pdfs=True
        )
        
        self.header_extractor = BankStatementHeaderExtractor(
            output_dir=os.path.join(output_dir, "headers"),
            keywords_csv="data/master_csv/bankstmt_allkeys.csv",
            ifsc_master_csv="data/master_csv/IFSC_master.csv"
        )
        
        self.bank_statement_extractor = BankStatementExtractor(
            output_dir=output_dir
        )
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Check if file is a PDF based on extension and content."""
        if not file_path.lower().endswith('.pdf'):
            return False
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return True
        except:
            return False
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF readability and characteristics.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Analysis results including readability status
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Get basic PDF info
            page_count = len(doc)
            total_chars = 0
            total_words = 0
            pages_with_text = 0
            
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages_with_text += 1
                    words = text.split()
                    total_words += len(words)
                    total_chars += len(text)
            
            doc.close()
            
            # Calculate metrics
            chars_per_page = total_chars / page_count if page_count > 0 else 0
            avg_word_length = total_chars / total_words if total_words > 0 else 0
            text_page_ratio = pages_with_text / page_count if page_count > 0 else 0
            
            print(f"📊 PDF Analysis: {total_chars} chars, {total_words} words, avg_len: {avg_word_length:.1f}")
            print(f"📊 Additional checks: {chars_per_page:.1f} chars/page, {text_page_ratio*100:.1f}% pages with text")
            
            # Determine if PDF is readable based on metrics
            is_readable = (chars_per_page > 100 and  # At least 100 chars per page
                         text_page_ratio > 0.5 and   # At least 50% pages have text
                         avg_word_length > 3)        # Average word length > 3 chars
            
            print(f"📊 Readability decision: {'READABLE' if is_readable else 'SCANNED'}")
            
            return {
                'is_readable': is_readable,
                'page_count': page_count,
                'total_chars': total_chars,
                'total_words': total_words,
                'chars_per_page': chars_per_page,
                'avg_word_length': avg_word_length,
                'text_page_ratio': text_page_ratio
            }
            
        except Exception as e:
            print(f"❌ Error analyzing PDF: {str(e)}")
            return {
                'is_readable': False,
                'error': str(e)
            }
    
    def convert_image_to_pdf(self, image_path: str) -> Optional[str]:
        """Convert image to PDF using PIL."""
        try:
            from PIL import Image
            import tempfile
            
            # Open the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create temporary PDF file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                    pdf_path = temp_pdf.name
                    # Save as PDF
                    img.save(pdf_path, 'PDF', resolution=100.0)
                    return pdf_path
                    
        except Exception as e:
            print(f"❌ Error converting image to PDF: {str(e)}")
            return None
    
    def process_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Process a file (PDF or image) and extract information.
        
        Args:
            file_path (str): Path to the file
            session_id (str): Unique session ID for this processing request
            
        Returns:
            dict: Extracted information including headers and tables
        """
        try:
            # Check if it's a PDF or image
            is_pdf = file_path.lower().endswith('.pdf')
            
            if is_pdf:
                # Analyze the PDF
                analysis = self.analyze_pdf(file_path)
                
                # If PDF is not readable, convert it
                if not analysis['is_readable']:
                    print("📄 Converting scanned PDF to readable format...")
                    ocr_data, readable_pdf = self.table_extractor.ocr_processor.ocr_and_reconstruct(file_path)
                    if readable_pdf and os.path.exists(readable_pdf):
                        file_path = readable_pdf
                        print("✅ Conversion successful")
                    else:
                        print("❌ Failed to convert PDF")
                        return {'error': 'Failed to convert scanned PDF'}
            else:
                # For images, directly use OCR
                print("📄 Processing image with OCR...")
                ocr_data, readable_pdf = self.table_extractor.ocr_processor.ocr_and_reconstruct(file_path)
                if readable_pdf and os.path.exists(readable_pdf):
                    file_path = readable_pdf
                    print("✅ OCR successful")
                else:
                    print("❌ Failed to process image")
                    return {'error': 'Failed to process image'}
                
                # Set basic analysis data for images
                analysis = {
                    'is_readable': True,  # After OCR
                    'page_count': 1,      # Single image
                }
            
            # Now extract headers and tables from the readable file
            headers = self.header_extractor.extract_headers(file_path)
            table_df = self.bank_statement_extractor.extract_bank_statement_table(file_path)
            
            # Convert DataFrame to list of lists if not empty
            tables = []
            if table_df is not None and not table_df.empty:
                # Get column names as first row
                tables = [table_df.columns.tolist()]
                # Convert each row to list and append
                tables.extend(table_df.values.tolist())
                
                # Replace NaN/None with empty string
                tables = [['' if pd.isna(cell) else str(cell) for cell in row] for row in tables]
            
            # Convert headers to list format with bbox info
            header_list = []
            for key, value in headers.items():
                # Extract values from the header data
                if isinstance(value, dict):
                    header_value = value.get('value', '')
                    method = value.get('method', '')
                    confidence = value.get('confidence', 0)
                    validation_score = value.get('validation_score', 0)
                    doc_text = value.get('key_text', '')
                else:
                    header_value = value
                    method = ''
                    confidence = 0
                    validation_score = 0
                    doc_text = ''
                
                header_item = {
                    "key": key,
                    "value": header_value,  # Use the extracted value
                    "key_bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],  # Default bbox
                    "value_bbox": [[0, 0], [100, 0], [100, 20], [0, 20]], # Default bbox
                    "method": method,
                    "doc_text": doc_text,
                    "confidence": confidence,
                    "validation_score": validation_score
                }
                header_list.append(header_item)
            
            # Create page-wise data
            page_wise_data = []
            for page_num in range(analysis['page_count']):
                page_data = {
                    "page": page_num + 1,
                    "identified_doc_type": "BANKSTMT",
                    "rawtext": "",  # Raw text will be added by OCR
                    "headerInfo": header_list if page_num == 0 else [],
                    "paymentSts": "UNPAID",
                    "incl_Tax": False,
                    "lineInfo": {
                        "lineData": tables if page_num == 0 else [],
                        "tableInfo": [
                            {
                                "key": col,
                                "position": idx + 1,
                                "coordinates": [idx * 200, (idx + 1) * 200]
                            } for idx, col in enumerate(tables[0] if tables else [])
                        ],
                        "excludeLine": [],
                        "tablePosition": [[0, 1380], [0, None]]
                    } if tables and page_num == 0 else {},
                    "pageWiseFilePath": f"temp/{session_id}/page{page_num + 1}",
                    "pageWisedocPath": f"temp/{session_id}/page{page_num + 1}/{os.path.basename(file_path)}-{page_num + 1}.pdf"
                }
                page_wise_data.append(page_data)
            
            # Create response in exact format
            response = {
                "status_code": 200,
                "status": "Success",
                "data": {
                    "processId": session_id,
                    "filePath": f"/files/inHouseOCR/{session_id}/{os.path.basename(file_path)}",
                    "fileDir": f"/files/inHouseOCR/{session_id}",
                    "document_type": "BANKSTMT",
                    "page_cnt": analysis['page_count'],
                    "isSingleDoc": True,
                    "obj_Type": "SINGLE_DOC_OBJ",
                    "fileType": "System-generated",
                    "pageWiseData": page_wise_data,
                    "lineTabulaData": tables
                }
            }
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return {'error': str(e)}
        
        finally:
            # Cleanup any temporary files
            self.table_extractor.cleanup_all_temp_files()

def _sanitize_value(value):
    """Sanitize values for JSON serialization."""
    if isinstance(value, float):
        if value.is_integer():
            # Convert float to int if it's a whole number
            return str(int(value))
        return str(value)
    elif isinstance(value, (int, bool)):
        return str(value)
    elif value is None:
        return ""
    return str(value)

def _sanitize_data(data):
    """Recursively sanitize all values in a data structure."""
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_data(item) for item in data]
    else:
        return _sanitize_value(data)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Statement OCR API",
    description="API for extracting data from bank statements using OCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF processor
pdf_processor = PDFProcessor()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/ocr_process/")
async def process_document(
    file: UploadFile = File(...),
    output_dir: str = Form(""),
    doctype: str = Form("BANKSTMT")
) -> Dict[str, Any]:
    """
    Process a document through OCR and extract structured data.
    
    Args:
        file: Uploaded file (PDF/Image)
        output_dir: Output directory for results
        doctype: Document type (default: BANKSTMT)
        
    Returns:
        dict: Extracted data including headers and tables
    """
    try:
        # Create unique output directory
        session_id = str(uuid.uuid4())
        if output_dir:
            output_path = os.path.join(output_dir, session_id)
        else:
            output_path = os.path.join("temp", session_id)
        os.makedirs(output_path, exist_ok=True)
        
        # Save uploaded file
        temp_file = None
        try:
            # Get original file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            if not file_ext:
                # Default to .pdf only if no extension found
                file_ext = '.pdf'
            
            # Create temporary file with original extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
            
            print(f"🏦 Processing bank statement: {file.filename}")
            
            # Get page count if it's a PDF
            if file_ext == '.pdf':
                doc = fitz.open(temp_path)
                page_count = len(doc)
                doc.close()
                print(f"📄 PDF has {page_count} pages")
            else:
                print(f"📄 Processing image file: {file_ext}")
            
            # Process the document
            result = pdf_processor.process_file(temp_path, session_id)
            
            # Save response to file
            response_file = os.path.join(output_path, f"{file.filename}_response.json")
            with open(response_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Response saved to: {response_file}")
            
            # Print summary
            if 'data' in result:
                headers = result['data']['pageWiseData'][0]['headerInfo'] if result['data']['pageWiseData'] else []
                tables = result['data']['lineTabulaData']
                print(f"📊 Extracted {len(headers)} header fields")
                print(f"📋 Extracted {len(tables)} table rows")
            
            return result
            
        finally:
            # Cleanup temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 