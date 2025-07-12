#!/usr/bin/env python3
"""
Bank Statement OCR Processing API

FastAPI application that provides OCR processing capabilities for bank statements.
Integrates with the comprehensive bank statement processor.
"""

import os
import json
import tempfile
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from comprehensive_bank_statement_processor import ComprehensiveBankStatementProcessor


# Initialize FastAPI app
app = FastAPI(
    title="Nekkanti OCR API",
    description="API for extracting headers and tables from bank statements",
    version="1.0.0"
)

# Initialize processor
processor = ComprehensiveBankStatementProcessor()


class OCRResponse(BaseModel):
    """Response model for OCR processing"""
    status_code: int
    status: str
    data: dict


@app.post("/ocr_process/")
async def ocr_process_file(
    file: UploadFile = File(..., description="Bank statement file to process (PDF or Image)"),
    output_dir: str = Form("", description="Output directory (should be blank)"),
    doctype: str = Form(..., description="Document type (INVOICE or BANKSTMT)")
):
    """
    Process a bank statement file using OCR and extract headers and tables.
    
    Args:
        file: Uploaded bank statement file (PDF or image format: PNG, JPG, JPEG, BMP, TIFF, TIF)
        output_dir: Output directory (should be blank)
        doctype: Document type - must be "BANKSTMT" for bank statements
    
    Returns:
        JSON response with extracted headers and transaction data
    """
    
    # Validate document type
    if doctype not in ["INVOICE", "BANKSTMT"]:
        raise HTTPException(
            status_code=422,
            detail="Invalid doctype. Must be either 'INVOICE' or 'BANKSTMT'"
        )
    
    # Currently only support BANKSTMT
    if doctype != "BANKSTMT":
        raise HTTPException(
            status_code=422,
            detail="Currently only 'BANKSTMT' document type is supported"
        )
    
    # Validate file type
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    if not any(file.filename.lower().endswith(ext) for ext in supported_extensions):
        raise HTTPException(
            status_code=422,
            detail="Supported file types: PDF, PNG, JPG, JPEG, BMP, TIFF, TIF"
        )
    
    # Create temporary file for processing
    temp_file_path = None
    try:
        # Get file extension from original filename
        import os
        file_extension = os.path.splitext(file.filename)[1].lower()
        if not file_extension:
            file_extension = '.pdf'  # Default fallback
        
        # Create temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
        
        # Process the bank statement
        result = processor.process_bank_statement(temp_file_path)
        
        # Return the response
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except Exception as e:
        # Handle processing errors
        error_response = {
            "status_code": 500,
            "status": "error",
            "message": f"Error processing file: {str(e)}",
            "data": {}
        }
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Nekkanti OCR API",
        "version": "1.0.0",
        "endpoints": {
            "ocr_process": "/ocr_process/ (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running successfully"
    }


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8888,
        reload=True
    ) 