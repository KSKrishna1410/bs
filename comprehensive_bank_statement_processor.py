#!/usr/bin/env python3
"""
Comprehensive Bank Statement Processor

This module combines header extraction and table extraction capabilities
to process bank statements and return results in a standardized JSON format.
"""

import os
import json
import uuid
import pandas as pd
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from bank_statement_header_extractor import BankStatementHeaderExtractor
from bank_statement_extractor import BankStatementExtractor


class ComprehensiveBankStatementProcessor:
    """
    Comprehensive processor that extracts both headers and transaction tables
    from bank statements and returns results in standardized JSON format.
    """
    
    def __init__(self, output_dir="comprehensive_output"):
        """
        Initialize the comprehensive processor.
        
        Args:
            output_dir (str): Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize sub-processors
        self.header_extractor = BankStatementHeaderExtractor(
            output_dir=os.path.join(output_dir, "headers")
        )
        self.table_extractor = BankStatementExtractor(
            output_dir=os.path.join(output_dir, "tables")
        )
        
    def _generate_process_id(self) -> str:
        """Generate a unique process ID."""
        return str(uuid.uuid4())
    
    def _extract_raw_text_from_page(self, pdf_path: str, page_num: int) -> str:
        """
        Extract raw text from a specific page of the PDF.
        
        Args:
            pdf_path (str): Path to PDF file
            page_num (int): Page number (0-based)
            
        Returns:
            str: Raw text from the page
        """
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                doc.close()
                return ""
            
            page = doc[page_num]
            raw_text = page.get_text()
            doc.close()
            return raw_text.strip()
        except Exception as e:
            print(f"❌ Error extracting text from page {page_num + 1}: {e}")
            return ""
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get the total number of pages in the PDF."""
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            print(f"❌ Error getting page count: {e}")
            return 0
    
    def _convert_headers_to_page_format(self, headers: Dict[str, Any]) -> List[Dict]:
        """
        Convert extracted headers to the page-wise headerInfo format.
        
        Args:
            headers (dict): Headers from BankStatementHeaderExtractor
            
        Returns:
            List[Dict]: Headers in page format
        """
        header_info = []
        
        for field_name, field_data in headers.items():
            value = field_data.get('value', '')
            # Convert value to string to handle both string and integer values (like Account Length)
            value_str = str(value) if value is not None else ''
            if value_str and value_str.strip():  # Only include non-empty values
                header_info.append({
                    "key": field_name,
                    "value": value_str,
                    "key_bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],  # Placeholder bbox
                    "value_bbox": [[0, 0], [100, 0], [100, 20], [0, 20]],  # Placeholder bbox
                    "method": field_data.get('method', 'extracted'),
                    "doc_text": field_data.get('key_text', field_name),
                    "confidence": field_data.get('confidence', 0.0)
                })
        
        return header_info
    
    def _convert_table_to_line_info(self, table_df: pd.DataFrame) -> Dict:
        """
        Convert table DataFrame to lineInfo format.
        
        Args:
            table_df (pd.DataFrame): Extracted table data
            
        Returns:
            Dict: Line info in the required format
        """
        if table_df is None or table_df.empty:
            return {}
        
        # Convert DataFrame to list of lists for lineData
        line_data = []
        
        # Add header row
        headers = table_df.columns.tolist()
        line_data.append(headers)
        
        # Add data rows
        for _, row in table_df.iterrows():
            row_data = []
            for value in row:
                if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                    row_data.append("null")
                else:
                    row_data.append(str(value))
            line_data.append(row_data)
        
        # Generate table info (column positions)
        table_info = []
        for i, header in enumerate(headers):
            table_info.append({
                "key": header,
                "position": i + 1,
                "coordinates": [i * 200, (i + 1) * 200]  # Placeholder coordinates
            })
        
        return {
            "lineData": line_data,
            "tableInfo": table_info,
            "excludeLine": [],  # Placeholder for excluded lines
            "tablePosition": [[0, len(line_data) * 30], [0, None]]  # Placeholder position
        }
    
    def _detect_document_type(self, raw_text: str) -> str:
        """
        Detect document type based on content.
        
        Args:
            raw_text (str): Raw text from the page
            
        Returns:
            str: Document type
        """
        text_lower = raw_text.lower()
        
        # Bank statement indicators
        bank_indicators = [
            'bank statement', 'account statement', 'transaction', 'balance',
            'withdrawal', 'deposit', 'ifsc', 'account number', 'bank'
        ]
        
        # Invoice indicators  
        invoice_indicators = [
            'invoice', 'bill', 'amount due', 'tax', 'gst', 'total amount',
            'payment', 'due date'
        ]
        
        bank_score = sum(1 for indicator in bank_indicators if indicator in text_lower)
        invoice_score = sum(1 for indicator in invoice_indicators if indicator in text_lower)
        
        if bank_score > invoice_score:
            return "BANKSTMT"
        elif invoice_score > 0:
            return "INVOICE"
        else:
            return "BANKSTMT"  # Default to bank statement
    
    def _determine_payment_status(self, raw_text: str) -> Optional[str]:
        """Determine payment status from text content."""
        text_lower = raw_text.lower()
        
        if any(term in text_lower for term in ['paid', 'payment received', 'settled']):
            return "PAID"
        elif any(term in text_lower for term in ['unpaid', 'due', 'outstanding', 'pending']):
            return "UNPAID"
        else:
            return None
    
    def _has_tax_included(self, raw_text: str) -> bool:
        """Check if tax is included based on content."""
        text_lower = raw_text.lower()
        return any(term in text_lower for term in ['gst', 'tax', 'vat', 'service tax', 'incl tax'])
    
    def _convert_table_to_tabula_format(self, table_df: pd.DataFrame) -> List[List[str]]:
        """
        Convert table DataFrame to tabula-style format for lineTabulaData.
        
        Args:
            table_df (pd.DataFrame): Extracted table data
            
        Returns:
            List[List[str]]: Table data in tabula format
        """
        if table_df is None or table_df.empty:
            return []
        
        tabula_data = []
        
        # Add header row
        headers = table_df.columns.tolist()
        tabula_data.append(headers)
        
        # Add data rows
        for _, row in table_df.iterrows():
            row_data = []
            for value in row:
                if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                    row_data.append(" ")  # Use space instead of null for tabula format
                else:
                    row_data.append(str(value).strip())
            tabula_data.append(row_data)
        
        return tabula_data
    
    def process_bank_statement(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a bank statement PDF and return comprehensive results.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Comprehensive results in standardized JSON format
        """
        if not os.path.exists(pdf_path):
            return {
                "status_code": 404,
                "status": "Error",
                "message": f"File not found: {pdf_path}"
            }
        
        try:
            print(f"🏦 Processing bank statement: {os.path.basename(pdf_path)}")
            
            # Generate process ID and metadata
            process_id = self._generate_process_id()
            file_name = os.path.basename(pdf_path)
            base_name = os.path.splitext(file_name)[0]
            
            # Create process-specific directories
            process_dir = os.path.join(self.output_dir, process_id)
            os.makedirs(process_dir, exist_ok=True)
            
            # Get page count
            page_count = self._get_page_count(pdf_path)
            if page_count == 0:
                return {
                    "status_code": 400,
                    "status": "Error", 
                    "message": "Could not read PDF or PDF has no pages"
                }
            
            print(f"📄 PDF has {page_count} pages")
            
            # Extract headers (document-level)
            print("🔍 Extracting headers...")
            headers = self.header_extractor.extract_headers(pdf_path)
            
            # Extract transaction table (document-level)
            print("📊 Extracting transaction table...")
            transaction_table = self.table_extractor.extract_bank_statement_table(pdf_path)
            
            # Process each page
            print("📑 Processing individual pages...")
            page_wise_data = []
            
            for page_num in range(page_count):
                print(f"   Processing page {page_num + 1}/{page_count}...")
                
                # Extract raw text for this page
                raw_text = self._extract_raw_text_from_page(pdf_path, page_num)
                
                # Detect document type for this page
                doc_type = self._detect_document_type(raw_text)
                
                # Convert headers to page format (only for first page or pages with headers)
                header_info = []
                if page_num == 0:  # Main headers usually on first page
                    header_info = self._convert_headers_to_page_format(headers)
                
                # Determine payment status and tax inclusion
                payment_status = self._determine_payment_status(raw_text)
                has_tax = self._has_tax_included(raw_text)
                
                # Generate line info (simplified - could be enhanced with page-specific table extraction)
                line_info = {}
                if transaction_table is not None and not transaction_table.empty and page_num <= 1:
                    # Include table info on first couple of pages
                    line_info = self._convert_table_to_line_info(transaction_table)
                
                # Create page directories
                page_dir = os.path.join(process_dir, f"page{page_num + 1}")
                os.makedirs(page_dir, exist_ok=True)
                
                page_data = {
                    "page": page_num + 1,
                    "identified_doc_type": doc_type,
                    "rawtext": raw_text,
                    "headerInfo": header_info,
                    "paymentSts": payment_status,
                    "incl_Tax": has_tax if payment_status else None,
                    "lineInfo": line_info,
                    "pageWiseFilePath": page_dir,
                    "pageWisedocPath": os.path.join(page_dir, f"{base_name}-{page_num + 1}.pdf")
                }
                
                page_wise_data.append(page_data)
            
            # Convert transaction table to tabula format
            print("🔄 Converting table to tabula format...")
            tabula_data = []
            if transaction_table is not None and not transaction_table.empty:
                tabula_data = self._convert_table_to_tabula_format(transaction_table)
            
            # Build comprehensive response
            response = {
                "status_code": 200,
                "status": "Success",
                "data": {
                    "processId": process_id,
                    "filePath": f"/files/inHouseOCR/{process_id}/{file_name}",
                    "fileDir": f"/files/inHouseOCR/{process_id}",
                    "document_type": "BANKSTMT",
                    "page_cnt": page_count,
                    "isSingleDoc": True,
                    "obj_Type": "SINGLE_DOC_OBJ",
                    "fileType": "System-generated",
                    "pageWiseData": page_wise_data,
                    "lineTabulaData": tabula_data
                }
            }
            
            # Save response to file
            output_file = os.path.join(process_dir, f"{base_name}_response.json")
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=2)
            
            print(f"✅ Processing complete!")
            print(f"💾 Response saved to: {output_file}")
            print(f"📊 Extracted {len(headers)} header fields")
            print(f"📋 Extracted {len(tabula_data)} table rows")
            
            return response
            
        except Exception as e:
            print(f"❌ Error processing bank statement: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status_code": 500,
                "status": "Error",
                "message": f"Processing failed: {str(e)}"
            }
        
        finally:
            # Cleanup temporary files (but only after successful processing)
            try:
                # Only cleanup if processing was successful
                if 'response' in locals() and response.get('status_code') == 200:
                    self.header_extractor.table_extractor.cleanup_all_temp_files()
                    self.table_extractor.table_extractor.cleanup_all_temp_files()
            except:
                pass
    
    def process_multiple_files(self, pdf_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_files (list): List of PDF file paths
            
        Returns:
            dict: Dictionary mapping file names to processing results
        """
        results = {}
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                file_name = os.path.basename(pdf_file)
                print(f"\n{'='*60}")
                print(f"Processing: {file_name}")
                print(f"{'='*60}")
                
                result = self.process_bank_statement(pdf_file)
                results[file_name] = result
                
                if result.get("status_code") == 200:
                    data = result.get("data", {})
                    header_count = len([p for p in data.get("pageWiseData", []) if p.get("headerInfo")])
                    table_rows = len(data.get("lineTabulaData", []))
                    print(f"✅ Success: {header_count} header pages, {table_rows} table rows")
                else:
                    print(f"❌ Failed: {result.get('message', 'Unknown error')}")
            else:
                print(f"⚠️ File not found: {pdf_file}")
                results[os.path.basename(pdf_file)] = {
                    "status_code": 404,
                    "status": "Error",
                    "message": f"File not found: {pdf_file}"
                }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the comprehensive processor
    processor = ComprehensiveBankStatementProcessor()
    
    # Test with available files
    test_files = []
    bankstmt_dir = "BankStatements SK2"
    
    if os.path.exists(bankstmt_dir):
        test_files = [
            os.path.join(bankstmt_dir, file) 
            for file in os.listdir(bankstmt_dir) 
            if file.endswith(".pdf")
        ][:3]  # Test first 3 files
    
    if test_files:
        print(f"📁 Found {len(test_files)} PDF files for testing:")
        for i, file in enumerate(test_files):
            print(f"   {i+1}. {os.path.basename(file)}")
        
        # Process the files
        results = processor.process_multiple_files(test_files)
        
        # Summary
        print(f"\n📊 PROCESSING SUMMARY:")
        for file_name, result in results.items():
            status = result.get("status", "Unknown")
            print(f"   {file_name}: {status}")
            
    else:
        # Test with a single file if available
        single_test = "BankStatements SK2/axis_bank__statement_for_september_2024_unlocked.pdf"
        if os.path.exists(single_test):
            print(f"🧪 Testing with single file: {os.path.basename(single_test)}")
            result = processor.process_bank_statement(single_test)
            
            if result.get("status_code") == 200:
                print("✅ Single file test successful!")
            else:
                print(f"❌ Single file test failed: {result.get('message')}")
        else:
            print("❌ No test files found")
            print("Available files:")
            try:
                for file in os.listdir("BankStatements SK2"):
                    if file.endswith(".pdf"):
                        print(f"  - {file}")
            except:
                print("  Could not list directory") 