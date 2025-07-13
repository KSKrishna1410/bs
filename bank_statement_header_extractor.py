#!/usr/bin/env python3
"""
Bank Statement Header Extractor - Improved Version

This module extracts ONLY header information from bank statements using:
- Smart filtering to only extract Header field types (not Line/transaction data)
- Improved spatial analysis with stricter constraints  
- Better validation and pattern matching
- Confidence scoring for best matches
"""

import os
import pandas as pd
import json
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from pdf_to_table import DocumentTableExtractor


class BankStatementHeaderExtractor:
    """
    Improved extractor for bank statement headers with better accuracy.
    
    Key improvements:
    - Only extracts fields marked as "Header" type (no transaction data)
    - Better spatial analysis with strict constraints
    - Robust validation with pattern matching
    - Confidence scoring for optimal matches
    """
    
    def __init__(self, output_dir="header_extraction_output", keywords_csv="bankstmt_allkeys.csv", ifsc_master_csv="IFSC_master.csv"):
        """
        Initialize the header extractor.
        
        Args:
            output_dir (str): Directory for output files
            keywords_csv (str): Path to CSV file with keyword mappings
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the table extractor for OCR results
        self.table_extractor = DocumentTableExtractor(
            output_dir=os.path.join(self.output_dir, "temp_ocr"),
            save_reconstructed_pdfs=False
        )
        
        # Load keyword mappings - ONLY HEADER FIELDS
        self.keywords_df = self._load_keywords(keywords_csv)
        self.header_mappings = self._create_header_mappings()
        
        # Load IFSC master database
        self.ifsc_master_df = self._load_ifsc_master(ifsc_master_csv)
        
        # Enhanced configuration for 90%+ accuracy
        self.y_tolerance = 12  # Reduced for more precise spatial matching
        self.x_tolerance = 8   # Reduced for more precise spatial matching  
        self.max_distance_threshold = 80  # Reduced for stricter proximity
        self.min_confidence_score = 0.7  # Raised for higher quality results
        
        # Enhanced scoring weights for better prioritization
        self.spatial_weight = 0.25    # Reduced - smart patterns are more reliable
        self.validation_weight = 0.55  # Increased - validation is critical
        self.ocr_confidence_weight = 0.2  # OCR confidence for tie-breaking
        
    def _load_keywords(self, csv_path: str) -> pd.DataFrame:
        """Load keyword mappings from CSV file - ONLY Header field types."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Keywords CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # CRITICAL FIX: Only load Header field types, ignore Line and Others
        header_df = df[df['field_type'] == 'Header'].copy()
        
        print(f"📋 Loaded {len(header_df)} HEADER-only keyword mappings from {csv_path}")
        print(f"   Ignored {len(df) - len(header_df)} non-header fields (Line/Others)")
        
        return header_df
    
    def _create_header_mappings(self) -> Dict[str, List[Dict]]:
        """Create field mappings organized by field name - ONLY headers."""
        mappings = {}
        
        for _, row in self.keywords_df.iterrows():
            field_name = row['field_name']
            if field_name not in mappings:
                mappings[field_name] = []
            
            mappings[field_name].append({
                'key': row['key'],
                'data_type': row['data_type'], 
                'field_type': row['field_type']  # Will always be 'Header'
            })
        
        print(f"🗂️ Created mappings for {len(mappings)} header field types:")
        for field_name in mappings.keys():
            print(f"   • {field_name}")
        
        return mappings
    
    def _load_ifsc_master(self, ifsc_csv_path: str) -> pd.DataFrame:
        """Load IFSC master database for bank lookup."""
        try:
            if not os.path.exists(ifsc_csv_path):
                print(f"⚠️ IFSC master CSV file not found: {ifsc_csv_path}")
                return pd.DataFrame()
            
            print(f"📋 Loading IFSC master database from {ifsc_csv_path}")
            df = pd.read_csv(ifsc_csv_path)
            
            # Clean up the data
            df = df.dropna(subset=['IFSC'])  # Remove entries without IFSC
            df['IFSC'] = df['IFSC'].str.upper().str.strip()  # Normalize IFSC codes
            
            # Handle the account length column - try both possible column names
            account_length_col = None
            if 'A/c Length (Min)' in df.columns:
                account_length_col = 'A/c Length (Min)'
            elif 'A/c Length (Min).1' in df.columns:
                account_length_col = 'A/c Length (Min).1'
            elif 'Account Length (Min)' in df.columns:
                account_length_col = 'Account Length (Min)'
            
            # Clean up account length data if available
            if account_length_col:
                df[account_length_col] = pd.to_numeric(df[account_length_col], errors='coerce')
                df[account_length_col] = df[account_length_col].fillna(0).astype(int)
                print(f"✅ Found account length column: {account_length_col}")
            else:
                print("⚠️ No account length column found in IFSC master")
                df['A/c Length (Min)'] = 0  # Default to 0 if not found
                account_length_col = 'A/c Length (Min)'
            
            # Standardize column name
            if account_length_col != 'A/c Length (Min)':
                df['A/c Length (Min)'] = df[account_length_col]
            
            # Create a lookup dictionary for faster access
            self.ifsc_lookup = df.set_index('IFSC').to_dict('index')
            
            print(f"✅ Loaded {len(df)} IFSC entries from {len(df['BANK'].unique())} banks")
            return df
            
        except Exception as e:
            print(f"❌ Error loading IFSC master database: {e}")
            return pd.DataFrame()
    
    def _lookup_bank_details_by_ifsc(self, ifsc_code: str) -> Dict[str, Any]:
        """
        Look up bank details using IFSC code.
        
        Args:
            ifsc_code (str): IFSC code to lookup
            
        Returns:
            Dict containing bank details or empty dict if not found
        """
        if not hasattr(self, 'ifsc_lookup') or not self.ifsc_lookup:
            return {}
            
        ifsc_code = ifsc_code.upper().strip()
        
        if ifsc_code in self.ifsc_lookup:
            bank_info = self.ifsc_lookup[ifsc_code]
            
            # Helper function to clean values and handle NaN
            def clean_value(val):
                if pd.isna(val) or val is None:
                    return ''
                val_str = str(val).strip()
                if val_str.lower() == 'nan':
                    return ''
                return val_str
            
            # Get account length (ensure it's an integer)
            account_length = bank_info.get('A/c Length (Min)', 0)
            if pd.isna(account_length):
                account_length = 0
            else:
                account_length = int(account_length)
            
            return {
                'Bank Name': clean_value(bank_info.get('BANK', '')),
                'Bank Branch': clean_value(bank_info.get('BRANCH', '')),
                'Bank Address': clean_value(bank_info.get('ADDRESS', '')),
                'Bank City': clean_value(bank_info.get('CITY1', '')),
                'Bank State': clean_value(bank_info.get('STATE', '')),
                'Bank Phone': clean_value(bank_info.get('PHONE', '')),
                'Account Length': account_length  # Add account length to the returned info
            }
        
        return {}
    
    def _enhance_headers_with_ifsc_lookup(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance extracted headers with IFSC-based bank lookup.
        
        Args:
            headers (dict): Original extracted headers
            
        Returns:
            Dict with enhanced headers including bank details
        """
        enhanced_headers = headers.copy()
        
        # Look for IFSC code in extracted headers
        ifsc_code = None
        ifsc_field_key = None
        
        for field_name, field_data in headers.items():
            if field_name == 'IFSC Code' and field_data:
                if isinstance(field_data, dict):
                    ifsc_code = field_data.get('value', '')
                else:
                    ifsc_code = str(field_data)
                ifsc_field_key = field_name
                break
        
        if ifsc_code:
            print(f"🔍 Looking up bank details for IFSC: {ifsc_code}")
            bank_details = self._lookup_bank_details_by_ifsc(ifsc_code)
            
            if bank_details:
                print(f"✅ Found bank details for {ifsc_code}:")
                for key, value in bank_details.items():
                    if value:  # Only add non-empty values
                        enhanced_headers[key] = {
                            'value': value,
                            'key_text': ifsc_code,
                            'keyword': 'IFSC lookup',
                            'data_type': 'String',
                            'field_type': 'Header',
                            'method': 'ifsc_lookup',
                            'distance': 0.0,
                            'confidence': 0.99,  # High confidence for IFSC lookup
                            'validation_score': 0.98,
                            'spatial_score': 1.0
                        }
                        print(f"   • {key}: {value}")
            else:
                print(f"❌ No bank details found for IFSC: {ifsc_code}")
        else:
            print(f"ℹ️ No IFSC code found in headers - cannot perform bank lookup")
        
        return enhanced_headers
    
    def _get_text_and_positions(self, pdf_path: str) -> List[Tuple[str, List[List[float]], float]]:
        """Get text and positions from PDF using the most efficient method."""
        try:
            # Check if PDF is readable using our existing logic
            is_readable = self.table_extractor._is_pdf_readable(pdf_path)
            
            if is_readable:
                print(f"📄 PDF is readable - extracting text directly")
                return self._extract_text_from_readable_pdf(pdf_path)
            else:
                print(f"📄 PDF is scanned - converting to readable first")
                return self._extract_text_from_scanned_pdf(pdf_path)
                
        except Exception as e:
            print(f"❌ Error getting text and positions: {e}")
            return []
    
    def _extract_text_from_readable_pdf(self, pdf_path: str) -> List[Tuple[str, List[List[float]], float]]:
        """Extract text and positions directly from readable PDF."""
        import fitz  # PyMuPDF
        
        text_elements = []
        
        try:
            doc = fitz.open(pdf_path)
            
            # Only process first 2 pages for headers (headers are usually on first page)
            max_pages = min(2, len(doc))
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # Get text blocks with positions
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text and len(text) > 1:  # Skip very short text
                                    # Get bounding box
                                    bbox = span["bbox"]  # (x0, y0, x1, y1)
                                    
                                    # Convert to our format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    x0, y0, x1, y1 = bbox
                                    bbox_formatted = [
                                        [x0, y0],  # Top-left
                                        [x1, y0],  # Top-right
                                        [x1, y1],  # Bottom-right
                                        [x0, y1]   # Bottom-left
                                    ]
                                    
                                    confidence = 1.0  # High confidence for direct text extraction
                                    text_elements.append((text, bbox_formatted, confidence))
            
            doc.close()
            print(f"📄 Extracted {len(text_elements)} text elements from first {max_pages} page(s)")
            return text_elements
            
        except Exception as e:
            print(f"❌ Error extracting text from readable PDF: {e}")
            return []
    
    def _extract_text_from_scanned_pdf(self, pdf_path: str) -> List[Tuple[str, List[List[float]], float]]:
        """Extract text from scanned PDF by first making it readable."""
        try:
            # Use NekkantiOCR to make the PDF readable
            ocr_data, reconstructed_pdf_path = self.table_extractor.ocr_processor.ocr_and_reconstruct(pdf_path)
            
            if reconstructed_pdf_path and os.path.exists(reconstructed_pdf_path):
                print(f"📄 PDF reconstructed, now extracting text from readable version")
                return self._extract_text_from_readable_pdf(reconstructed_pdf_path)
            else:
                print(f"❌ Failed to reconstruct PDF")
                return []
                
        except Exception as e:
            print(f"❌ Error processing scanned PDF: {e}")
            return []
    
    def _calculate_distance(self, bbox1: List[List[float]], bbox2: List[List[float]]) -> float:
        """Calculate Euclidean distance between two bounding box centers."""
        x1, y1 = (bbox1[0][0] + bbox1[2][0]) / 2, (bbox1[0][1] + bbox1[2][1]) / 2
        x2, y2 = (bbox2[0][0] + bbox2[2][0]) / 2, (bbox2[0][1] + bbox2[2][1]) / 2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _clean_text_for_matching(self, text: str) -> str:
        """Clean and normalize text for keyword matching."""
        # Remove special characters but keep spaces for multi-word matching
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned
    
    def _add_smart_pattern_matches(self, text_results: List[Tuple], key_matches: List[Dict]) -> None:
        """
        Enhanced smart pattern-based matches for 90%+ accuracy.
        Covers multiple bank statement formats and edge cases.
        """
        # print("🔍 Looking for enhanced smart patterns...")
        
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            text_lower = text.lower()
            
            # Pattern 1: Statement date patterns (multiple formats)
            date_patterns = [
                # "as on date DD-MM-YYYY" or "as on DD-MM-YYYY"
                (r'as\s+on\s+(date\s+)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Statement Date'),
                # "statement as on DD-MM-YYYY"
                (r'statement\s+as\s+on\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Statement Date'),
                # "for the period ending DD-MM-YYYY"
                (r'period\s+ending\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Statement Date'),
                # "statement date: DD-MM-YYYY"
                (r'statement\s+date[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Statement Date'),
                # "date of statement: DD-MM-YYYY"
                (r'date\s+of\s+statement[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Statement Date'),
            ]
            
            for pattern, field_name in date_patterns:
                if re.search(pattern, text_lower):
                    key_matches.append({
                        'field_name': field_name,
                        'matched_text': text,
                        'keyword': 'date pattern',
                        'bbox': bbox,
                        'data_type': 'Date',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.95,
                        'text_index': text_idx
                    })
            
            # Pattern 2: Enhanced date range patterns
            range_patterns = [
                # "between DD-MM-YYYY to DD-MM-YYYY"
                (r'between\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "from DD-MM-YYYY to DD-MM-YYYY"
                (r'from\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "DD-MM-YYYY TO DD-MM-YYYY"
                (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "statement period: DD-MM-YYYY to DD-MM-YYYY"
                (r'statement\s+period[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "period: DD-MM-YYYY to DD-MM-YYYY"
                (r'period[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
            ]
            
            for pattern, field_type in range_patterns:
                if re.search(pattern, text_lower):
                    if field_type == 'both':
                        # Add both FROM and TO matches
                        for suffix, field_name in [('From', 'Statement Date From'), ('To', 'Statement Date To')]:
                            key_matches.append({
                                'field_name': field_name,
                                'matched_text': text,
                                'keyword': 'date range pattern',
                                'bbox': bbox,
                                'data_type': 'Date',
                                'field_type': 'Header',
                                'confidence': confidence,
                                'match_score': 0.95,
                                'text_index': text_idx
                            })
            
            # Pattern 3: Enhanced account number patterns with better detection
            account_patterns = [
                # "Account No. XXXXXXXXXXX6582" - Fixed for case insensitive matching
                (r'account\s+no\.?\s+[Xx]+\d{3,8}', 'Account Number'),
                # "A/C No. XXXXXXXXXXX6582" or "A/C No XXXXXXXXXXX6582"  
                (r'a/c\s+no\.?\s+[Xx]+\d{3,8}', 'Account Number'),
                # "Account Number: 322205001059"
                (r'account\s+number[:\s]+\d{8,20}', 'Account Number'),
                # "Account No 6613041256" - Fixed for Kotak format
                (r'account\s+no\.?\s+\d{8,20}', 'Account Number'),
                # "A/C: 1234567890123456"
                (r'a/c[:\s]+\d{8,20}', 'Account Number'),
                # "Account: XXXXXXXXXXX6582"
                (r'account[:\s]+[Xx]+\d{3,8}', 'Account Number'),
                # "Detailed Statement for a/c no. XXXXXXXXXXX6582" - Fixed for Axis Bank
                (r'detailed\s+statement\s+for\s+a/c\s+no\.\s+[Xx]+\d{3,8}', 'Account Number'),
                # "Statement for a/c no. XXXXXXXXXXX6582" - Additional pattern
                (r'statement\s+for\s+a/c\s+no\.\s+[Xx]+\d{3,8}', 'Account Number'),
                # "Account Name: [Name] Account Number: [Number]"
                (r'account\s+name:.*account\s+number[:\s]+\d{8,20}', 'Account Number'),
                # "A/C No XXXXXXXXXXX6582" (without punctuation)
                (r'a/c\s+no\s+[Xx]+\d{3,8}', 'Account Number'),
                # "For Account Number: [Number]"
                (r'for\s+account\s+number[:\s]+\d{8,20}', 'Account Number'),
                # More flexible patterns as fallbacks
                (r'account.*no.*[Xx]+\d{3,8}', 'Account Number'),
                (r'a/c.*no.*[Xx]+\d{3,8}', 'Account Number'),
            ]
            
            for pattern, field_name in account_patterns:
                if re.search(pattern, text_lower):
                    key_matches.append({
                        'field_name': field_name,
                        'matched_text': text,
                        'keyword': 'account pattern',
                        'bbox': bbox,
                        'data_type': 'String',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.9,
                        'text_index': text_idx
                    })
            
            # Pattern 4: Enhanced IFSC code patterns with contamination filtering
            # Only match IFSC codes that are NOT in transaction context
            ifsc_patterns = [
                # "IFSC CODE : IOBA0000384"
                (r'ifsc\s*code\s*[:\s]*[A-Z]{4}\d{7}', 'IFSC Code'),
                # "IFSC: HDFC0001628"
                (r'ifsc[:\s]+[A-Z]{4}\d{7}', 'IFSC Code'),
                # "RTGS/NEFT IFSC: HDFC0001628"
                (r'rtgs/?neft\s*ifsc[:\s]*[A-Z]{4}\d{7}', 'IFSC Code'),
                # "IFS Code: SBIN0016345"
                (r'ifs\s*code[:\s]*[A-Z]{4}\d{7}', 'IFSC Code'),
                # Standalone IFSC - but only if not in transaction context
                (r'\b[A-Z]{4}\d{7}\b', 'IFSC Code'),
            ]
            
            for pattern, field_name in ifsc_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # CRITICAL: Filter out transaction-related IFSC codes
                    transaction_indicators = [
                        'neft', 'rtgs', 'imps', 'upi', 'transfer', 'payment', 'transaction',
                        'credit', 'debit', 'dr', 'cr', 'withdrawal', 'deposit',
                        # Date patterns indicating transactions
                        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                        # Amount patterns
                        r'[\d,]+\.\d{2}', r'rs\.?\s*[\d,]+', r'₹\s*[\d,]+'
                    ]
                    
                    # Check if this text contains transaction indicators
                    is_transaction_context = any(
                        re.search(indicator, text_lower) for indicator in transaction_indicators
                    )
                    
                    # Also check if it's a very long line (likely transaction)
                    is_long_transaction_line = len(text) > 100
                    
                    # Skip if it looks like a transaction
                    if is_transaction_context or is_long_transaction_line:
                        # print(f"   🚫 Skipping transaction IFSC: '{text[:50]}...'")
                        continue
                    
                    key_matches.append({
                        'field_name': field_name,
                        'matched_text': text,
                        'keyword': 'IFSC pattern',
                        'bbox': bbox,
                        'data_type': 'String',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.95,
                        'text_index': text_idx
                    })
            
            # Pattern 5: Enhanced branch name patterns
            branch_patterns = [
                # "Branch Name : Secunderabad"
                (r'branch\s+name\s*[:\s]+[A-Za-z\s\-&]{3,30}', 'Bank Branch'),
                # "Branch: HYDERABAD - BANJARA HILLS"
                (r'branch[:\s]+[A-Za-z\s\-&]{3,30}', 'Bank Branch'),
                # "Home Branch: [Name]"
                (r'home\s+branch[:\s]+[A-Za-z\s\-&]{3,30}', 'Bank Branch'),
                # "BRN -Branch: [Name]" (like in Axis)
                (r'brn\s*-?\s*branch[:\s]+[A-Za-z\s\-&]{3,30}', 'Bank Branch'),
                # "Account Branch : [Name]" (like in HDFC)
                (r'account\s+branch[:\s]+[A-Za-z\s\-&]{3,30}', 'Bank Branch'),
            ]
            
            for pattern, field_name in branch_patterns:
                if re.search(pattern, text_lower) and len(text) < 60:  # Reasonable length for branch names
                    key_matches.append({
                        'field_name': field_name,
                        'matched_text': text,
                        'keyword': 'branch pattern',
                        'bbox': bbox,
                        'data_type': 'String',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.85,
                        'text_index': text_idx
                    })
                
            # Pattern 6: Enhanced balance patterns with better detection
            balance_patterns = [
                # Exact balance labels with amounts
                (r'opening\s+balance[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'closing\s+balance[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                (r'opening\s+bal[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'closing\s+bal[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                
                # Balance forward/carried forward formats
                (r'balance\s+b/?f[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'balance\s+c/?f[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                (r'bal\.?\s+b/?f[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'bal\.?\s+c/?f[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                
                # Short formats
                (r'^opening[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'^closing[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                
                # Just the label (value may be in adjacent text element)
                (r'opening\s+balance[:\s]*$', 'Opening Balance'),
                (r'closing\s+balance[:\s]*$', 'Closing Balance'),
                (r'opening\s+bal[:\s]*$', 'Opening Balance'),
                (r'closing\s+bal[:\s]*$', 'Closing Balance'),
                
                # Alternative formats
                (r'balance\s+at\s+beginning[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'balance\s+at\s+end[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                (r'previous\s+balance[:\s]+[\d,]+\.?\d*', 'Opening Balance'),
                (r'current\s+balance[:\s]+[\d,]+\.?\d*', 'Closing Balance'),
                
                # Indian banking formats
                (r'प्रारंभिक\s+शेष[:\s]+[\d,]+\.?\d*', 'Opening Balance'),  # Hindi
                (r'अंतिम\s+शेष[:\s]+[\d,]+\.?\d*', 'Closing Balance'),    # Hindi
                
                # Summary formats
                (r'total\s+credit[:\s]+[\d,]+\.?\d*', 'Total Credit'),
                (r'total\s+debit[:\s]+[\d,]+\.?\d*', 'Total Debit'),
            ]
            
            for pattern, field_name in balance_patterns:
                if re.search(pattern, text_lower):
                    key_matches.append({
                        'field_name': field_name,
                        'matched_text': text,
                        'keyword': 'balance pattern',
                        'bbox': bbox,
                        'data_type': 'Double',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.9,
                        'text_index': text_idx
                    })
            
            # Pattern 7: Additional statement period patterns for specific banks
            period_patterns = [
                # "Statement From : 07/08/2024 To : 28/08/2024"
                (r'statement\s+from[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "Transaction Period: From 01/04/2023 To 31/03/2024"
                (r'transaction\s+period[:\s]+from\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
                # "From 29/07/2024 To 29/08/2024"
                (r'from\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'both'),
            ]
            
            for pattern, field_type in period_patterns:
                if re.search(pattern, text_lower):
                    if field_type == 'both':
                        # Add both FROM and TO matches
                        key_matches.append({
                            'field_name': 'Statement Date From',
                            'matched_text': text,
                            'keyword': 'period pattern',
                            'bbox': bbox,
                            'data_type': 'Date',
                            'field_type': 'Header',
                            'confidence': confidence,
                            'match_score': 0.9,
                            'text_index': text_idx
                        })
                        key_matches.append({
                            'field_name': 'Statement Date To',
                            'matched_text': text,
                            'keyword': 'period pattern',
                            'bbox': bbox,
                            'data_type': 'Date',
                            'field_type': 'Header',
                            'confidence': confidence,
                            'match_score': 0.9,
                            'text_index': text_idx
                        })
    
    def _find_header_key_matches(self, text_results: List[Tuple]) -> List[Dict]:
        """
        Find potential HEADER key matches in text results.
        
        Args:
            text_results: List of (text, bbox, confidence) tuples
            
        Returns:
            List of dictionaries with key match information
        """
        key_matches = []
        
        print(f"🔍 Analyzing {len(text_results)} text elements for header keywords...")
        
        # Show more extracted text for analysis (commented out for cleaner output)
        # print("📝 Sample extracted text (first 100 elements):")
        # for i in range(min(100, len(text_results))):
        #     text, _, _ = text_results[i]
        #     print(f"   {i:2d}: '{text}'")
        
        # Look for additional smart patterns beyond CSV keywords
        self._add_smart_pattern_matches(text_results, key_matches)
        
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            cleaned_text = self._clean_text_for_matching(text)
            
            # Skip very short text that's unlikely to be a meaningful keyword
            if len(cleaned_text) < 3:
                continue
            
            # Debug: Show some sample text being analyzed (commented out for cleaner output)
            # if text_idx < 20:
            #     print(f"   Analyzing text {text_idx}: '{text}' -> '{cleaned_text}'")
            
            # Check against header field mappings only
            for field_name, keywords in self.header_mappings.items():
                for keyword_info in keywords:
                    keyword = keyword_info['key']
                    cleaned_keyword = self._clean_text_for_matching(keyword)
                    
                    # Calculate match score
                    match_score = self._calculate_keyword_match_score(cleaned_text, cleaned_keyword)
                    
                    # Lower threshold for debugging
                    if match_score > 0.5:  # Lowered from 0.7
                        key_matches.append({
                            'field_name': field_name,
                            'matched_text': text,
                            'keyword': keyword,
                            'bbox': bbox,
                            'data_type': keyword_info['data_type'],
                            'field_type': keyword_info['field_type'],
                            'confidence': confidence,
                            'match_score': match_score,
                            'text_index': text_idx
                        })
                        # print(f"   ✅ MATCH: '{text}' -> {field_name} (score: {match_score:.2f})")
                        break  # Take first good match for this text
        
        # print(f"🔍 Found {len(key_matches)} potential HEADER key matches")
        return key_matches
    
    def _calculate_keyword_match_score(self, text: str, keyword: str) -> float:
        """Calculate how well a text matches a keyword (0-1 score)."""
        if not text or not keyword:
            return 0.0
            
        # Exact match gets highest score
        if text == keyword:
            return 1.0
            
        # Check if keyword is contained in text
        if keyword in text:
            # Score based on how much of the text is the keyword
            return len(keyword) / len(text)
            
        # Check if text is contained in keyword  
        if text in keyword:
            return len(text) / len(keyword)
            
        # Check word-level overlap for multi-word keywords
        text_words = set(text.split())
        keyword_words = set(keyword.split())
        
        if text_words and keyword_words:
            intersection = text_words.intersection(keyword_words)
            union = text_words.union(keyword_words)
            return len(intersection) / len(union) if union else 0.0
        
        return 0.0
    
    def _extract_value_from_smart_match(self, key_match: Dict) -> Optional[str]:
        """
        Enhanced value extraction from smart pattern matches.
        Handles multiple bank formats and patterns for 90%+ accuracy.
        """
        text = key_match['matched_text']
        field_name = key_match['field_name']
        keyword = key_match.get('keyword', '')
        
        # print(f"      🎯 Smart extracting {field_name} from: '{text[:50]}...'")
        
        # Date extraction for various patterns
        if field_name in ['Statement Date', 'Statement Date From', 'Statement Date To']:
            if 'date pattern' in keyword:
                # Extract date from patterns like "as on date DD-MM-YYYY"
                date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if date_match:
                    extracted = date_match.group()
                    # print(f"         Extracted Date: '{extracted}'")
                    return extracted
                    
            elif 'date range pattern' in keyword or 'period pattern' in keyword:
                # Extract dates from range patterns
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if len(dates) >= 2:
                    if field_name == 'Statement Date From':
                        extracted = dates[0]
                    elif field_name == 'Statement Date To':
                        extracted = dates[-1]
                    else:  # Statement Date
                        extracted = dates[-1]  # Usually the end date
                    # print(f"         Extracted Date Range {field_name}: '{extracted}'")
                    return extracted
                elif len(dates) == 1:
                    # Single date found
                    extracted = dates[0]
                    # print(f"         Extracted Single Date: '{extracted}'")
                    return extracted
                    
            elif 'between' in keyword:
                # Legacy pattern support
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if len(dates) >= 2:
                    if field_name == 'Statement Date From':
                        extracted = dates[0]
                    else:  # Statement Date To
                        extracted = dates[1]
                    # print(f"         Extracted Between Date: '{extracted}'")
                    return extracted
        
        # Account number extraction
        elif field_name == 'Account Number':
            if 'account pattern' in keyword:
                # Enhanced account number extraction with priority order
                extraction_patterns = [
                    # Masked account numbers (highest priority) - handle both cases
                    (r'[Xx]+\d{3,8}', 0.95),
                    # Account numbers in specific contexts with capture groups
                    (r'account\s+number[:\s]+(\d{8,20})', 0.95),
                    (r'account\s+no\.?\s+(\d{8,20})', 0.95),  # Fixed for Kotak format
                    (r'a/?c\s*no\.?\s*[:\s]*(\d{8,20})', 0.9),
                    (r'a/?c[:\s]+(\d{8,20})', 0.85),
                    (r'for\s+account\s+number[:\s]+(\d{8,20})', 0.9),
                    # Detailed statement patterns (for Axis Bank) - handle both cases
                    (r'detailed\s+statement\s+for\s+a/?c\s+no\.?\s*([Xx]+\d{3,8})', 0.95),
                    (r'statement\s+for\s+a/?c\s+no\.?\s*([Xx]+\d{3,8})', 0.95),
                    (r'a/?c\s+no\s+([Xx]+\d{3,8})', 0.9),
                    # Account patterns - handle both cases  
                    (r'account\s+no\.?\s+([Xx]+\d{3,8})', 0.95),
                    # Pure numeric account numbers (lower priority)
                    (r'\d{8,20}', 0.8),
                ]
                
                for pattern, confidence in extraction_patterns:
                    # Search in lowercase but extract from original text to preserve case
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        if match.groups():
                            # Extract from capture group
                            extracted = match.group(1)
                        else:
                            # Direct match
                            extracted = match.group()
                        
                        # Convert lowercase x back to uppercase X for display
                        if 'x' in extracted.lower() and extracted.lower().startswith('x'):
                            extracted = extracted.upper()
                        
                        # Enhanced validation - reject if it contains invalid terms
                        invalid_terms = ['account', 'number', 'status', 'type', 'name', 'holder', 'product', 'service']
                        if not any(term in extracted.lower() for term in invalid_terms):
                            # Validate it's not part of a longer invalid string
                            if len(extracted.split()) <= 2:  # Account numbers shouldn't be phrases
                                # print(f"         Extracted Account Number: '{extracted}' (pattern: {pattern})")
                                return extracted
                                
            # Legacy support
            elif 'account number pattern' in keyword:
                # Extract masked account number like "XXXXXXXXXXX6582" - handle both cases
                acc_match = re.search(r'[Xx]+\d{3,8}', text, re.IGNORECASE)
                if acc_match:
                    extracted = acc_match.group().upper()  # Convert to uppercase
                    return extracted
                # Extract actual account number
                elif re.search(r'account\s+number[:\s]+(\d{8,18})', text, re.IGNORECASE):
                    acc_match = re.search(r'\d{8,18}', text)
                    if acc_match:
                        extracted = acc_match.group()
                        return extracted
                        
        # IFSC code extraction
        elif field_name == 'IFSC Code' and ('IFSC pattern' in keyword or 'IFSC Code' in text):
            # Enhanced IFSC code extraction
            patterns = [
                # IFSC with various separators
                r'ifsc\s*code\s*[:\s]*([A-Z]{4}\d{7})',
                r'ifsc[:\s]+([A-Z]{4}\d{7})',
                r'ifs\s*code[:\s]*([A-Z]{4}\d{7})',
                r'rtgs/?neft\s*ifsc[:\s]*([A-Z]{4}\d{7})',
                # Standalone IFSC codes
                r'\b([A-Z]{4}\d{7})\b',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if match.groups():
                        extracted = match.group(1).upper()
                    else:
                        extracted = match.group().upper()
                    
                    # Validate it's a proper IFSC (4 letters + 7 digits)
                    if re.match(r'^[A-Z]{4}\d{7}$', extracted):
                        # print(f"         Extracted IFSC Code: '{extracted}'")
                        return extracted
                        
        # Branch name extraction
        elif field_name == 'Bank Branch':
            if 'branch pattern' in keyword:
                patterns = [
                    # "Branch Name : Secunderabad"
                    r'branch\s+name\s*[:\s]+([A-Za-z\s\-&]{3,30})',
                    # "Branch: HYDERABAD - BANJARA HILLS"
                    r'branch[:\s]+([A-Za-z\s\-&]{3,30})',
                    # "Home Branch: [Name]"
                    r'home\s+branch[:\s]+([A-Za-z\s\-&]{3,30})',
                    # "BRN -Branch: [Name]"
                    r'brn\s*-?\s*branch[:\s]+([A-Za-z\s\-&]{3,30})',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.groups():
                        extracted = match.group(1).strip()
                        # Clean up the extracted value
                        extracted = re.sub(r'\s+', ' ', extracted)  # Normalize spaces
                        if 3 <= len(extracted) <= 50:
                            # print(f"         Extracted Branch Name: '{extracted}'")
                            return extracted
                            
            # Legacy support
            elif 'branch name pattern' in keyword:
                # Extract branch name from "Branch Name : Secunderabad"
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        extracted = parts[1].strip()
                        # print(f"         Extracted Branch Name: '{extracted}'")
                        return extracted
                        
        # Balance extraction
        elif field_name in ['Opening Balance', 'Closing Balance'] and 'balance pattern' in keyword:
            # Enhanced balance extraction with multiple strategies
            patterns = [
                # Direct amount patterns
                r'[\d,]+\.\d{2}',  # 12,345.67
                r'[\d,]+\.?\d*',   # 12,345 or 12345.67
                r'₹\s*[\d,]+\.?\d*',  # ₹12,345.67
                r'rs\.?\s*[\d,]+\.?\d*',  # Rs. 12,345.67
                # Amount at end of text
                r'[\d,]+\.?\d*\s*$',
                # Amount after colon or space
                r'[:\s]+[\d,]+\.?\d*',
            ]
            
            extracted_amount = None
            
            for pattern in patterns:
                balance_match = re.search(pattern, text)
                if balance_match:
                    candidate_amount = balance_match.group()
                    # Clean up the extracted value
                    candidate_amount = re.sub(r'[^\d.,]', '', candidate_amount)  # Remove non-numeric chars
                    candidate_amount = candidate_amount.replace(',', '')  # Remove thousands separators
                    
                    if candidate_amount and candidate_amount != '.':
                        try:
                            # Validate it's a reasonable number
                            amount = float(candidate_amount)
                            if 0.01 <= amount <= 999999999:  # Reasonable range
                                extracted_amount = candidate_amount
                                break
                        except ValueError:
                            continue
            
            if extracted_amount:
                # print(f"         Extracted Balance: '{extracted_amount}'")
                return extracted_amount
            
            # If no amount found in same text, it might be a label-only match
            # In this case, let spatial matching handle finding the nearby value
            if re.search(r'(opening|closing|balance)\s*(balance|bal)?[:\s]*$', text.lower()):
                # This is likely a label-only match, let spatial matching handle it
                return None
            
        # Fallback for any unhandled patterns
        if ('pattern' in keyword and keyword != 'IFSC pattern'):
            # Try to extract the most relevant part based on field type
            if field_name in ['Statement Date', 'Statement Date From', 'Statement Date To']:
                date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if date_match:
                    return date_match.group()
            elif field_name == 'Account Number':
                # Look for any reasonable account number pattern
                for pattern in [r'X+\d{3,8}', r'\d{8,20}']:
                    match = re.search(pattern, text)
                    if match:
                        return match.group()
            elif field_name == 'IFSC Code':
                ifsc_match = re.search(r'[A-Z]{4}\d{7}', text.upper())
                if ifsc_match:
                    return ifsc_match.group()
        
        return None
    
    def _find_value_candidates_for_key(self, key_match: Dict, text_results: List[Tuple]) -> List[Dict]:
        """
        Find potential value candidates for a given key using improved spatial analysis.
        
        Returns list of candidates with scores for ranking.
        """
        key_bbox = key_match['bbox']
        key_center_x = (key_bbox[0][0] + key_bbox[2][0]) / 2
        key_center_y = (key_bbox[0][1] + key_bbox[2][1]) / 2
        key_right = key_bbox[1][0]  # Right edge of key
        key_bottom = key_bbox[2][1]  # Bottom edge of key
        
        candidates = []
        
        for i, (text, bbox, confidence) in enumerate(text_results):
            # Skip if this is the key text itself
            if i == key_match['text_index']:
                continue
                
            # Skip very short text unless it could be valid for the data type
            if len(text.strip()) < 2 and key_match['data_type'] != 'String':
                continue
            
            val_center_x = (bbox[0][0] + bbox[2][0]) / 2
            val_center_y = (bbox[0][1] + bbox[2][1]) / 2
            val_left = bbox[0][0]
            val_top = bbox[0][1]
            
            # Method 1: Right-aligned (value to the right of key on same line)
            if (val_left >= key_right - 5 and  # Value starts near or after key ends
                abs(val_center_y - key_center_y) <= self.y_tolerance):
                
                distance = self._calculate_distance(key_bbox, bbox)
                spatial_score = max(0, 1 - distance / self.max_distance_threshold)
                
                candidates.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'method': 'right_aligned',
                        'distance': distance,
                    'spatial_score': spatial_score,
                        'index': i
                })
            
            # Method 2: Bottom-aligned (value below key, similar x position)
            elif (abs(val_center_x - key_center_x) <= self.x_tolerance and
                  val_top >= key_bottom - 5 and val_top <= key_bottom + 30):
                
                distance = self._calculate_distance(key_bbox, bbox)
                spatial_score = max(0, 1 - distance / self.max_distance_threshold)
                
                candidates.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'method': 'bottom_aligned',
                        'distance': distance,
                    'spatial_score': spatial_score,
                        'index': i
                })
        
        return candidates
    
    def _validate_and_score_value(self, value_text: str, data_type: str, field_name: str) -> Tuple[bool, float]:
        """
        Enhanced validation with stricter rules for 90%+ accuracy.
            
        Returns:
            (is_valid, confidence_score)
        """
        value_text = value_text.strip()
        
        if not value_text:
            return False, 0.0
        
        # Enhanced rejection list with more comprehensive field name detection
        wrong_values = [
            # Field names
            'account number', 'account no', 'account status', 'account type', 'account name', 'account holder',
            'ifsc code', 'ifsc', 'micr code', 'micr', 'branch code', 'branch name', 'branch email', 'branch address',
            'opening balance', 'closing balance', 'statement date', 'date', 'period', 'transaction date',
            'currency', 'inr', 'usd', 'eur', 'ckyc number', 'customer id', 'branch address',
            'transaction', 'description', 'particulars', 'narration', 'product name', 'froduct name',
            # Common wrong extractions
            'credit', 'debit', 'balance', 'amount', 'total', 'charges', 'fee', 'interest',
            'statement', 'summary', 'details', 'information', 'report', 'document',
            'page', 'continued', 'contd', 'nil', 'na', 'not applicable',
            # Common partial extractions
            'branch phone', 'phone', 'email', 'website', 'address', 'pin', 'code'
        ]
        
        if value_text.lower().replace(':', '').strip() in wrong_values:
            return False, 0.0
        
        if data_type == 'Date':
            # Enhanced date validation with stricter patterns
            date_patterns = [
                (r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', 0.95),  # Perfect DD/MM/YYYY
                (r'^\d{2,4}[/-]\d{1,2}[/-]\d{1,2}$', 0.95),  # Perfect YYYY/MM/DD
                (r'^\d{1,2}[/-]\d{1,2}[/-]\d{2}$', 0.9),   # Perfect DD/MM/YY
                (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.85),  # Date somewhere in text
                (r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}', 0.8),  # DD Month YYYY
                (r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}', 0.8),  # Month DD, YYYY
            ]
            
            # Special handling for statement date ranges
            if field_name in ['Statement Date From', 'Statement Date To']:
                # Enhanced date range patterns
                range_patterns = [
                    (r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', 0.98),
                    (r'between\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.98),
                    (r'from\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.98),
                    (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+TO\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.95),
                ]
                
                for pattern, score in range_patterns:
                    if re.search(pattern, value_text, re.IGNORECASE):
                        return True, score
            
            for pattern, score in date_patterns:
                if re.search(pattern, value_text):
                    # Strict rejection of obviously wrong dates
                    reject_words = [
                        'transaction', 'upi', 'payment', 'transfer', 'withdrawal', 'deposit', 
                        'address', 'plot', 'hno', 'street', 'city', 'pin', 'state',
                        'name', 'account', 'branch', 'bank', 'customer', 'holder'
                    ]
                    if any(word in value_text.lower() for word in reject_words):
                        continue
                    
                    # Validate year range (must be reasonable)
                    year_match = re.search(r'(\d{4})', value_text)
                    if year_match:
                        year = int(year_match.group(1))
                        if year < 2000 or year > 2030:  # Reasonable year range
                            continue
                    
                    return True, score
                    
            return False, 0.0
            
        elif data_type == 'Double':
            # Enhanced numeric validation with stricter patterns
            numeric_patterns = [
                (r'^\s*₹?\s*[\d,]+\.\d{2}\s*$', 0.95),  # Perfect currency with 2 decimals
                (r'^\s*[\d,]+\.\d{2}\s*$', 0.9),        # Perfect number with 2 decimals
                (r'^\s*₹?\s*[\d,]+\s*$', 0.85),         # Clean integer amounts
                (r'^\s*[\d,]+\.?\d*\s*$', 0.8),         # Clean numbers
                (r'₹\s*[\d,]+\.?\d*', 0.75),            # Currency with rupee symbol
            ]
            
            for pattern, score in numeric_patterns:
                if re.search(pattern, value_text):
                    # Strict rejection of transaction-like text  
                    reject_words = ['upi', 'transaction', 'transfer', 'payment', 'neft', 'rtgs', 'imps']
                    if any(word in value_text.lower() for word in reject_words):
                        continue
                    
                    # Enhanced balance validation
                    if field_name in ['Opening Balance', 'Closing Balance']:
                        try:
                            # Extract numeric value
                            numeric_part = re.sub(r'[^\d.]', '', value_text)
                            if numeric_part:
                                amount = float(numeric_part)
                                # Reject unreasonable amounts (too small or too large)
                                if amount < 0.01 or amount > 999999999:
                                    continue
                                # Give higher score for reasonable balance amounts
                                if 1 <= amount <= 10000000:  # Reasonable balance range
                                    score = min(score + 0.1, 0.98)
                        except:
                            continue
                    
                    return True, score
                    
            return False, 0.0
            
        elif data_type == 'String':
            if field_name == 'Account Number':
                # Enhanced account number validation with multiple patterns
                account_patterns = [
                    # Masked account numbers like "XXXXXXXXXXX6582"
                    (r'^X+\d{3,8}$', 0.95),
                    # Pure numeric account numbers
                    (r'^\d{8,20}$', 0.9),
                    # Alphanumeric account numbers
                    (r'^[A-Za-z0-9]{8,25}$', 0.85),
                    # Account numbers with spaces/hyphens
                    (r'^[A-Za-z0-9\s\-]{8,25}$', 0.8),
                ]
                
                for pattern, score in account_patterns:
                    if re.match(pattern, value_text.strip()):
                        # Enhanced rejection of obvious non-account-numbers
                        reject_words = [
                            'upi', 'transaction', 'transfer', 'payment', 'account', 'status', 'type', 
                            'name', 'holder', 'branch', 'bank', 'customer', 'product', 'service',
                            'statement', 'summary', 'balance', 'amount', 'currency', 'date'
                        ]
                        if any(word in value_text.lower() for word in reject_words):
                            return False, 0.0
                        
                        # Reject if it's clearly a description or address
                        if len(value_text.split()) > 2:  # Account numbers shouldn't have multiple words
                            return False, 0.0
                        
                        return True, score
                        
            elif field_name == 'IFSC Code':
                # Stricter IFSC validation - must be exactly 4 letters + 7 digits
                clean_value = value_text.upper().replace(' ', '').replace(':', '').replace('-', '')
                
                if re.match(r'^[A-Z]{4}\d{7}$', clean_value):
                    # Additional validation: first 4 letters should be bank code
                    bank_codes = [
                        'UTIB', 'ICIC', 'HDFC', 'SBIN', 'AXIS', 'PUNB', 'CNRB', 'BARB',
                        'IOBA', 'UBIN', 'ORBC', 'VIJB', 'YESB', 'KVBL', 'KKBK', 'CBIN',
                        'IDFB', 'FDRL', 'SCBL', 'HSBC', 'CIUB', 'DBSS', 'DEUT'
                    ]
                    bank_code = clean_value[:4]
                    if bank_code in bank_codes:
                        return True, 0.98  # High confidence for known bank codes
                    else:
                        return True, 0.9   # Lower confidence for unknown bank codes
                
                # Reject obvious non-IFSC codes
                reject_patterns = [r'micr', r'code', r'branch', r'email', r'address', r'number', r'statement']
                if any(re.search(pattern, value_text.lower()) for pattern in reject_patterns):
                    return False, 0.0
                    
            elif field_name == 'Bank Branch':
                # Enhanced branch name validation
                if 3 <= len(value_text) <= 50:
                    # Reject transaction-like text and field names
                    reject_words = [
                        'upi', 'transaction', 'transfer', 'payment', 'code', 'number', 'email', 
                        'address', 'currency', 'inr', 'usd', 'account', 'customer', 'id',
                        'phone', 'mobile', 'website', 'pin', 'zip'
                    ]
                    if any(word in value_text.lower() for word in reject_words):
                        return False, 0.0
                    
                    # Reject if it looks like a code (all caps + numbers)
                    if re.match(r'^[A-Z0-9\-:]+$', value_text) and len(value_text) < 10:
                        return False, 0.0
                    
                    # Give higher score for proper branch names
                    if re.search(r'[A-Za-z]{3,}', value_text):  # Has meaningful text
                        return True, 0.85
                    else:
                        return True, 0.7
                        
            # General string validation with enhanced rejection
            if len(value_text) >= 2:
                # Final comprehensive check against field names and common wrong values
                if value_text.lower().replace(':', '').strip() not in wrong_values:
                    # Additional checks for obviously wrong values
                    if not re.match(r'^[A-Z0-9\s\-:.()]+$', value_text.upper()):  # Contains only reasonable characters
                        return True, 0.6
                    elif len(value_text) > 50:  # Too long to be a reasonable header value
                        return False, 0.0
                    else:
                        return True, 0.6
                
        return False, 0.0
    
    def _find_best_value_for_key(self, key_match: Dict, text_results: List[Tuple], used_values: set) -> Optional[Dict]:
        """Find the best value for a given key using comprehensive scoring."""
        
        # print(f"🔍 Finding value for key: '{key_match['matched_text']}' ({key_match['field_name']})")
        
        # First, try to extract value directly from smart pattern matches
        if ('pattern' in key_match.get('keyword', '') or 
            'as on date' in key_match.get('keyword', '') or
            'between' in key_match.get('keyword', '') or
            'IFSC Code' in key_match.get('matched_text', '') or
            'Branch Name' in key_match.get('matched_text', '') or
            'Opening Balance' in key_match.get('matched_text', '') or
            'Closing Balance' in key_match.get('matched_text', '') or
            'Account Number' in key_match.get('matched_text', '') or
            'balance pattern' in key_match.get('keyword', '') or
            'branch pattern' in key_match.get('keyword', '') or
            'account pattern' in key_match.get('keyword', '') or
            'date pattern' in key_match.get('keyword', '') or
            'date range pattern' in key_match.get('keyword', '') or
            'period pattern' in key_match.get('keyword', '')):
            smart_value = self._extract_value_from_smart_match(key_match)
            if smart_value:
                # Validate the extracted value
                is_valid, validation_score = self._validate_and_score_value(
                    smart_value, 
                    key_match['data_type'],
                    key_match['field_name']
                )
                
                if is_valid:
                    # print(f"   ✅ SMART VALUE extracted: '{smart_value}' (validation: {validation_score:.2f})")
                    return {
                        'text': smart_value,
                        'bbox': key_match['bbox'],
                        'confidence': key_match['confidence'],
                        'method': 'smart_pattern',
                        'distance': 0.0,
                        'spatial_score': 1.0,
                        'validation_score': validation_score,
                        'final_score': validation_score * self.validation_weight + key_match['confidence'] * self.ocr_confidence_weight + 1.0 * self.spatial_weight,
                        'index': key_match['text_index']
                    }
                # else:
                #     print(f"   ❌ SMART VALUE '{smart_value}' failed validation")
        
        # For smart pattern matches that failed, don't fall back to spatial analysis
        if key_match.get('match_score', 0) > 0.8:  # High confidence smart pattern
            return None
        
        # If smart extraction failed, fall back to spatial analysis
        candidates = self._find_value_candidates_for_key(key_match, text_results)
        
        # print(f"   Found {len(candidates)} spatial candidates")
        
        if not candidates:
            return None
        
        # Score each candidate (detailed evaluation commented out for cleaner output)
        scored_candidates = []
        
        for candidate in candidates:
            if candidate['index'] in used_values:
                continue
                
            # Skip if candidate text is too similar to the key text (likely the same element)
            if self._text_similarity(candidate['text'], key_match['matched_text']) > 0.8:
                continue
                
            # print(f"   Evaluating candidate: '{candidate['text']}' (method: {candidate['method']}, distance: {candidate['distance']:.1f})")
                
            # Validate the value
            is_valid, validation_score = self._validate_and_score_value(
                candidate['text'], 
                key_match['data_type'],
                key_match['field_name']
            )
            
            # print(f"      Validation: valid={is_valid}, score={validation_score:.2f}")
            
            if not is_valid:
                continue
            
            # Calculate comprehensive score with higher weight on validation
            final_score = (
                candidate['spatial_score'] * self.spatial_weight +  # Spatial proximity (reduced weight)
                validation_score * self.validation_weight +           # Data type validation (increased weight)
                candidate['confidence'] * self.ocr_confidence_weight       # OCR confidence
            )
            
            # print(f"      Final score: {final_score:.2f} (spatial: {candidate['spatial_score']:.2f}, validation: {validation_score:.2f})")
            
            scored_candidates.append({
                **candidate,
                'validation_score': validation_score,
                'final_score': final_score
            })
        
        if not scored_candidates:
            # print(f"   ❌ No candidates passed validation")
            return None
            
        # Return the best scoring candidate
        best_candidate = max(scored_candidates, key=lambda x: x['final_score'])
        
        # Stricter minimum threshold
        min_threshold = self.min_confidence_score
        if best_candidate['final_score'] >= min_threshold:
            # print(f"   ✅ Selected best candidate: '{best_candidate['text']}' (score: {best_candidate['final_score']:.2f})")
            return best_candidate
        # else:
        #     print(f"   ❌ Best candidate score {best_candidate['final_score']:.2f} below threshold {min_threshold}")
            
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings (0-1)."""
        if not text1 or not text2:
            return 0.0
        
        text1_clean = self._clean_text_for_matching(text1)
        text2_clean = self._clean_text_for_matching(text2)
        
        if text1_clean == text2_clean:
            return 1.0
        
        # Calculate word overlap
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _post_process_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final post-processing validation to ensure 90%+ accuracy.
        Removes low-quality extractions and applies final validation rules.
        """
        # print("🔍 Post-processing headers for quality assurance...")
        
        cleaned_headers = {}
        
        for field_name, field_data in headers.items():
            value = field_data.get('value', '')
            confidence = field_data.get('confidence', 0)
            method = field_data.get('method', '')
            
            # Apply strict quality gates
            should_keep = True
            rejection_reason = None
            
            # Quality Gate 1: Minimum confidence threshold (but keep empty fields)
            if confidence < self.min_confidence_score and method != 'not_found':
                should_keep = False
                rejection_reason = f"confidence {confidence:.2f} below threshold {self.min_confidence_score}"
                
            # Quality Gate 2: Field-specific validation (skip empty fields)
            elif field_name == 'Account Number' and value.strip():
                # Account numbers must be meaningful
                if (len(value) < 6 or 
                    value.lower() in ['account', 'number', 'status', 'type', 'name', 'holder'] or
                    'account' in value.lower() or
                    len(value.split()) > 2):  # Should not be a phrase
                    should_keep = False
                    rejection_reason = "invalid account number format"
                    
            elif field_name == 'IFSC Code' and value.strip():
                # IFSC codes must be exact format
                clean_ifsc = value.upper().replace(' ', '').replace(':', '').replace('-', '')
                if not re.match(r'^[A-Z]{4}\d{7}$', clean_ifsc):
                    should_keep = False
                    rejection_reason = "invalid IFSC format"
                else:
                    # Update with cleaned version
                    field_data['value'] = clean_ifsc
                    
            elif field_name == 'Bank Branch' and value.strip():
                # Branch names should be reasonable
                if (len(value) < 3 or len(value) > 50 or
                    value.lower() in ['branch', 'code', 'phone', 'email', 'currency', 'inr'] or
                    re.match(r'^[A-Z0-9\-:]+$', value) and len(value) < 10):  # Avoid codes
                    should_keep = False
                    rejection_reason = "invalid branch name"
                    
            elif field_name in ['Statement Date', 'Statement Date From', 'Statement Date To'] and value.strip():
                # Dates should be proper format
                if not re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value):
                    should_keep = False
                    rejection_reason = "invalid date format"
                else:
                    # Validate year is reasonable
                    year_match = re.search(r'(\d{4})', value)
                    if year_match:
                        year = int(year_match.group(1))
                        if year < 2000 or year > 2030:
                            should_keep = False
                            rejection_reason = f"unreasonable year {year}"
                            
            elif field_name in ['Opening Balance', 'Closing Balance'] and value.strip():
                # Balances should be reasonable numbers
                try:
                    # Clean and validate amount
                    clean_amount = re.sub(r'[^\d.]', '', value)
                    if clean_amount:
                        amount = float(clean_amount)
                        if amount < 0.01 or amount > 999999999:
                            should_keep = False
                            rejection_reason = f"unreasonable amount {amount}"
                    else:
                        should_keep = False
                        rejection_reason = "no numeric value found"
                except ValueError:
                    should_keep = False
                    rejection_reason = "cannot parse as number"
            
            elif field_name in ['Bank Name', 'Bank Branch', 'Bank Address', 'Bank City', 'Bank State', 'Bank Phone']:
                # IFSC-derived fields should be kept as-is (high confidence from lookup)
                if method == 'ifsc_lookup' and value.strip():
                    # Handle NaN values from IFSC database
                    if pd.isna(value) or str(value).lower() == 'nan':
                        should_keep = False
                        rejection_reason = "NaN value from IFSC database"
                    elif len(str(value).strip()) >= 2:
                        should_keep = True
                    else:
                        should_keep = False
                        rejection_reason = "value too short"
                elif value.strip() and len(str(value).strip()) < 2:
                    should_keep = False
                    rejection_reason = "value too short"
            
            # Quality Gate 3: Penalize spatial matches with low confidence when smart patterns available
            if (method in ['right_aligned', 'bottom_aligned', 'left_aligned'] and 
                confidence < 0.8 and
                field_name in ['IFSC Code', 'Account Number'] and
                value.strip()):  # These should use smart patterns
                should_keep = False
                rejection_reason = "low confidence spatial match for pattern-detectable field"
            
            if should_keep:
                cleaned_headers[field_name] = field_data
                # print(f"   ✅ Kept {field_name}: '{value}' (conf: {confidence:.2f})")
            # else:
            #     print(f"   ❌ Rejected {field_name}: '{value}' - {rejection_reason}")
        
        # print(f"📊 Post-processing: {len(headers)} → {len(cleaned_headers)} headers (removed {len(headers) - len(cleaned_headers)})")
        
        return cleaned_headers
    
    def extract_headers(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract header information from a bank statement PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dictionary with extracted header information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"🏦 Extracting HEADERS ONLY from: {os.path.basename(pdf_path)}")
        
        try:
            # Get text and positions
            text_results = self._get_text_and_positions(pdf_path)
            if not text_results:
                print("❌ No text extracted")
                return {}
            
            # Find header key matches only
            key_matches = self._find_header_key_matches(text_results)
            if not key_matches:
                print("❌ No header key matches found")
                return {}
            
            # Start with standard template (all fields with empty values)
            standard_headers = self._get_standard_header_template()
            
            # Extract key-value pairs and update template
            used_values = set()
            found_fields = []
            
            # Sort by match score and field importance
            key_matches.sort(key=lambda x: (-x['match_score'], x['field_name']))
            
            for key_match in key_matches:
                field_name = key_match['field_name']
                
                # Skip if we already found this field
                if field_name in found_fields:
                    continue
                
                # Find best value for this key
                best_value = self._find_best_value_for_key(key_match, text_results, used_values)
                
                if best_value:
                    # Update the template with found values
                    standard_headers[field_name] = {
                        'value': best_value['text'],
                            'key_text': key_match['matched_text'],
                            'keyword': key_match['keyword'],
                            'data_type': key_match['data_type'],
                            'field_type': key_match['field_type'],
                        'method': best_value['method'],
                        'distance': best_value['distance'],
                        'confidence': best_value['final_score'],
                        'validation_score': best_value['validation_score'],
                        'spatial_score': best_value['spatial_score']
                    }
                    used_values.add(best_value['index'])
                    found_fields.append(field_name)
                    
                    # print(f"✅ {field_name}: '{best_value['text']}' (score: {best_value['final_score']:.2f}, method: {best_value['method']})")
                # else:
                #     print(f"⚠️ No valid value found for {field_name}")
            
            print(f"📊 Extracted initial header fields: {', '.join(found_fields)} (+ {len(standard_headers) - len(found_fields)} empty fields)")
            
            # Enhanced account number detection with IFSC-based length as primary method
            if not standard_headers['Account Number']['value'].strip():
                print("🔄 Account Number not found with keywords, trying enhanced detection...")
                
                # Primary method: IFSC-based length detection
                ifsc_account = self._find_account_number_with_ifsc_length(text_results)
                
                if ifsc_account:
                    standard_headers['Account Number'] = {
                        'value': ifsc_account['text'],
                        'key_text': ifsc_account['original_text'],
                        'keyword': f'IFSC-based length detection (Bank: {ifsc_account.get("bank_name", "Unknown")})',
                        'data_type': 'String',
                        'field_type': 'Header',
                        'method': ifsc_account['method'],
                        'distance': ifsc_account['distance'],
                        'confidence': ifsc_account['final_score'],
                        'validation_score': ifsc_account['validation_score'],
                        'spatial_score': ifsc_account['spatial_score']
                    }
                    found_fields.append('Account Number')
                    print(f"   ✅ Account Number found with IFSC length: {ifsc_account['text']}")
                else:
                    print("   ❌ IFSC-based detection failed, trying regex fallback...")
                    
                    # Second fallback: Regex-based detection
                    regex_account = self._find_account_number_with_regex(text_results)
                    
                    if regex_account:
                        standard_headers['Account Number'] = {
                            'value': regex_account['text'],
                            'key_text': regex_account['original_text'],
                            'keyword': 'regex pattern ^\d{9,18}$',
                            'data_type': 'String',
                            'field_type': 'Header',
                            'method': regex_account['method'],
                            'distance': regex_account['distance'],
                            'confidence': regex_account['final_score'],
                            'validation_score': regex_account['validation_score'],
                            'spatial_score': regex_account['spatial_score']
                        }
                        found_fields.append('Account Number')
                        print(f"   ✅ Account Number found with regex: {regex_account['text']}")
                    else:
                        print("   ❌ No account number found with regex pattern, trying label proximity...")
                        
                        # Third fallback: Look near account labels
                        label_account = self._find_account_number_near_labels(text_results)
                        
                        if label_account:
                            standard_headers['Account Number'] = {
                                'value': label_account['text'],
                                'key_text': label_account['label_text'],
                                'keyword': f'near label: {label_account["label_text"]}',
                                'data_type': 'String',
                                'field_type': 'Header',
                                'method': label_account['method'],
                                'distance': label_account['distance'],
                                'confidence': label_account['final_score'],
                                'validation_score': label_account['validation_score'],
                                'spatial_score': label_account['spatial_score']
                            }
                            found_fields.append('Account Number')
                            print(f"   ✅ Account Number found near label: {label_account['text']}")
                        else:
                            print("   ❌ No account number found with any method")
            
            # Enhance headers with IFSC lookup before post-processing
            enhanced_headers = self._enhance_headers_with_ifsc_lookup(standard_headers)
            final_headers = self._post_process_headers(enhanced_headers)
            
            # Save final results with IFSC-enhanced data
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_headers.json")
            
            with open(output_file, 'w') as f:
                json.dump(final_headers, f, indent=2)
            
            print(f"💾 Saved {len(final_headers)} final headers to: {output_file}")
            print(f"📊 Final header fields: {', '.join(final_headers.keys())}")
            
            return final_headers
            
        except Exception as e:
            print(f"❌ Error extracting headers: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
        finally:
            # Clean up temp files
            self.table_extractor.cleanup_all_temp_files()
    
    def extract_headers_batch(self, pdf_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract headers from multiple PDF files."""
        results = {}
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                file_name = os.path.basename(pdf_file)
                print(f"\n{'='*60}")
                print(f"Processing: {file_name}")
                print(f"{'='*60}")
                
                headers = self.extract_headers(pdf_file)
                results[file_name] = headers
                
                if headers:
                    print(f"✅ Success: {len(headers)} header fields extracted")
                else:
                    print(f"❌ Failed to extract headers")
            else:
                print(f"⚠️ File not found: {pdf_file}")
                results[os.path.basename(pdf_file)] = {}
        
        return results
    
    def _get_standard_header_template(self) -> Dict[str, Any]:
        """
        Get a template with all possible header fields, initialized to empty values.
        This ensures consistent output structure regardless of what's found.
        """
        return {
            # Core header fields from extraction
            'Statement Date': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'Date',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Statement Date From': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'Date',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Statement Date To': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'Date',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Opening Balance': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'Double',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Closing Balance': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'Double',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Account Number': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'IFSC Code': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Bank Branch': {
                'value': '',
                'key_text': '',
                'keyword': '',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            # IFSC-derived fields (populated via lookup)
            'Bank Name': {
                'value': '',
                'key_text': '',
                'keyword': 'IFSC lookup',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Bank Address': {
                'value': '',
                'key_text': '',
                'keyword': 'IFSC lookup',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Bank City': {
                'value': '',
                'key_text': '',
                'keyword': 'IFSC lookup',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Bank State': {
                'value': '',
                'key_text': '',
                'keyword': 'IFSC lookup',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            },
            'Bank Phone': {
                'value': '',
                'key_text': '',
                'keyword': 'IFSC lookup',
                'data_type': 'String',
                'field_type': 'Header',
                'method': 'not_found',
                'distance': 0.0,
                'confidence': 0.0,
                'validation_score': 0.0,
                'spatial_score': 0.0
            }
        }
    
    def _find_account_number_with_regex(self, text_results: List[Tuple]) -> Optional[Dict]:
        """
        Find account number using regex pattern as fallback when keyword approach fails.
        
        General Rules for Indian Bank Account Numbers:
        - Length: Usually between 9 to 18 digits
        - Digits only: Account numbers are numeric only (no letters)
        - No special characters, spaces, or letters
        - Pattern: ^\d{9,18}$
        
        Args:
            text_results: List of (text, bbox, confidence) tuples
            
        Returns:
            Dict with account number info or None if not found
        """
        print("🔍 Trying regex-based account number detection as fallback...")
        
        # Pattern for account numbers: 9-18 digits only
        account_pattern = r'^\d{9,18}$'
        
        potential_accounts = []
        
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            # Clean the text - remove common separators that might be in account numbers
            cleaned_text = re.sub(r'[\s\-\.]', '', text.strip())
            
            # Check if it matches the account number pattern
            if re.match(account_pattern, cleaned_text):
                # Use the enhanced validation function
                if self._is_valid_account_number(cleaned_text, text, text_results, text_idx, near_account_label=False):
                    # Calculate a confidence score based on length and position
                    length_score = 0.8 if 12 <= len(cleaned_text) <= 16 else 0.6  # Optimal length range
                    
                    potential_accounts.append({
                        'text': cleaned_text,
                        'original_text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'method': 'regex_pattern',
                        'distance': 0.0,
                        'spatial_score': 1.0,
                        'validation_score': length_score,
                        'final_score': length_score * 0.7 + confidence * 0.3,  # Weighted score
                        'index': text_idx,
                        'length': len(cleaned_text)
                    })
        
        if not potential_accounts:
            print("   ❌ No account numbers found with regex pattern")
            return None
        
        # Sort by confidence and length preference (12-16 digits are most common)
        potential_accounts.sort(key=lambda x: (-x['final_score'], -x['length']))
        
        best_account = potential_accounts[0]
        print(f"   ✅ Found account number with regex: '{best_account['text']}' (confidence: {best_account['final_score']:.2f})")
        
        return best_account
    
    def _find_account_number_near_labels(self, text_results: List[Tuple]) -> Optional[Dict]:
        """
        Enhanced account number detection that looks for numbers near account-related labels.
        Handles edge cases where account labels exist but numbers are not immediately adjacent.
        
        Args:
            text_results: List of (text, bbox, confidence) tuples
            
        Returns:
            Dict with account number info or None if not found
        """
        print("🔍 Trying enhanced account detection near labels...")
        
        # Account-related labels to look for
        account_labels = [
            'account number', 'account no', 'a/c no', 'a/c number', 'account', 'a/c',
            'account details', 'customer account', 'saving account', 'savings account'
        ]
        
        potential_accounts = []
        
        # Find text elements that contain account labels
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            text_lower = text.lower().strip()
            
            # Check if this text contains an account label
            for label in account_labels:
                if label in text_lower:
                    print(f"   📍 Found account label: '{text}' at index {text_idx}")
                    
                    # Look for numbers in this text and surrounding text (next 5 elements)
                    search_range = range(text_idx, min(len(text_results), text_idx + 6))
                    
                    for search_idx in search_range:
                        search_text, search_bbox, search_confidence = text_results[search_idx]
                        
                        # Check for fragmented account numbers (digits with spaces)
                        # Pattern like "54440 17044 80" - digits separated by spaces
                        if re.match(r'^[\d\s]+$', search_text.strip()) and len(search_text.strip()) > 8:
                            # Remove spaces and check if it's a valid account number length
                            cleaned_number = re.sub(r'\s+', '', search_text.strip())
                            if len(cleaned_number) >= 9 and len(cleaned_number) <= 18:
                                # Apply validation
                                if self._is_valid_account_number(cleaned_number, search_text, text_results, search_idx, near_account_label=True):
                                    distance_from_label = search_idx - text_idx
                                    confidence_score = max(0.5, 0.9 - (distance_from_label * 0.1))
                                    
                                    potential_accounts.append({
                                        'text': cleaned_number,
                                        'original_text': search_text,
                                        'label_text': text,
                                        'bbox': search_bbox,
                                        'confidence': search_confidence,
                                        'method': 'label_proximity',
                                        'distance': 0.0,
                                        'spatial_score': 1.0,
                                        'validation_score': confidence_score,
                                        'final_score': confidence_score * 0.8 + search_confidence * 0.2,
                                        'index': search_idx,
                                        'distance_from_label': distance_from_label
                                    })
                                    
                                    print(f"      ✅ Found fragmented account near label: '{cleaned_number}' from '{search_text}' (distance: {distance_from_label})")
                        
                        # Also check for solid account numbers (original logic)
                        numbers = re.findall(r'\d{8,18}', search_text)
                        
                        for number in numbers:
                            # Apply the same validation as regex approach
                            if self._is_valid_account_number(number, search_text, text_results, search_idx, near_account_label=True):
                                distance_from_label = search_idx - text_idx
                                confidence_score = max(0.5, 0.9 - (distance_from_label * 0.1))
                                
                                potential_accounts.append({
                                    'text': number,
                                    'original_text': search_text,
                                    'label_text': text,
                                    'bbox': search_bbox,
                                    'confidence': search_confidence,
                                    'method': 'label_proximity',
                                    'distance': 0.0,
                                    'spatial_score': 1.0,
                                    'validation_score': confidence_score,
                                    'final_score': confidence_score * 0.8 + search_confidence * 0.2,
                                    'index': search_idx,
                                    'distance_from_label': distance_from_label
                                })
                                
                                print(f"      ✅ Found potential account near label: '{number}' (distance: {distance_from_label})")
                    
                    break  # Found a label in this text, no need to check other labels
        
        if not potential_accounts:
            print("   ❌ No account numbers found near labels")
            return None
        
        # Sort by confidence and proximity to label
        potential_accounts.sort(key=lambda x: (-x['final_score'], x['distance_from_label']))
        
        best_account = potential_accounts[0]
        print(f"   ✅ Best account near label: '{best_account['text']}' (score: {best_account['final_score']:.2f})")
        
        return best_account
    
    def _is_valid_account_number(self, number: str, context_text: str, text_results: List[Tuple], text_idx: int, near_account_label: bool = False) -> bool:
        """
        Enhanced validation for account numbers with context awareness.
        
        Args:
            number (str): The number to validate
            context_text (str): The text containing the number
            text_results: Full list of text results for context analysis
            text_idx (int): Index of the current text in text_results
            
        Returns:
            bool: Whether the number is likely a valid account number
        """
        # Basic length check
        if len(number) < 9 or len(number) > 18:
            return False
        
        # Skip very short numbers for account detection
        if len(number) < 10:
            return False
        
        # Skip very long numbers (likely transaction IDs)
        if len(number) > 16:
            return False
        
        # Skip if it's likely a phone number (10 digits starting with 6-9, but be lenient if near account label)
        if not near_account_label and len(number) == 10 and number.startswith(('9', '8', '7', '6')):
            return False
        
        # Skip if it's likely a date
        if len(number) == 8 and (number.startswith(('19', '20')) or number[2:4] in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']):
            return False
        
        # Get surrounding context for better validation
        surrounding_context = ' '.join([
            text_results[max(0, text_idx-2)][0] if text_idx >= 2 else '',
            text_results[max(0, text_idx-1)][0] if text_idx >= 1 else '',
            context_text,
            text_results[min(len(text_results)-1, text_idx+1)][0] if text_idx < len(text_results)-1 else '',
            text_results[min(len(text_results)-1, text_idx+2)][0] if text_idx < len(text_results)-2 else ''
        ]).lower()
        
        # Skip if context suggests it's an amount (but be lenient if near account label)
        amount_indicators = ['rs', 'inr', 'rupees', 'amount', 'balance', 'credit', 'debit', 'total', '₹', 'cr', 'dr']
        if not near_account_label and any(indicator in surrounding_context for indicator in amount_indicators):
            return False
        
        # Skip if it appears in a transaction-like context (but be lenient if near account label)
        transaction_indicators = [
            'transaction', 'txn', 'ref', 'reference', 'upi', 'imps', 'neft', 'rtgs',
            'transfer', 'payment', 'withdrawal', 'deposit', '/ic', 'ltd/', 'bank/',
            'yespay', 'paytm', 'bharatpe'
        ]
        if not near_account_label and any(indicator in surrounding_context for indicator in transaction_indicators):
            return False
        
        return True
    
    def _find_account_number_with_ifsc_length(self, text_results: List[Tuple]) -> Optional[Dict]:
        """
        Enhanced account number detection using IFSC code and bank-specific account length.
        This is the primary method that uses the new A/c Length (Min) column from IFSC master.
        
        Args:
            text_results: List of (text, bbox, confidence) tuples
            
        Returns:
            Dict with account number info or None if not found
        """
        print("🔍 Trying IFSC-based account length detection (primary method)...")
        
        # Step 1: Find IFSC code in the document
        ifsc_code = None
        for text, bbox, confidence in text_results:
            # Look for IFSC patterns
            ifsc_patterns = [
                r'ifsc\s*code\s*[:\s]*([A-Z]{4}\d{7})',
                r'ifsc[:\s]+([A-Z]{4}\d{7})',
                r'ifs\s*code[:\s]*([A-Z]{4}\d{7})',
                r'rtgs/?neft\s*ifsc[:\s]*([A-Z]{4}\d{7})',
                r'\b([A-Z]{4}\d{7})\b'
            ]
            
            for pattern in ifsc_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    potential_ifsc = match.group(1).upper()
                    
                    # Validate it's a proper IFSC and not in transaction context
                    if re.match(r'^[A-Z]{4}\d{7}$', potential_ifsc):
                        # Check if it's not in transaction context
                        transaction_indicators = [
                            'neft', 'rtgs', 'imps', 'upi', 'transfer', 'payment', 'transaction',
                            'credit', 'debit', 'dr', 'cr', 'withdrawal', 'deposit'
                        ]
                        
                        text_lower = text.lower()
                        is_transaction = any(indicator in text_lower for indicator in transaction_indicators)
                        is_long_line = len(text) > 100
                        
                        if not is_transaction and not is_long_line:
                            ifsc_code = potential_ifsc
                            print(f"   ✅ Found IFSC code: {ifsc_code}")
                            break
            
            if ifsc_code:
                break
        
        if not ifsc_code:
            print("   ❌ No IFSC code found - cannot use length-based detection")
            return None
        
        # Step 2: Look up bank details and account length
        bank_details = self._lookup_bank_details_by_ifsc(ifsc_code)
        if not bank_details:
            print(f"   ❌ No bank details found for IFSC: {ifsc_code}")
            return None
        
        account_length = bank_details.get('Account Length', 0)
        if account_length <= 0:
            print(f"   ❌ No account length found for IFSC: {ifsc_code}")
            return None
        
        bank_name = bank_details.get('Bank Name', '')
        print(f"   📊 Bank: {bank_name}, Expected account length: {account_length}")
        
        # Step 3: Search for account numbers with the specific length
        potential_accounts = []
        
        # Create flexible length range (±2 digits for tolerance)
        min_length = max(8, account_length - 2)
        max_length = account_length + 2
        
        # Pattern for account numbers with the expected length
        account_pattern = rf'^\d{{{min_length},{max_length}}}$'
        
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            # Clean the text - remove common separators
            cleaned_text = re.sub(r'[\s\-\.]', '', text.strip())
            
            # Check if it matches the expected length pattern
            if re.match(account_pattern, cleaned_text):
                # Additional validation for the specific length
                if len(cleaned_text) == account_length:
                    priority_score = 0.95  # Exact length match
                elif min_length <= len(cleaned_text) <= max_length:
                    priority_score = 0.85  # Within tolerance
                else:
                    continue
                
                # Apply enhanced validation
                if self._is_valid_account_number(cleaned_text, text, text_results, text_idx, near_account_label=False):
                    potential_accounts.append({
                        'text': cleaned_text,
                        'original_text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'method': 'ifsc_length_based',
                        'ifsc_code': ifsc_code,
                        'bank_name': bank_name,
                        'expected_length': account_length,
                        'actual_length': len(cleaned_text),
                        'distance': 0.0,
                        'spatial_score': 1.0,
                        'validation_score': priority_score,
                        'final_score': priority_score * 0.8 + confidence * 0.2,
                        'index': text_idx
                    })
                    
                    print(f"      ✅ Found account with length {len(cleaned_text)}: '{cleaned_text}' (expected: {account_length})")
            
            # Also check for fragmented account numbers (like "54440 17044 80")
            if re.match(r'^[\d\s]+$', text.strip()) and len(text.strip()) > 8:
                fragmented_number = re.sub(r'\s+', '', text.strip())
                if min_length <= len(fragmented_number) <= max_length:
                    if self._is_valid_account_number(fragmented_number, text, text_results, text_idx, near_account_label=False):
                        priority_score = 0.9 if len(fragmented_number) == account_length else 0.8
                        
                        potential_accounts.append({
                            'text': fragmented_number,
                            'original_text': text,
                            'bbox': bbox,
                            'confidence': confidence,
                            'method': 'ifsc_length_based_fragmented',
                            'ifsc_code': ifsc_code,
                            'bank_name': bank_name,
                            'expected_length': account_length,
                            'actual_length': len(fragmented_number),
                            'distance': 0.0,
                            'spatial_score': 1.0,
                            'validation_score': priority_score,
                            'final_score': priority_score * 0.8 + confidence * 0.2,
                            'index': text_idx
                        })
                        
                        print(f"      ✅ Found fragmented account with length {len(fragmented_number)}: '{fragmented_number}' from '{text}' (expected: {account_length})")
        
        if not potential_accounts:
            print(f"   ❌ No account numbers found with expected length {account_length}")
            return None
        
        # Sort by exact length match first, then by confidence
        potential_accounts.sort(key=lambda x: (
            -(x['actual_length'] == x['expected_length']),  # Exact length match first
            -x['final_score']  # Then by confidence
        ))
        
        best_account = potential_accounts[0]
        print(f"   ✅ Best IFSC-based account: '{best_account['text']}' (length: {best_account['actual_length']}, expected: {best_account['expected_length']}, score: {best_account['final_score']:.2f})")
        
        return best_account


# Example usage and testing
if __name__ == "__main__":
    # Initialize the improved header extractor
    extractor = BankStatementHeaderExtractor()
    
    # Test with available files (limited to first 3 for demo)
    all_test_files = [os.path.join("BankStatements SK2", file) 
                      for file in os.listdir("BankStatements SK2") 
                      if file.endswith(".pdf")]
    test_files = all_test_files # Test first 3 files
    
    print(f"📁 Found {len(all_test_files)} PDF files, testing first {len(test_files)}:")
    for i, file in enumerate(test_files):
        print(f"   {i+1}. {os.path.basename(file)}")
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Testing improved extractor on: {os.path.basename(test_file)}")
            print("="*70)
        
            headers = extractor.extract_headers(test_file)
            
            if headers:
                print(f"\n📊 Header Extraction Summary:")
                print(f"   Total header fields extracted: {len(headers)}")
                
                for field_name, field_data in headers.items():
                    value = field_data['value']
                    score = field_data.get('confidence', 0)
                    print(f"     • {field_name}: '{value}' (confidence: {score:.2f})")
            else:
                print("❌ No headers were extracted")
        else:
            print("❌ No test files found")
            print("Available files in BankStatements SK2:")
            try:
                for file in os.listdir("BankStatements SK2"):
                    if file.endswith(".pdf"):
                        print(f"  - {file}")
            except:
                print("  Could not list directory")