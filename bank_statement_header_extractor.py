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
    
    def __init__(self, output_dir="header_extraction_output", keywords_csv="bankstmt_allkeys.csv"):
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
        
        # Improved configuration for matching
        self.y_tolerance = 25  # Increased from 15 for debugging
        self.x_tolerance = 15   # Increased from 8 for debugging
        self.max_distance_threshold = 150  # Increased from 100 for debugging
        self.min_confidence_score = 0.5  # Lowered from 0.7 for debugging
        
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
        Add smart pattern-based matches beyond CSV keywords.
        This helps find headers in different bank statement formats.
        """
        print("🔍 Looking for smart patterns...")
        
        for text_idx, (text, bbox, confidence) in enumerate(text_results):
            
            # Pattern 1: Statement date in text like "as on date DD-MM-YYYY"
            if re.search(r'as on date\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text.lower()):
                # Extract the date part
                date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                if date_match:
                    key_matches.append({
                        'field_name': 'Statement Date',
                        'matched_text': text,
                        'keyword': 'as on date',
                        'bbox': bbox,
                        'data_type': 'Date',
                        'field_type': 'Header',
                        'confidence': confidence,
                        'match_score': 0.9,
                        'text_index': text_idx
                    })
                    print(f"   ✅ SMART MATCH: Statement Date from '{text[:50]}...'")
            
            # Pattern 2: Date ranges in text like "between DD-MM-YYYY to DD-MM-YYYY"
            elif re.search(r'between\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text.lower()):
                # Add match for FROM date
                key_matches.append({
                    'field_name': 'Statement Date From',
                    'matched_text': text,
                    'keyword': 'between',
                    'bbox': bbox,
                    'data_type': 'Date',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.9,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: Statement Date From from '{text[:50]}...'")
                
                # Add match for TO date
                key_matches.append({
                    'field_name': 'Statement Date To',
                    'matched_text': text,
                    'keyword': 'between',
                    'bbox': bbox,
                    'data_type': 'Date',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.9,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: Statement Date To from '{text[:50]}...'")
            
            # Pattern 3: Account numbers with masking like "XXXXXXXXXXX6582"
            elif re.search(r'^.*a/c.*no.*X+\d{3,6}', text, re.IGNORECASE):
                key_matches.append({
                    'field_name': 'Account Number',
                    'matched_text': text,
                    'keyword': 'account number pattern',
                    'bbox': bbox,
                    'data_type': 'String',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.85,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: Account Number from '{text[:50]}...'")
            
            # Pattern 3b: Account numbers in format "Account No. XXXXXXXXXXX6582"
            elif re.search(r'account\s+no\..*X+\d{3,6}', text, re.IGNORECASE):
                key_matches.append({
                    'field_name': 'Account Number',
                    'matched_text': text,
                    'keyword': 'account number pattern',
                    'bbox': bbox,
                    'data_type': 'String',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.85,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: Account Number from '{text[:50]}...'")
            
            # Pattern 4: IFSC codes in format ABCD1234567
            elif re.search(r'\b[A-Z]{4}\d{7}\b', text):
                key_matches.append({
                    'field_name': 'IFSC Code',
                    'matched_text': text,
                    'keyword': 'IFSC pattern',
                    'bbox': bbox,
                    'data_type': 'String',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.9,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: IFSC Code from '{text[:50]}...'")
            
            # Pattern 5: Branch names after "Branch" keyword
            elif re.search(r'branch.*name.*:', text.lower()) and len(text) < 50:
                key_matches.append({
                    'field_name': 'Bank Branch',
                    'matched_text': text,
                    'keyword': 'branch name pattern',
                    'bbox': bbox,
                    'data_type': 'String',
                    'field_type': 'Header',
                    'confidence': confidence,
                    'match_score': 0.8,
                    'text_index': text_idx
                })
                print(f"   ✅ SMART MATCH: Branch Name from '{text[:50]}...'")
    
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
        
        print(f"🔍 Found {len(key_matches)} potential HEADER key matches")
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
        Extract the actual value from smart pattern matches where value is embedded in the text.
        """
        text = key_match['matched_text']
        field_name = key_match['field_name']
        keyword = key_match.get('keyword', '')
        
        print(f"      🔧 Smart extraction for {field_name} from: '{text}'")
        
        if field_name == 'Statement Date' and 'as on date' in keyword:
            # Extract date from "Relationship summary as on date 30-09-2024"
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
            if date_match:
                extracted = date_match.group()
                print(f"         Extracted Statement Date: '{extracted}'")
                return extracted
                
        elif field_name == 'Statement Date From' and 'between' in keyword:
            # Extract FROM date from "between 01-09-2024 to 30-09-2024"
            from_match = re.search(r'between\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text.lower())
            if from_match:
                extracted = from_match.group(1)
                print(f"         Extracted Statement Date From: '{extracted}'")
                return extracted
            # If full range, return it (will be parsed later)
            return text
            
        elif field_name == 'Statement Date To' and 'between' in keyword:
            # Extract TO date from "between 01-09-2024 to 30-09-2024"
            to_match = re.search(r'to\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text.lower())
            if to_match:
                extracted = to_match.group(1)
                print(f"         Extracted Statement Date To: '{extracted}'")
                return extracted
                
        elif field_name == 'Account Number' and 'account number pattern' in keyword:
            # Extract masked account number like "XXXXXXXXXXX6582"
            acc_match = re.search(r'X+\d{3,6}', text)
            if acc_match:
                extracted = acc_match.group()
                print(f"         Extracted Account Number: '{extracted}'")
                return extracted
                
        elif field_name == 'IFSC Code' and ('IFSC pattern' in keyword or 'IFSC Code' in text):
            # Extract IFSC code like "UTIB0000068" from "IFSC Code : UTIB0000068"
            ifsc_match = re.search(r'\b[A-Z]{4}\d{7}\b', text)
            if ifsc_match:
                extracted = ifsc_match.group()
                print(f"         Extracted IFSC Code: '{extracted}'")
                return extracted
                
        elif field_name == 'Bank Branch' and ('branch name pattern' in keyword or 'Branch Name' in text):
            # Extract branch name from "Branch Name : Secunderabad"
            if ':' in text:
                # Split by colon and take the part after it
                parts = text.split(':', 1)
                if len(parts) > 1:
                    extracted = parts[1].strip()
                    print(f"         Extracted Branch Name: '{extracted}'")
                    return extracted
                    
        print(f"         ❌ Could not extract value from smart match")
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
        Validate if the value matches expected data type and return confidence score.
        
        Returns:
            (is_valid, confidence_score)
        """
        value_text = value_text.strip()
        
        if not value_text:
            return False, 0.0
        
        if data_type == 'Date':
            # Improved date validation patterns
            date_patterns = [
                (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.9),  # DD/MM/YYYY
                (r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}', 0.9),  # YYYY/MM/DD
                (r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}', 0.8),  # DD Month YYYY
                (r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}', 0.8),  # Month DD, YYYY
                (r'\d{1,2}[/-]\d{1,2}[/-]\d{2}', 0.7)  # DD/MM/YY
            ]
            
            # Special handling for statement date ranges
            if field_name in ['Statement Date From', 'Statement Date To']:
                # Look for date range patterns like "01-09-2024 to 30-09-2024"
                range_patterns = [
                    (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.95),
                    (r'between\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.95),
                    (r'from\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+to\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 0.95),
                ]
                
                for pattern, score in range_patterns:
                    if re.search(pattern, value_text.lower()):
                        return True, score
            
            for pattern, score in date_patterns:
                if re.search(pattern, value_text):
                    # Additional validation: reject obviously wrong dates
                    reject_words = ['transaction', 'upi', 'payment', 'transfer', 'withdrawal', 'deposit']
                    if any(word in value_text.lower() for word in reject_words):
                        continue
                    return True, score
                    
            return False, 0.0
            
        elif data_type == 'Double':
            # Improved numeric validation  
            numeric_patterns = [
                (r'^\s*₹?\s*[\d,]+\.?\d*\s*$', 0.9),  # Clean currency amounts
                (r'^\s*[\d,]+\.?\d*\s*$', 0.8),  # Clean numbers
                (r'₹\s*[\d,]+\.?\d*', 0.7),  # Currency with rupee symbol
                (r'\$\s*[\d,]+\.?\d*', 0.7),  # Currency with dollar symbol
            ]
            
            for pattern, score in numeric_patterns:
                if re.search(pattern, value_text):
                    # Reject transaction-like text  
                    reject_words = ['upi', 'transaction', 'transfer', 'payment']
                    if any(word in value_text.lower() for word in reject_words):
                        continue
                    return True, score
                    
            return False, 0.0
            
        elif data_type == 'String':
            # For account numbers, branches, IFSC codes
            if field_name == 'Account Number':
                # Account numbers: should be alphanumeric, reasonable length
                if re.match(r'^[A-Za-z0-9\-\s]{8,20}$', value_text):
                    # Reject transaction-like text
                    reject_words = ['upi', 'transaction', 'transfer', 'payment']
                    if any(word in value_text.lower() for word in reject_words):
                        return False, 0.0
                    return True, 0.9
                elif re.match(r'^\d{8,18}$', value_text):  # Pure numeric account number
                    return True, 0.8
                # Look for masked account numbers like "XXXXXXXXXXX6582"
                elif re.match(r'^X+\d{3,6}$', value_text):
                    return True, 0.85
                    
            elif field_name == 'IFSC Code':
                # IFSC codes: 4 letters + 7 digits
                if re.match(r'^[A-Z]{4}\d{7}$', value_text.upper().replace(' ', '')):
                    return True, 0.95
                elif re.match(r'^[A-Z]{4}[0-9A-Z]{7}$', value_text.upper().replace(' ', '')):
                    return True, 0.8
                    
            elif field_name == 'Bank Branch':
                # Branch names: should be reasonable text
                if len(value_text) >= 3 and len(value_text) <= 50:
                    # Reject transaction-like text
                    reject_words = ['upi', 'transaction', 'transfer', 'payment']
                    if any(word in value_text.lower() for word in reject_words):
                        return False, 0.0
                    return True, 0.8
                    
            # General string validation
            if len(value_text) >= 2:
                return True, 0.6
                
        return False, 0.0
    
    def _find_best_value_for_key(self, key_match: Dict, text_results: List[Tuple], used_values: set) -> Optional[Dict]:
        """Find the best value for a given key using comprehensive scoring."""
        
        print(f"🔍 Finding value for key: '{key_match['matched_text']}' ({key_match['field_name']})")
        
        # First, try to extract value directly from smart pattern matches
        if ('pattern' in key_match.get('keyword', '') or 
            'as on date' in key_match.get('keyword', '') or
            'between' in key_match.get('keyword', '') or
            'IFSC Code' in key_match.get('matched_text', '') or
            'Branch Name' in key_match.get('matched_text', '')):
            smart_value = self._extract_value_from_smart_match(key_match)
            if smart_value:
                # Validate the extracted value
                is_valid, validation_score = self._validate_and_score_value(
                    smart_value, 
                    key_match['data_type'],
                    key_match['field_name']
                )
                
                if is_valid:
                    print(f"   ✅ SMART VALUE extracted: '{smart_value}' (validation: {validation_score:.2f})")
                    return {
                        'text': smart_value,
                        'bbox': key_match['bbox'],
                        'confidence': key_match['confidence'],
                        'method': 'smart_pattern',
                        'distance': 0.0,
                        'spatial_score': 1.0,
                        'validation_score': validation_score,
                        'final_score': validation_score * 0.8 + key_match['confidence'] * 0.2,
                        'index': key_match['text_index']
                    }
                else:
                    print(f"   ❌ SMART VALUE '{smart_value}' failed validation")
        
        # If smart extraction failed, fall back to spatial analysis
        candidates = self._find_value_candidates_for_key(key_match, text_results)
        
        print(f"   Found {len(candidates)} spatial candidates")
        
        if not candidates:
            return None
        
        # Score each candidate (detailed evaluation commented out for cleaner output)
        scored_candidates = []
        
        for candidate in candidates:
            if candidate['index'] in used_values:
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
            
            # Calculate comprehensive score
            final_score = (
                candidate['spatial_score'] * 0.4 +  # Spatial proximity
                validation_score * 0.4 +           # Data type validation
                candidate['confidence'] * 0.2       # OCR confidence
            )
            
            # print(f"      Final score: {final_score:.2f} (spatial: {candidate['spatial_score']:.2f}, validation: {validation_score:.2f})")
            
            scored_candidates.append({
                **candidate,
                'validation_score': validation_score,
                'final_score': final_score
            })
        
        if not scored_candidates:
            print(f"   ❌ No candidates passed validation")
            return None
            
        # Return the best scoring candidate
        best_candidate = max(scored_candidates, key=lambda x: x['final_score'])
        
        # Lower minimum threshold for debugging
        min_threshold = 0.5  # Lowered from 0.7
        if best_candidate['final_score'] >= min_threshold:
            print(f"   ✅ Selected best candidate: '{best_candidate['text']}' (score: {best_candidate['final_score']:.2f})")
            return best_candidate
        else:
            print(f"   ❌ Best candidate score {best_candidate['final_score']:.2f} below threshold {min_threshold}")
            
        return None
    
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
            
            # Extract key-value pairs
            extracted_headers = {}
            used_values = set()
            
            # Sort by match score and field importance
            key_matches.sort(key=lambda x: (-x['match_score'], x['field_name']))
            
            for key_match in key_matches:
                field_name = key_match['field_name']
                
                # Skip if we already found this field
                if field_name in extracted_headers:
                    continue
                
                # Find best value for this key
                best_value = self._find_best_value_for_key(key_match, text_results, used_values)
                
                if best_value:
                    extracted_headers[field_name] = {
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
                    
                    print(f"✅ {field_name}: '{best_value['text']}' "
                          f"(score: {best_value['final_score']:.2f}, method: {best_value['method']})")
                else:
                    print(f"⚠️ No valid value found for {field_name}")
            
            # Save results
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_headers.json")
            
            with open(output_file, 'w') as f:
                json.dump(extracted_headers, f, indent=2)
            
            print(f"💾 Saved {len(extracted_headers)} headers to: {output_file}")
            print(f"📊 Extracted header fields: {', '.join(extracted_headers.keys())}")
            
            return extracted_headers
            
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


# Example usage and testing
if __name__ == "__main__":
    # Initialize the improved header extractor
    extractor = BankStatementHeaderExtractor()
    
    # Test with available files (limited to first 3 for demo)
    all_test_files = [os.path.join("BankStatements SK2", file) 
                      for file in os.listdir("BankStatements SK2") 
                      if file.endswith(".pdf")]
    test_files = all_test_files[:3]  # Test first 3 files
    
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
                
            break  # Test just one file for now
    else:
        print("❌ No test files found")
        print("Available files in BankStatements SK2:")
        try:
            for file in os.listdir("BankStatements SK2"):
                if file.endswith(".pdf"):
                    print(f"  - {file}")
        except:
            print("  Could not list directory") 