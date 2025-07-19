"""
PDF to table extraction utilities
"""

import os
import json
import numpy as np
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import pandas as pd
import warnings
from img2table.document import PDF
from pathlib import Path

from .nekkanti_ocr import NekkantiOCR

warnings.filterwarnings("ignore")


class DocumentTableExtractor:
    """
    A comprehensive document table extractor that can handle:
    - Readable PDFs (using img2table directly)
    - Scanned PDFs (using NekkantiOCR for OCR + reconstruction + table extraction)
    - Image files (using NekkantiOCR for OCR + reconstruction + table extraction)
    
    The class automatically detects the document type and uses the appropriate extraction method.
    """
    def __init__(self, output_dir="comprehensive_output", save_reconstructed_pdfs=True):
        """Initialize the table extractor"""
        self.output_dir = output_dir
        self.tables_dir = os.path.join(output_dir, "tables")
        self.extracted_tables_dir = os.path.join(output_dir, "extracted_tables")
        self.temp_ocr_dir = os.path.join(self.tables_dir, "temp_ocr_processing")
        
        # Create all required directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.extracted_tables_dir, exist_ok=True)
        os.makedirs(self.temp_ocr_dir, exist_ok=True)
        
        # Option to save reconstructed PDFs
        self.save_reconstructed_pdfs = save_reconstructed_pdfs
        
        # Directory for reconstructed PDFs
        self.reconstructed_pdf_dir = os.path.join(output_dir, "reconstructed_pdfs")
        if self.save_reconstructed_pdfs:
            os.makedirs(self.reconstructed_pdf_dir, exist_ok=True)
        
        # Initialize the NekkantiOCR class with our temp directory
        self.ocr_processor = NekkantiOCR(output_dir=self.output_dir)

    def _clean_dataframe_text(self, df):
        """
        Clean all text in a DataFrame to remove Unicode artifacts and OCR noise.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        import re
        
        def clean_text(text):
            if pd.isna(text) or text is None:
                return text
            
            text = str(text)
            
            # Remove Unicode escape sequences like x005F_xFFFE_
            text = re.sub(r'x[0-9A-Fa-f]{4}_x[0-9A-Fa-f]{4}_?', '', text)
            text = re.sub(r'x[0-9A-Fa-f]{2,8}_?', '', text)
            
            # Remove other common OCR artifacts
            text = re.sub(r'_x[0-9A-Fa-f]+_?', '', text)
            text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', '', text)  # Keep printable chars + extended Unicode
            
            # Remove multiple spaces and clean up
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove very short meaningless strings that are likely artifacts
            if len(text) <= 2 and not text.isdigit() and not text.isalpha():
                return ''
            
            return text
        
        cleaned_df = df.copy()
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(clean_text)
        
        return cleaned_df

    def _is_pdf(self, file_path):
        """Check if the input file is a PDF."""
        return file_path.lower().endswith('.pdf')

    def _is_image(self, file_path):
        """Check if the input file is an image."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)

    def _is_pdf_readable(self, pdf_path):
        """
        Check if a PDF is readable (contains meaningful text) or scanned (image-only).
        Returns True if readable, False if scanned.
        """
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            meaningful_words = 0
            total_pages = len(doc)
            pages_with_text = 0
            
            # Check first few pages for text content
            pages_to_check = min(3, total_pages)
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                page_text_length = len(text)
                total_text_length += page_text_length
                
                # Count if this page has significant text
                if page_text_length > 50:  # At least 50 characters per page
                    pages_with_text += 1
                
                # Count meaningful words (more than 2 characters, contains letters)
                words = text.split()
                page_meaningful_words = 0
                for word in words:
                    if len(word) > 2 and any(c.isalpha() for c in word):
                        page_meaningful_words += 1
                meaningful_words += page_meaningful_words
            
            doc.close()
            
            # Balanced criteria for readable PDFs:
            # 1. At least 300 characters across checked pages (reduced from 500)
            # 2. At least 30 meaningful words (reduced from 50) 
            # 3. Average word length should be reasonable (2-50 chars - much more flexible for bank statements)
            # 4. At least 100 characters per page on average (reduced from 150)
            # 5. At least 60% of checked pages should have text (reduced from 80%)
            avg_word_length = total_text_length / max(meaningful_words, 1)
            avg_chars_per_page = total_text_length / max(pages_to_check, 1)
            text_page_ratio = pages_with_text / max(pages_to_check, 1)
            
            is_readable = (
                total_text_length > 300 and 
                meaningful_words > 30 and 
                2 <= avg_word_length <= 50 and  # Much more flexible for bank docs
                avg_chars_per_page > 100 and
                text_page_ratio >= 0.6
            )
            
            print(f"📊 PDF Analysis: {total_text_length} chars, {meaningful_words} words, avg_len: {avg_word_length:.1f}")
            print(f"📊 Additional checks: {avg_chars_per_page:.1f} chars/page, {text_page_ratio:.1%} pages with text")
            print(f"📊 Readability decision: {'READABLE' if is_readable else 'SCANNED'}")
            
            return is_readable
            
        except Exception as e:
            print(f"⚠️ Error checking PDF readability: {e}")
            return False

    def _post_process_table(self, df):
        """
        Enhanced post-processing to merge multiple consecutive rows that belong to the same logical row.
        Handles cases like Canara Bank where a single transaction spans multiple rows.
        """
        if df.empty or len(df) < 2:
            return df
            
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Create a list to store rows to drop
        rows_to_drop = []
        
        # Enhanced logic to handle multiple consecutive rows belonging to same logical row
        i = 1
        while i < len(df):
            # Skip if this row is already marked for deletion
            if i in rows_to_drop:
                i += 1
                continue
                
            try:
                # Count non-empty cells in the current row
                row_data = df.iloc[i].astype(str).str.strip().replace('None', pd.NA)
                row_data = row_data.replace(['nan', '', 'NaN'], pd.NA)
                non_empty = row_data.dropna()
                
                # If only one cell is non-empty or two cells are non-empty, it's a candidate for merging
                if len(non_empty) == 1 or len(non_empty) == 2:
                    # Find the target row to merge into (look backwards for the last complete row)
                    target_row_idx = self._find_target_row_for_merge(df, i, rows_to_drop)
                    
                    if target_row_idx is not None:
                        # Now collect all consecutive rows that should be merged into the target
                        rows_to_merge = []
                        j = i
                        
                        # Continue collecting rows until we find a complete row or reach the end
                        while j < len(df):
                            if j in rows_to_drop:
                                j += 1
                                continue
                                
                            curr_row_data = df.iloc[j].astype(str).str.strip().replace('None', pd.NA)
                            curr_row_data = curr_row_data.replace(['nan', '', 'NaN'], pd.NA)
                            curr_non_empty = curr_row_data.dropna()
                            
                            # If this row has 1-2 non-empty cells, it's part of the continuation
                            if len(curr_non_empty) == 1 or len(curr_non_empty) == 2:
                                rows_to_merge.append(j)
                                j += 1
                            else:
                                # This is a complete row, stop collecting
                                break
                        
                        # Merge all collected rows into the target row
                        for merge_idx in rows_to_merge:
                            self._merge_row_into_target(df, merge_idx, target_row_idx)
                            rows_to_drop.append(merge_idx)
                        
                        # Skip ahead to the next unprocessed row
                        i = j
                    else:
                        i += 1
                else:
                    # This is a complete row, move to next
                    i += 1
                    
            except Exception as e:
                print(f"⚠️ Warning: Error processing row {i} in enhanced post-processing: {e}")
                i += 1
                continue
        
        # Drop the marked rows after the loop
        if rows_to_drop:
            df = df.drop(rows_to_drop)
        
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        
        # Apply duplicate row detection and removal
        df = self._remove_duplicate_partial_rows(df)
        
        # Remove summary rows and truncate table at summary boundary
        df = self._remove_summary_rows(df)
        
        return df
    
    def _find_target_row_for_merge(self, df, current_idx, rows_to_drop):
        """
        Find the target row to merge continuation rows into.
        Look backwards for the last complete row that hasn't been marked for deletion.
        """
        for target_idx in range(current_idx - 1, -1, -1):
            if target_idx not in rows_to_drop:
                # Check if this row is a complete row (has more than 2 non-empty cells)
                row_data = df.iloc[target_idx].astype(str).str.strip().replace('None', pd.NA)
                row_data = row_data.replace(['nan', '', 'NaN'], pd.NA)
                non_empty = row_data.dropna()
                
                # If this row has 3+ non-empty cells, it's likely a complete row
                if len(non_empty) >= 3:
                    return target_idx
                # If it has 1-2 non-empty cells, continue looking backwards
                elif len(non_empty) >= 1:
                    continue
                else:
                    # Empty row, continue looking
                    continue
        
        # If no suitable target found, try to use the immediate previous row
        if current_idx > 0 and (current_idx - 1) not in rows_to_drop:
            return current_idx - 1
            
        return None
    
    def _merge_row_into_target(self, df, source_idx, target_idx):
        """
        Merge a source row into a target row by appending text to the appropriate columns.
        """
        # Get non-empty cells from source row
        source_row = df.iloc[source_idx].astype(str).str.strip().replace('None', pd.NA)
        source_row = source_row.replace(['nan', '', 'NaN'], pd.NA)
        
        for col_idx, value in enumerate(source_row):
            if pd.notna(value) and str(value).strip():
                # Get the current value in the target row
                target_val = str(df.iloc[target_idx, col_idx]) if pd.notna(df.iloc[target_idx, col_idx]) else ""
                source_val = str(value).strip()
                
                # Merge the content
                if target_val.strip():
                    df.iloc[target_idx, col_idx] = target_val + " " + source_val
                else:
                    df.iloc[target_idx, col_idx] = source_val

    def _extract_tables_with_img2table(self, pdf_path):
        """
        Extract tables from a readable PDF using img2table directly.
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Remove any "_readable" or "_reconstructed" suffix from the base name for cleaner output
        if base_name.endswith("_readable"):
            base_name = base_name[:-9]
        elif base_name.endswith("_reconstructed"):
            base_name = base_name[:-14]
        
        output_path = os.path.join(self.extracted_tables_dir, f"{base_name}.xlsx")

        try:
            print(f"📄 Using img2table directly on readable PDF: {pdf_path}")
            
            # Check if this is a Canara Bank statement
            is_canara_bank = False
            try:
                doc = fitz.open(pdf_path)
                first_page_text = doc[0].get_text().lower()
                doc.close()
                is_canara_bank = 'canara bank' in first_page_text
                if is_canara_bank:
                    print("📋 Detected Canara Bank statement - using specialized settings")
            except Exception as e:
                print(f"⚠️ Could not check bank type: {e}")
            
            # Use img2table with adjusted parameters to prevent division by zero
            pdf = PDF(pdf_path)
            
            # Special handling for Canara Bank statements
            if is_canara_bank:
                try:
                    tables = pdf.extract_tables(
                        borderless_tables=True,
                        implicit_columns=True,
                        implicit_rows=True,
                        min_confidence=40
                    )
                except Exception as e:
                    print(f"⚠️ First attempt failed for Canara Bank: {e}")
                    # Try with alternative settings
                    try:
                        tables = pdf.extract_tables(
                            borderless_tables=True,
                            implicit_columns=True,
                            implicit_rows=True,
                            min_confidence=30
                        )
                    except Exception as e2:
                        print(f"⚠️ Second attempt failed for Canara Bank: {e2}")
                        print("ℹ️ Falling back to OCR method for Canara Bank statement")
                        return []
            else:
                # Standard settings for other banks
                try:
                    tables = pdf.extract_tables(
                        borderless_tables=True,
                        implicit_columns=True,
                        implicit_rows=True,
                        min_confidence=50
                    )
                except Exception as e:
                    print(f"⚠️ First attempt failed: {e}")
                    try:
                        # Try with alternative settings
                        tables = pdf.extract_tables(
                            borderless_tables=True,
                            implicit_columns=True,
                            implicit_rows=True,
                            min_confidence=30
                        )
                    except Exception as e2:
                        print(f"⚠️ Second attempt failed: {e2}")
                        return []

            if not tables:
                print(f"ℹ️ No tables found - will try alternative method")
                return []

            total_tables_found = sum(len(page_tables) for page_tables in tables.values())
            if total_tables_found == 0:
                print(f"ℹ️ No valid tables found - will try alternative method")
                return []

            extracted_tables = []
            
            with pd.ExcelWriter(output_path) as writer:
                for page_num, page_tables in tables.items():
                    print(f"📊 Processing page {page_num}: Found {len(page_tables)} table(s)")
                    
                    for table_num, table in enumerate(page_tables):
                        try:
                            # Add error checking for table.df access
                            if not hasattr(table, 'df'):
                                print(f"⚠️ Invalid table structure on page {page_num}, table {table_num}")
                                continue
                                
                            df = table.df.copy()

                            if df.empty or df.shape[0] < 2:
                                print(f"⚠️ Skipping empty/small table on page {page_num}")
                                continue

                            # Clean Unicode artifacts and OCR noise first
                            df = self._clean_dataframe_text(df)
                            
                            # Apply post-processing to merge single-cell rows
                            df = self._post_process_table(df)

                            if df.empty:
                                print(f"⚠️ Table on page {page_num} became empty after post-processing")
                                continue

                            # Save to Excel with sheet name
                            sheet_name = f"page{page_num}_table{table_num}"[:31]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            table_info = {
                                'page_number': page_num,
                                'table_index': table_num,
                                'sheet_name': sheet_name,
                                'shape': df.shape,
                                'columns': df.columns.tolist()
                            }
                            extracted_tables.append(table_info)
                            
                            print(f"✅ Table {table_num} on page {page_num}: {df.shape[0]} rows × {df.shape[1]} columns")
                                
                        except Exception as table_error:
                            print(f"⚠️ Error processing individual table {table_num} on page {page_num}: {table_error}")
                            continue

            if extracted_tables:
                print(f"💾 All tables saved to: {output_path}")
                return extracted_tables
            else:
                print(f"ℹ️ No valid tables found - will try alternative method")
                return []
                
        except Exception as e:
            print(f"ℹ️ Could not process with img2table ({str(e)}) - will try alternative method")
            print(f"💡 This is normal for some PDFs and the system will automatically use OCR instead")
            return []

    def _extract_tables_with_ocr(self, input_path):
        """
        Extract tables using NekkantiOCR for non-readable PDFs or images.
        """
        print(f"🔄 Using NekkantiOCR for table extraction...")
        
        try:
            # Use the NekkantiOCR pipeline without auto-saving to Excel (we'll handle that)
            ocr_data, reconstructed_pdf_path, extracted_tables = self.ocr_processor.ocr_reconstruct_and_extract_tables(
                input_path, save_to_excel=False
            )
            
            if not extracted_tables:
                print(f"❌ No tables extracted from {input_path}")
                return []
            
            # Get base name for consistent file naming
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            # Remove any "_readable" or "_reconstructed" suffix from the base name for cleaner output
            if base_name.endswith("_readable"):
                base_name = base_name[:-9]
            elif base_name.endswith("_reconstructed"):
                base_name = base_name[:-14]
            elif base_name.endswith("_temp"):
                base_name = base_name[:-5]
            
            # Create Excel file in DocumentTableExtractor's output directory
            output_path = os.path.join(self.extracted_tables_dir, f"{base_name}.xlsx")
            
            # Debug: Print file path for troubleshooting
            print(f"📁 Creating Excel file at: {output_path}")
            print(f"📁 Base name used: {base_name}")
            print(f"📁 Original input: {input_path}")
            
            # Convert NekkantiOCR results to our format and save to Excel
            formatted_tables = []
            
            print(f"💾 Saving extracted tables to: {output_path}")
            
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                for table_info in extracted_tables:
                    # NekkantiOCR already applied post-processing correctly, so we use the DataFrame as-is
                    df = table_info['dataframe'].copy()
                    
                    # Clean Unicode artifacts and OCR noise
                    df = self._clean_dataframe_text(df)
                    
                    print(f"📊 Processing table from page {table_info['page_number']}: {df.shape[0]}×{df.shape[1]}")
                    
                    if df.empty:
                        print(f"⚠️ Table on page {table_info['page_number']} is empty from NekkantiOCR")
                        continue
                    
                    # Create consistent sheet name
                    sheet_name = f"page{table_info['page_number']}_table{table_info['table_index']}"[:31]
                    
                    # Save to Excel
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format table info to match our expected format
                    formatted_table = {
                        'page_number': table_info['page_number'],
                        'table_index': table_info['table_index'],
                        'sheet_name': sheet_name,
                        'shape': df.shape,
                        'columns': df.columns.tolist()
                    }
                    formatted_tables.append(formatted_table)
                    
                    print(f"✅ Table {table_info['table_index']} on page {table_info['page_number']}: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Save reconstructed PDF if option is enabled
            if self.save_reconstructed_pdfs and os.path.exists(reconstructed_pdf_path):
                permanent_pdf_path = os.path.join(self.reconstructed_pdf_dir, f"{base_name}_reconstructed.pdf")
                try:
                    import shutil
                    shutil.copy2(reconstructed_pdf_path, permanent_pdf_path)
                    print(f"💾 Reconstructed PDF saved to: {permanent_pdf_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save reconstructed PDF: {e}")
            
            print(f"✅ Successfully extracted {len(formatted_tables)} table(s) using NekkantiOCR")
            
            # Ensure file exists in expected locations for BankStatementExtractor
            backup_locations = [
                os.path.join(self.extracted_tables_dir, f"{base_name}.xlsx"),
                os.path.join(self.temp_ocr_dir, f"{base_name}.xlsx")
            ]
            
            for backup_location in backup_locations:
                backup_dir = os.path.dirname(backup_location)
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir, exist_ok=True)
                
                if os.path.exists(output_path) and not os.path.exists(backup_location):
                    try:
                        import shutil
                        shutil.copy2(output_path, backup_location)
                        print(f"📋 Backup copy created: {backup_location}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not create backup copy: {e}")
            
            return formatted_tables
            
        except Exception as e:
            print(f"❌ Error extracting tables with NekkantiOCR: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _remove_duplicate_partial_rows(self, df):
        """
        Remove duplicate partial rows where the same transaction appears multiple times
        with different levels of completeness. Keep the row with more information.
        """
        if df.empty or len(df) < 2:
            return df
            
        rows_to_remove = []
        
        # Compare consecutive rows to find partial duplicates
        for i in range(len(df) - 1):
            if i in rows_to_remove:
                continue
                
            current_row = df.iloc[i].astype(str).str.strip()
            next_row = df.iloc[i + 1].astype(str).str.strip()
            
            # Clean and normalize data for comparison
            current_clean = current_row.replace(['None', 'nan', '', 'NaN'], pd.NA)
            next_clean = next_row.replace(['None', 'nan', '', 'NaN'], pd.NA)
            
            current_non_empty = current_clean.dropna()
            next_non_empty = next_clean.dropna()
            
            # Skip if either row is mostly empty
            if len(current_non_empty) < 2 or len(next_non_empty) < 2:
                continue
            
            # Check if rows have different financial data (dates, amounts) - don't remove if they differ
            if self._have_different_financial_data(current_clean, next_clean):
                continue
            
            # Check if rows are similar (potential duplicates)
            similarity_score = self._calculate_row_similarity(current_clean, next_clean)
            
            # If similarity is very high (>85%), consider them duplicates
            # Made more conservative to avoid removing legitimate rows  
            if similarity_score > 0.85:
                # Keep the row with more complete information
                if len(current_non_empty) >= len(next_non_empty):
                    # Current row has more or equal info, remove next row
                    rows_to_remove.append(i + 1)
                    print(f"   🗑️ Removing duplicate partial row {i+1} (less complete than row {i})")
                else:
                    # Next row has more info, remove current row
                    rows_to_remove.append(i)
                    print(f"   🗑️ Removing duplicate partial row {i} (less complete than row {i+1})")
        
        # Remove identified duplicate rows
        if rows_to_remove:
            df = df.drop(rows_to_remove).reset_index(drop=True)
            print(f"   🧹 Removed {len(rows_to_remove)} duplicate partial row(s)")
        
        return df
    
    def _calculate_row_similarity(self, row1, row2):
        """
        Calculate similarity between two rows based on overlapping non-empty values.
        Returns a score between 0 and 1 where 1 means identical content.
        """
        row1_values = set()
        row2_values = set()
        
        # Collect non-empty values from both rows
        for val in row1:
            if pd.notna(val) and str(val).strip():
                # Split multi-word values to catch partial matches
                words = str(val).strip().split()
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        row1_values.add(word.lower())
        
        for val in row2:
            if pd.notna(val) and str(val).strip():
                words = str(val).strip().split()
                for word in words:
                    if len(word) > 2:
                        row2_values.add(word.lower())
        
        if not row1_values or not row2_values:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(row1_values.intersection(row2_values))
        union = len(row1_values.union(row2_values))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity if key financial patterns match (dates, amounts)
        financial_patterns = ['/', '-', '.', ',']
        row1_has_financial = any(pattern in str(val) for val in row1 for pattern in financial_patterns)
        row2_has_financial = any(pattern in str(val) for val in row2 for pattern in financial_patterns)
        
        if row1_has_financial and row2_has_financial:
            # Check for matching date or amount patterns
            for val1 in row1:
                if pd.notna(val1):
                    val1_str = str(val1).strip()
                    for val2 in row2:
                        if pd.notna(val2):
                            val2_str = str(val2).strip()
                            # If we find exact matches in financial data, boost similarity
                            if val1_str == val2_str and (len(val1_str) > 5):
                                similarity += 0.2
                                break
        
        return min(similarity, 1.0)  # Cap at 1.0
    
    def _have_different_financial_data(self, row1, row2):
        """
        Check if two rows have different financial data (dates, amounts).
        If they differ significantly in financial data, they shouldn't be considered duplicates.
        """
        # Extract financial patterns from both rows
        row1_financial = self._extract_financial_patterns(row1)
        row2_financial = self._extract_financial_patterns(row2)
        
        # If either row has no financial data, can't compare
        if not row1_financial or not row2_financial:
            return False
        
        # Check for different dates
        row1_dates = [item for item in row1_financial if self._looks_like_date(item)]
        row2_dates = [item for item in row2_financial if self._looks_like_date(item)]
        
        if row1_dates and row2_dates:
            # If dates are completely different, these are different transactions
            if not any(d1 == d2 for d1 in row1_dates for d2 in row2_dates):
                return True
        
        # Check for different amounts
        row1_amounts = [item for item in row1_financial if self._looks_like_amount(item)]
        row2_amounts = [item for item in row2_financial if self._looks_like_amount(item)]
        
        if row1_amounts and row2_amounts:
            # If amounts are completely different, these are different transactions
            if not any(a1 == a2 for a1 in row1_amounts for a2 in row2_amounts):
                return True
        
        return False
    
    def _extract_financial_patterns(self, row):
        """Extract potential financial data (dates, amounts) from a row."""
        financial_items = []
        for val in row:
            if pd.notna(val) and str(val).strip():
                val_str = str(val).strip()
                # Look for date-like or amount-like patterns
                if (self._looks_like_date(val_str) or self._looks_like_amount(val_str)):
                    financial_items.append(val_str)
        return financial_items
    
    def _looks_like_date(self, text):
        """Check if text looks like a date."""
        import re
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YYYY or DD/MM/YYYY
            r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}-[A-Z]{3}-\d{2,4}',       # DD-MMM-YYYY
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)
    
    def _looks_like_amount(self, text):
        """Check if text looks like a financial amount."""
        import re
        # Look for numbers with decimals, commas, and currency indicators
        amount_patterns = [
            r'\d+\.\d{2}',              # 123.45
            r'\d{1,3}(,\d{3})*\.\d{2}', # 1,234.56
            r'\d+,\d{3}\.\d{2}',        # 1,234.56
        ]
        return any(re.search(pattern, text) for pattern in amount_patterns)
    
    def _remove_summary_rows(self, df):
        """
        Remove summary rows and truncate table at summary boundary.
        Summary rows contain terms like 'closing balance', 'total', 'summary' etc.
        """
        if df.empty or len(df) <= 2:  # Keep tables with 2 or fewer rows
            return df
            
        summary_keywords = [
            'transaction total', 'total transaction', 'total dr/cr', 'dr/cr total',
            'closing balance', 'final balance', 'end balance', 'net balance',
            'balance forward', 'carried forward', 'brought forward',
            'grand total', 'summary', 'statement summary'
        ]
        
        # Start checking from row 2 onwards (skip potential header row)
        # Only consider rows in the latter half of the table as potential summaries
        start_index = max(2, len(df) - 5)  # Check last 5 rows or start from row 2
        
        for i in range(start_index, len(df)):
            row_text = ' '.join(str(val).lower() for val in df.iloc[i] if pd.notna(val))
            
            # Check if this row contains summary keywords
            is_summary = any(keyword in row_text for keyword in summary_keywords)
            
            # Additional check: Summary rows often have fewer unique values or repeated patterns
            row_values = [str(val).strip() for val in df.iloc[i] if pd.notna(val) and str(val).strip()]
            has_empty_cells = len(row_values) < (len(df.columns) * 0.6)  # More than 40% empty
            
            if is_summary and (has_empty_cells or len(set(row_values)) < 3):
                print(f"   🔚 Found summary row at index {i}: '{row_text[:50]}...'")
                print(f"   ✂️ Truncating table at summary boundary - removing {len(df) - i} row(s)")
                # Return everything before the summary row
                return df.iloc[:i].reset_index(drop=True)
        
        # If no summary row found, return the full table
        return df

    def process_document(self, input_path):
        """
        Main method to process any document (image, readable PDF, or scanned PDF).
        Uses img2table directly for readable PDFs, NekkantiOCR for everything else.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"🔄 Processing document: {input_path}")
        
        extracted_tables = []
        
        # Determine processing strategy based on file type
        if self._is_image(input_path):
            print("📸 Detected image file - using NekkantiOCR...")
            extracted_tables = self._extract_tables_with_ocr(input_path)
            
        elif self._is_pdf(input_path):
            if self._is_pdf_readable(input_path):
                print("📄 Detected readable PDF - using img2table directly...")
                extracted_tables = self._extract_tables_with_img2table(input_path)
                
                # If no tables found with img2table, fallback to OCR
                if not extracted_tables:
                    print("⚠️ No tables extracted with img2table - trying OCR fallback...")
                    extracted_tables = self._extract_tables_with_ocr(input_path)
                    
                # Additional fallback: if img2table failed with an error, try OCR
                # This handles cases where readability detection was wrong
            else:
                print("📄 Detected scanned PDF - using NekkantiOCR...")
                extracted_tables = self._extract_tables_with_ocr(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path}")
        
        return {
            'input_path': input_path,
            'extracted_tables': extracted_tables,
            'total_tables': len(extracted_tables)
        }

    def process_single_file(self, input_path, cleanup_temp=True):
        """
        Convenience method to process a single file and optionally clean up temporary files.
        
        Args:
            input_path (str): Path to the input file
            cleanup_temp (bool): Whether to clean up temporary files after processing
            
        Returns:
            dict: Processing result with extracted tables information
        """
        try:
            result = self.process_document(input_path)
            return result
        finally:
            # Only cleanup temp files after successful processing
            if cleanup_temp and 'result' in locals():
                self.cleanup_all_temp_files()

    def cleanup_temp_directory(self):
        """Clean up the temporary OCR directory."""
        try:
            if hasattr(self.ocr_processor, 'output_dir') and os.path.exists(self.ocr_processor.output_dir):
                import shutil
                shutil.rmtree(self.ocr_processor.output_dir)
                print(f"🧹 Cleaned up temporary directory: {self.ocr_processor.output_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temp directory: {e}")
            
    def cleanup_all_temp_files(self):
        """Clean up all temporary files and directories."""
        self.cleanup_temp_directory()
        
        # Also clean up any temp files that might be in the main output directory
        try:
            for file in os.listdir(self.output_dir):
                if file.startswith("temp_") or file.endswith("_temp.png"):
                    temp_file_path = os.path.join(self.output_dir, file)
                    os.remove(temp_file_path)
                    print(f"🧹 Removed temp file: {temp_file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up some temp files: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize the extractor with reconstructed PDF saving enabled
    extractor = DocumentTableExtractor(save_reconstructed_pdfs=True)
    
    bank_statements_folder = "BankStatements SK2"
    if not os.path.exists(bank_statements_folder):
        print(f"❌ Bank statements folder not found: {bank_statements_folder}")
        print("Please update the path to match your directory structure")
        exit(1)
        
    for file in os.listdir(bank_statements_folder):
        if file.endswith(".pdf"):
            input_path = os.path.join(bank_statements_folder, file)
            print(f"Processing file: {input_path}")
            # Test with a file
            # input_path = "BankStatements SK2/hdfc.pdf"  # This is a scanned PDF
            try:
                # Process the document
                result = extractor.process_document(input_path)
                
                print("\n" + "="*50)
                print("📊 PROCESSING SUMMARY")
                print("="*50)
                print(f"Input file: {result['input_path']}")
                print(f"Total tables extracted: {result['total_tables']}")
                
                if result['extracted_tables']:
                    print("\nExtracted tables:")
                    for i, table_info in enumerate(result['extracted_tables'], 1):
                        print(f"  {i}. Page {table_info['page_number']}, Table {table_info['table_index']}")
                        print(f"     Size: {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
                        print(f"     Sheet: {table_info['sheet_name']}")
                else:
                    print("❌ No tables were extracted")
                    
                # Show where files are saved
                print(f"\n📁 Output files:")
                print(f"   📊 Tables: {extractor.extracted_tables_dir}/")
                if extractor.save_reconstructed_pdfs:
                    print(f"   📄 Reconstructed PDFs: {extractor.reconstructed_pdf_dir}/")
                    
            except Exception as e:
                print(f"❌ Error processing document: {str(e)}")
                
            finally:
                # Clean up temporary files
                extractor.cleanup_all_temp_files()
