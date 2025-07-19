#!/usr/bin/env python3
"""
Bank statement table extraction module
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..utils.pdf_to_table import DocumentTableExtractor


class BankStatementExtractor:
    """
    Specialized extractor for bank statement transaction tables.
    
    Features:
    - Automatically identifies the main transaction table
    - Finds and concatenates continuation tables with same column structure
    - Handles headers and data consistency
    - Returns consolidated bank statement table
    """
    
    def __init__(self, output_dir="bank_statement_output", save_reconstructed_pdfs=True):
        """
        Initialize the BankStatementExtractor.
        
        Args:
            output_dir (str): Directory to save output files
            save_reconstructed_pdfs (bool): Whether to save reconstructed PDFs
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the underlying table extractor
        self.table_extractor = DocumentTableExtractor(
            output_dir=os.path.join(self.output_dir, "extracted_tables"),
            save_reconstructed_pdfs=save_reconstructed_pdfs
        )
    
    def _clean_dataframe_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all text in a DataFrame to remove Unicode artifacts and OCR noise.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        import re
        
        def clean_text(text):
            # Handle pandas Series
            if isinstance(text, pd.Series):
                return text.apply(clean_text)
                
            # Handle NaN, None, and empty values
            if pd.isna(text) or text is None or text == '':
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
            cleaned_df[col] = clean_text(cleaned_df[col])
        
        return cleaned_df
    
    def _is_transaction_table(self, df: pd.DataFrame, table_info: Dict) -> Tuple[bool, int]:
        """
        Determine if a table is likely a bank statement transaction table.
        
        Args:
            df (pd.DataFrame): The table dataframe
            table_info (dict): Table metadata
            
        Returns:
            tuple: (is_transaction_table, confidence_score)
        """
        print(f"      🔍 CONFIDENCE SCORING DEBUG for {table_info.get('sheet_name', 'unknown')}:")
        print(f"         📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if df.empty or df.shape[0] < 3 or df.shape[1] < 3:
            print(f"         ❌ TOO SMALL: empty={df.empty}, rows={df.shape[0]}, cols={df.shape[1]}")
            return False, 0
        
        confidence = 0
        
        # Check for typical bank statement columns (case-insensitive)
        transaction_keywords = [
            'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
            'reference', 'particulars', 'details', 'value', 'withdrawal', 'deposit',
            'txn', 'ref', 'chq', 'cheque', 'transfer', 'payment', 'receipt'
        ]
        
        # Convert all text to lowercase for checking
        df_text = df.astype(str).apply(lambda x: x.str.lower())
        all_text = ' '.join(df_text.values.flatten())
        
        print(f"         📝 Sample text: {all_text[:100]}...")
        
        # Count keyword matches
        found_keywords = [keyword for keyword in transaction_keywords if keyword in all_text]
        keyword_matches = len(found_keywords)
        keyword_score = keyword_matches * 10
        confidence += keyword_score
        
        print(f"         🔍 Keywords found: {found_keywords} = {keyword_score} points")
        
        # Prefer tables with more rows (transaction tables are usually longer)
        row_score = 0
        if df.shape[0] > 10:
            row_score = 20
            confidence += row_score
            print(f"         📏 Row bonus (>10 rows): +{row_score} points")
        elif df.shape[0] > 5:
            row_score = 10
            confidence += row_score
            print(f"         📏 Row bonus (>5 rows): +{row_score} points")
        else:
            print(f"         📏 No row bonus ({df.shape[0]} rows)")
        
        # Prefer tables with 4-8 columns (typical for transaction tables)
        col_score = 0
        if 4 <= df.shape[1] <= 8:
            col_score = 15
            confidence += col_score
            print(f"         📐 Column bonus (4-8 cols): +{col_score} points")
        else:
            print(f"         📐 No column bonus ({df.shape[1]} cols not in 4-8 range)")
        
        # Check for date-like patterns in first few columns
        date_score = 0
        for col_idx in range(min(3, df.shape[1])):
            col_data = df.iloc[:, col_idx].astype(str)
            date_like_values = [val for val in col_data 
                              if any(pattern in val.lower() for pattern in ['/', '-', '2024', '2023', '2022', '2021'])]
            date_like_count = len(date_like_values)
            
            print(f"         📅 Col {col_idx} date patterns: {date_like_count}/{len(col_data)} = {date_like_values[:3]}...")
            
            if date_like_count > df.shape[0] * 0.3:  # 30% of values look date-like
                date_score = 25
                confidence += date_score
                print(f"         📅 Date bonus (col {col_idx}): +{date_score} points")
                break
        
        if date_score == 0:
            print(f"         📅 No date bonus found")
        
        # Check for amount-like patterns (numbers, currency symbols)
        amount_patterns = 0
        amount_score = 0
        for col_idx in range(df.shape[1]):
            col_data = df.iloc[:, col_idx].astype(str)
            for val in col_data:
                if any(char in val for char in ['₹', '$', '€', '£', '.', ',']):
                    if any(char.isdigit() for char in val):
                        amount_patterns += 1
                        print(f"         💰 Found amount pattern in col {col_idx}: '{val}'")
                        break
        
        if amount_patterns >= 1:
            amount_score = 30
            confidence += amount_score
            print(f"         💰 Amount bonus ({amount_patterns} cols): +{amount_score} points")
        else:
            print(f"         💰 No amount patterns found")
        
        # Penalize very small tables
        penalty = 0
        if df.shape[0] < 5:
            penalty = 20
            confidence -= penalty
            print(f"         ⚠️ Small table penalty: -{penalty} points")
        
        final_decision = confidence > 50
        print(f"         🎯 FINAL SCORE: {confidence} (threshold: 50) = {'TRANSACTION TABLE' if final_decision else 'NOT TRANSACTION'}")
        print()
        
        return final_decision, confidence
    
    def _are_tables_compatible(self, main_df: pd.DataFrame, candidate_df: pd.DataFrame) -> bool:
        """
        Check if two tables can be concatenated (same structure).
        
        Args:
            main_df (pd.DataFrame): Main transaction table
            candidate_df (pd.DataFrame): Candidate table for concatenation
            
        Returns:
            bool: True if tables are compatible for concatenation
        """
        # Both should be non-empty
        if main_df.empty or candidate_df.empty:
            return False
        
        # Check if candidate table is actually transaction data (has date-like patterns)
        # Look for date patterns in any column, not just the first one
        total_date_like_count = 0
        total_cells = 0
        
        for col_idx in range(min(3, candidate_df.shape[1])):  # Check first 3 columns
            col_data = candidate_df.iloc[:, col_idx].astype(str)
            date_like_count = sum(1 for val in col_data 
                                if any(pattern in val.lower() for pattern in ['-', '/', '2024', '2023', '2022', '2021']))
            total_date_like_count += date_like_count
            total_cells += len(col_data)
        
        # If less than 15% of first few columns have date-like patterns, probably not transaction data
        date_pattern_ratio = total_date_like_count / total_cells if total_cells > 0 else 0
        if date_pattern_ratio < 0.15:
            print(f"   🚫 Candidate table rejected: insufficient date patterns ({total_date_like_count}/{total_cells} = {date_pattern_ratio:.1%})")
            return False
        
        # Allow some flexibility in column count for bank statements
        # Continuation tables often have slightly different column structures
        main_cols = main_df.shape[1]
        candidate_cols = candidate_df.shape[1]
        col_diff = abs(main_cols - candidate_cols)
        
        # Allow up to 2 column difference (common in bank statements)
        if col_diff > 2:
            print(f"   🚫 Column count too different: {main_cols} vs {candidate_cols}")
            return False
        
        # If candidate table has header row that should be skipped, analyze data rows
        if candidate_df.shape[0] > 1:
            # Use last few rows to avoid headers
            main_sample = main_df.tail(3).astype(str)
            candidate_sample = candidate_df.tail(3).astype(str)
            
            # Check compatibility based on data patterns in overlapping columns
            min_cols = min(main_cols, candidate_cols)
            compatibility_score = 0
            
            for col_idx in range(min_cols):
                main_col = main_sample.iloc[:, col_idx]
                candidate_col = candidate_sample.iloc[:, col_idx]
                
                # Check if columns have similar characteristics
                main_has_numbers = any(any(c.isdigit() for c in str(val)) for val in main_col)
                candidate_has_numbers = any(any(c.isdigit() for c in str(val)) for val in candidate_col)
                
                main_has_dates = any(any(pattern in str(val).lower() for pattern in ['-', '/']) for val in main_col)
                candidate_has_dates = any(any(pattern in str(val).lower() for pattern in ['-', '/']) for val in candidate_col)
                
                # Give points for similar patterns
                if main_has_numbers == candidate_has_numbers:
                    compatibility_score += 1
                if main_has_dates == candidate_has_dates:
                    compatibility_score += 1
            
            # Compatible if more than 50% of overlapping columns have similar patterns
            is_compatible = compatibility_score >= (min_cols * 0.5)
            
            if is_compatible:
                print(f"   ✅ Tables compatible: {main_cols} vs {candidate_cols} cols, score: {compatibility_score}/{min_cols*2}")
            else:
                print(f"   🚫 Tables incompatible: score {compatibility_score}/{min_cols*2} too low")
                
            return is_compatible
        
        return True
    
    def _remove_header_rows(self, df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove header rows from continuation tables.
        
        Args:
            df (pd.DataFrame): Table that might have header rows
            main_df (pd.DataFrame): Main table for reference
            
        Returns:
            pd.DataFrame: Table with header rows removed
        """
        if df.empty or df.shape[0] <= 1:
            return df
        
        # Convert to string for comparison
        df_str = df.astype(str)
        main_str = main_df.astype(str)
        
        # Check if first row looks like headers
        first_row = df_str.iloc[0]
        
        # If first row contains many non-numeric values while data rows have numbers, it's likely a header
        first_row_numeric_count = sum(1 for val in first_row if any(c.isdigit() for c in val))
        
        if df.shape[0] > 2:
            second_row = df_str.iloc[1]
            second_row_numeric_count = sum(1 for val in second_row if any(c.isdigit() for c in val))
            
            # If second row is much more numeric than first row, first row is likely header
            if second_row_numeric_count > first_row_numeric_count + 1:
                print(f"   🔧 Removing header row from continuation table")
                return df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def _extract_and_set_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract headers from the table and set them as column names, then remove header rows.
        
        Args:
            df (pd.DataFrame): Main transaction table
            
        Returns:
            pd.DataFrame: Table with proper column names and header rows removed
        """
        if df.empty or df.shape[0] <= 1:
            return df
        
        cleaned_df = df.copy()
        header_candidates = []
        rows_to_remove = []
        
        # Check first few rows for header patterns
        for i in range(min(3, len(df))):
            row = df.iloc[i].astype(str)
            
            # Header indicators - banking terms that suggest this is a header row
            header_indicators = [
                'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
                'reference', 'particulars', 'details', 'value', 'withdrawal', 'deposit',
                'txn', 'ref', 'chq', 'cheque', 'narration', 'opening', 'closing'
            ]
            
            row_text = ' '.join(row.values).lower()
            header_word_count = sum(1 for indicator in header_indicators if indicator in row_text)
            
            # Count numbers in the row
            number_count = sum(1 for val in row if any(c.isdigit() for c in val))
            
            # If row contains header words and few/no numbers, it's likely a header
            if header_word_count >= 2 and number_count <= 1:
                header_candidates.append((i, row.values.tolist()))
                rows_to_remove.append(i)
                print(f"   📋 Found header row {i}: {row.values[:3].tolist()}...")
            elif i == 0 and header_word_count >= 1 and number_count == 0:
                # First row with header words and no numbers is likely a header
                header_candidates.append((i, row.values.tolist()))
                rows_to_remove.append(i)
                print(f"   📋 Found header row {i}: {row.values[:3].tolist()}...")
        
        # Use the best header row to set column names
        if header_candidates:
            # Choose the header with most banking keywords
            best_header = None
            best_score = 0
            
            for row_idx, header_values in header_candidates:
                header_text = ' '.join(str(val).lower() for val in header_values)
                banking_keywords = [
                    'date', 'transaction', 'description', 'amount', 'balance', 'debit', 'credit',
                    'reference', 'particulars', 'details', 'narration', 'withdrawal', 'deposit'
                ]
                score = sum(1 for keyword in banking_keywords if keyword in header_text)
                
                if score > best_score:
                    best_score = score
                    best_header = header_values
            
            # Set column names from the best header
            if best_header:
                # Clean up header names
                clean_headers = []
                for header in best_header:
                    if pd.isna(header) or str(header).strip() in ['', 'nan', 'None']:
                        clean_headers.append(f'Column_{len(clean_headers)}')
                    else:
                        # Clean the header name
                        clean_name = str(header).strip()
                        # Remove common artifacts
                        clean_name = clean_name.replace('Account Statement', '').strip()
                        if clean_name == '' or clean_name.lower() in ['nan', 'none']:
                            clean_name = f'Column_{len(clean_headers)}'
                        clean_headers.append(clean_name)
                
                # Ensure we have the right number of columns
                while len(clean_headers) < df.shape[1]:
                    clean_headers.append(f'Column_{len(clean_headers)}')
                
                cleaned_df.columns = clean_headers[:df.shape[1]]
                print(f"   ✅ Set column names: {clean_headers[:df.shape[1]]}")
        
        # Remove identified header rows
        if rows_to_remove:
            cleaned_df = cleaned_df.drop(rows_to_remove).reset_index(drop=True)
            print(f"   ✂️ Removed {len(rows_to_remove)} header row(s)")
        
        return cleaned_df
    
    def _remove_irrelevant_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where all values are the same irrelevant text.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: DataFrame with irrelevant rows removed
        """
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        rows_to_remove = []
        
        # Irrelevant phrases that shouldn't be in transaction data
        irrelevant_phrases = [
            'account statement', 'statement', 'bank statement', 'monthly statement',
            'transaction statement', 'opening balance', 'closing balance',
            'continued', 'continued on next page', 'page', 'total', 'subtotal',
            'brought forward', 'carried forward', 'balance forward', 'summary'
        ]
        
        for i in range(len(df)):
            row = df.iloc[i].astype(str)
            
            # Get unique non-null values in the row
            unique_values = set()
            for val in row:
                if pd.notna(val) and str(val).strip() not in ['', 'nan', 'None']:
                    unique_values.add(str(val).strip().lower())
            
            # Check if all values are the same (or very similar)
            if len(unique_values) <= 1:
                if unique_values:
                    single_value = list(unique_values)[0]
                    # Check if this single repeated value is irrelevant
                    if any(phrase in single_value for phrase in irrelevant_phrases):
                        rows_to_remove.append(i)
                        print(f"   🗑️ Removing irrelevant row {i}: '{single_value}' repeated across columns")
                    elif len(single_value) < 3:  # Very short repeated text
                        rows_to_remove.append(i)
                        print(f"   🗑️ Removing short repeated text row {i}: '{single_value}'")
            
            # Also check for rows where most values are the same irrelevant text
            elif len(unique_values) == 2:
                values_list = list(unique_values)
                # If one value is much more common and irrelevant, remove the row
                for phrase in irrelevant_phrases:
                    if any(phrase in val for val in values_list):
                        rows_to_remove.append(i)
                        print(f"   🗑️ Removing mostly irrelevant row {i}: {values_list}")
                        break
        
        # Remove identified irrelevant rows
        if rows_to_remove:
            cleaned_df = cleaned_df.drop(rows_to_remove).reset_index(drop=True)
            print(f"   🧹 Removed {len(rows_to_remove)} irrelevant row(s)")
        
        return cleaned_df
    
    def extract_bank_statement_table(self, pdf_path: str, cleanup_temp: bool = True) -> Optional[pd.DataFrame]:
        """
        Extract and consolidate bank statement transaction table from PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            cleanup_temp (bool): Whether to clean up temporary files
            
        Returns:
            pd.DataFrame or None: Consolidated bank statement table
        """
        try:
            print(f"🏦 Extracting bank statement from: {os.path.basename(pdf_path)}")
            
            # Extract all tables from PDF
            result = self.table_extractor.process_document(pdf_path)
            
            if not result['extracted_tables']:
                print("❌ No tables found in the PDF")
                return None
            
            print(f"📊 Found {len(result['extracted_tables'])} table(s) - analyzing for bank statement...")
            
            # Load extracted tables and analyze them
            excel_file = None
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Remove any extension
            
            # Try to find the Excel file in the expected directory
            search_directories = [
                self.table_extractor.output_dir,
                os.path.join(self.table_extractor.output_dir, "extracted_tables"),
                self.output_dir,
                # Also search in the NekkantiOCR temporary directory for images
                os.path.join(self.table_extractor.output_dir, "temp_ocr_processing")
            ]
            
            # More flexible file matching - try different naming patterns
            possible_names = [
                base_name,
                base_name.replace("_readable", ""),
                base_name.replace("_reconstructed", ""),
                base_name.replace("_temp", ""),
                # Handle cases where the input had different extensions
                os.path.splitext(base_name)[0]
            ]
            
            for search_dir in search_directories:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if file.endswith('.xlsx'):
                            # Try exact match first
                            if any(name == file.replace('.xlsx', '') for name in possible_names):
                                excel_file = os.path.join(search_dir, file)
                                print(f"   ✅ Found Excel file (exact match): {excel_file}")
                                break
                            # Try partial match
                            elif any(name in file for name in possible_names):
                                excel_file = os.path.join(search_dir, file)
                                print(f"   ✅ Found Excel file (partial match): {excel_file}")
                                break
                    if excel_file:
                        break
            
            if not excel_file or not os.path.exists(excel_file):
                print("❌ Could not find extracted tables Excel file")
                print(f"   Looking for file containing any of: {possible_names}")
                for search_dir in search_directories:
                    if os.path.exists(search_dir):
                        print(f"   Available files in {search_dir}:")
                        try:
                            for file in os.listdir(search_dir):
                                if file.endswith('.xlsx'):
                                    print(f"     - {file}")
                        except Exception as e:
                            print(f"     Error listing files: {e}")
                return None
            
            # Read all sheets and analyze
            excel_data = pd.ExcelFile(excel_file)
            tables_analysis = []
            
            for sheet_name in excel_data.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Clean the DataFrame to remove Unicode artifacts and OCR noise
                df = self._clean_dataframe_text(df)
                
                # Find corresponding table info
                table_info = None
                for table in result['extracted_tables']:
                    if table['sheet_name'] == sheet_name:
                        table_info = table
                        break
                
                if table_info is None:
                    continue
                
                is_transaction, confidence = self._is_transaction_table(df, table_info)
                
                tables_analysis.append({
                    'sheet_name': sheet_name,
                    'dataframe': df,
                    'table_info': table_info,
                    'is_transaction': is_transaction,
                    'confidence': confidence,
                    'page_number': table_info['page_number'],
                    'table_index': table_info['table_index']
                })
                
                print(f"   📋 {sheet_name}: {df.shape[0]}×{df.shape[1]} "
                      f"({'Transaction table' if is_transaction else 'Other'}, confidence: {confidence})")
            
            # Find the main transaction table (highest confidence)
            transaction_tables = [t for t in tables_analysis if t['is_transaction']]
            
            if not transaction_tables:
                print("❌ No transaction tables identified")
                return None
            
            # Sort by confidence (highest first)
            transaction_tables.sort(key=lambda x: x['confidence'], reverse=True)
            main_table = transaction_tables[0]
            
            print(f"✅ Main transaction table: {main_table['sheet_name']} "
                  f"({main_table['dataframe'].shape[0]}×{main_table['dataframe'].shape[1]})")
            
            # Start with the main table
            consolidated_df = main_table['dataframe'].copy()
            concatenated_tables = [main_table['sheet_name']]
            
            # Clean headers from the main table
            print(f"🧹 Cleaning headers from main table...")
            consolidated_df = self._extract_and_set_headers(consolidated_df)
            
            # Remove irrelevant rows after header extraction
            print(f"🧹 Removing irrelevant rows...")
            consolidated_df = self._remove_irrelevant_rows(consolidated_df)
            
            # Look for continuation tables
            potential_continuations = []
            for table in tables_analysis:
                if (table['sheet_name'] != main_table['sheet_name'] and 
                    self._are_tables_compatible(main_table['dataframe'], table['dataframe'])):
                    potential_continuations.append(table)
            
            # Sort continuation tables by page and table order
            potential_continuations.sort(key=lambda x: (x['page_number'], x['table_index']))
            
            # Concatenate compatible tables
            for continuation in potential_continuations:
                cont_df = continuation['dataframe'].copy()
                
                # Remove potential header rows
                cont_df = self._remove_header_rows(cont_df, main_table['dataframe'])
                
                # Remove irrelevant rows from continuation table
                cont_df = self._remove_irrelevant_rows(cont_df)
                
                if not cont_df.empty:
                    # Handle column count differences
                    main_cols = len(consolidated_df.columns)
                    cont_cols = cont_df.shape[1]
                    
                    if cont_cols < main_cols:
                        # Add missing columns to continuation table
                        missing_cols = main_cols - cont_cols
                        for i in range(missing_cols):
                            cont_df[f'temp_col_{i}'] = pd.NA
                        print(f"   🔧 Added {missing_cols} missing columns to continuation table")
                    elif cont_cols > main_cols:
                        # Remove extra columns from continuation table
                        cont_df = cont_df.iloc[:, :main_cols]
                        print(f"   🔧 Removed {cont_cols - main_cols} extra columns from continuation table")
                    
                    # Ensure continuation table has same column names as main table
                    cont_df.columns = consolidated_df.columns
                    
                    concatenated_df = pd.concat([consolidated_df, cont_df], ignore_index=True)
                    consolidated_df = concatenated_df
                    concatenated_tables.append(continuation['sheet_name'])
                    print(f"   ➕ Added continuation table: {continuation['sheet_name']} "
                          f"({cont_df.shape[0]} rows)")
                else:
                    print(f"   ⚠️ Skipped empty continuation table: {continuation['sheet_name']}")
            
            # Final cleanup: remove any remaining problematic rows
            if not consolidated_df.empty:
                # Clean any remaining Unicode artifacts from the final result
                consolidated_df = self._clean_dataframe_text(consolidated_df)
                
                # Remove rows where all values are NaN or empty
                consolidated_df = consolidated_df.dropna(how='all')
                
                # Remove rows where all values are the same irrelevant text
                consolidated_df = self._remove_irrelevant_rows(consolidated_df)
                
                # Reset index
                consolidated_df = consolidated_df.reset_index(drop=True)
            
            print(f"🎉 Consolidated bank statement: {consolidated_df.shape[0]}×{consolidated_df.shape[1]}")
            print(f"   📝 Combined tables: {', '.join(concatenated_tables)}")
            print(f"   🏷️ Column names: {list(consolidated_df.columns)}")
            
            # Save consolidated table
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_bank_statement.xlsx")
            
            consolidated_df.to_excel(output_file, index=False)
            print(f"💾 Saved consolidated bank statement to: {output_file}")
            
            return consolidated_df
            
        except Exception as e:
            print(f"❌ Error extracting bank statement: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Only cleanup temp files after successful processing
            if cleanup_temp and 'consolidated_df' in locals():
                self.table_extractor.cleanup_all_temp_files()
    
    def process_multiple_files(self, pdf_files: List[str], cleanup_temp: bool = True) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Process multiple PDF files and extract bank statements from each.
        
        Args:
            pdf_files (list): List of PDF file paths
            cleanup_temp (bool): Whether to clean up temporary files after each file
            
        Returns:
            dict: Dictionary mapping file names to extracted DataFrames
        """
        results = {}
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                file_name = os.path.basename(pdf_file)
                print(f"\n{'='*60}")
                print(f"Processing: {file_name}")
                print(f"{'='*60}")
                
                bank_statement = self.extract_bank_statement_table(pdf_file, cleanup_temp)
                results[file_name] = bank_statement
                
                if bank_statement is not None:
                    print(f"✅ Success: {bank_statement.shape[0]} transactions extracted")
                else:
                    print(f"❌ Failed to extract bank statement")
            else:
                print(f"⚠️ File not found: {pdf_file}")
                results[os.path.basename(pdf_file)] = None
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the bank statement extractor
    extractor = BankStatementExtractor(save_reconstructed_pdfs=True)
    
    # Test with a single file
    # test_file = "BankStatements SK2/axis_bank__statement_for_september_2024_unlocked.pdf"
    # test_file = "BankStatements SK2/hdfc.pdf"
    for file in os.listdir("BankStatements SK2"):
        if file.endswith(".pdf"):
            test_file = os.path.join("BankStatements SK2", file)
            print(f"Processing: {file}")
            if os.path.exists(test_file):
                print("🧪 Testing Bank Statement Extractor")
                print("="*50)
                
                bank_statement = extractor.extract_bank_statement_table(test_file)
                
                if bank_statement is not None:
                    print(f"\n📊 Final Bank Statement Summary:")
                    print(f"   Shape: {bank_statement.shape[0]} rows × {bank_statement.shape[1]} columns")
                    print(f"   Columns: {list(bank_statement.columns)}")
                    
                    final_bs = "BankStatements_Results"
                    os.makedirs(final_bs, exist_ok=True)
                    bank_statement.to_csv(os.path.basename(test_file).replace(".pdf", ".csv"), index=False)
                    # Show first few rows
                    print(f"\n📋 First 5 rows:")
                    print(bank_statement.head().to_string())
                    
                    # Show data types
                    print(f"\n🔍 Column info:")
                    for i, col in enumerate(bank_statement.columns):
                        sample_values = bank_statement[col].dropna().head(3).tolist()
                        print(f"   {i}: {col} - {sample_values}")
                
            else:
                print(f"❌ Test file not found: {test_file}")
                print("Available files in BankStatements SK2:")
                try:
                    for file in os.listdir("BankStatements SK2"):
                        if file.endswith(".pdf"):
                            print(f"  - {file}")
                except:
                    print("  Could not list directory") 