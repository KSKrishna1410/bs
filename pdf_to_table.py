import os
import json
import numpy as np
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from paddleocr import PaddleOCR
import pymupdf
import pandas as pd
import warnings
from img2table.document import PDF
import fitz  # PyMuPDF

# Import the NekkantiOCR class for OCR and reconstruction
from nekkanti_ocr_table_reconstruction import NekkantiOCR

warnings.filterwarnings("ignore")


class DocumentTableExtractor:
    """
    A comprehensive document table extractor that can handle:
    - Readable PDFs (using img2table directly)
    - Scanned PDFs (using NekkantiOCR for OCR + reconstruction + table extraction)
    - Image files (using NekkantiOCR for OCR + reconstruction + table extraction)
    
    The class automatically detects the document type and uses the appropriate extraction method.
    """
    def __init__(self, output_dir="bs_tables_extracted", save_reconstructed_pdfs=True):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Option to save reconstructed PDFs
        self.save_reconstructed_pdfs = save_reconstructed_pdfs
        
        # Directory for reconstructed PDFs
        self.reconstructed_pdf_dir = "bs_reconstructed_pdfs"
        if self.save_reconstructed_pdfs:
            os.makedirs(self.reconstructed_pdf_dir, exist_ok=True)
        
        # Initialize the NekkantiOCR class for OCR and reconstruction
        # Use a temporary directory that we can easily clean up
        temp_ocr_dir = os.path.join(self.output_dir, "temp_ocr_processing")
        os.makedirs(temp_ocr_dir, exist_ok=True)  # Ensure it exists upfront
        self.ocr_processor = NekkantiOCR(output_dir=temp_ocr_dir)

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
        Post-process the DataFrame to merge single-cell rows with the previous row.
        Based on the user's provided logic from the notebook.
        """
        # Create a list to store rows to drop
        rows_to_drop = []
        
        # Iterate over rows (starting from index 1)
        for i in range(1, len(df)):
            # Check if the row index exists before accessing
            if i < len(df):
                # Count non-empty cells in the current row
                row_data = df.iloc[i].astype(str).str.strip().replace('None', pd.NA)
                non_empty = row_data.dropna()
                
                # If only one cell is non-empty or two cells are non-empty
                if len(non_empty) == 1 or len(non_empty) == 2:
                    col_idx = non_empty.index[0]  # Get the column where the data is
                    # Merge the content into the above row
                    df.at[i-1, col_idx] = str(df.at[i-1, col_idx]) + " " + str(df.at[i, col_idx])
                    # Mark the current row for deletion
                    rows_to_drop.append(i)
        
        # Drop the marked rows after the loop
        df = df.drop(rows_to_drop)
        
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        
        return df

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
        
        output_path = os.path.join(self.output_dir, f"{base_name}.xlsx")

        try:
            print(f"📄 Using img2table directly on readable PDF: {pdf_path}")
            
            # Use img2table with the parameters specified in the user's example
            pdf = PDF(pdf_path)
            tables = pdf.extract_tables(
                borderless_tables=True,
                implicit_columns=True,
                implicit_rows=True,
                min_confidence=50
            )

            if not tables:
                print(f"❌ No tables found in the PDF with img2table")
                return []

            total_tables_found = sum(len(page_tables) for page_tables in tables.values())
            if total_tables_found == 0:
                print(f"❌ No valid tables found in any page with img2table")
                return []

            extracted_tables = []
            
            with pd.ExcelWriter(output_path) as writer:
                for page_num, page_tables in tables.items():
                    print(f"📊 Processing page {page_num}: Found {len(page_tables)} table(s)")
                    
                    for table_num, table in enumerate(page_tables):
                        try:
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
                print(f"❌ No valid tables found to save")
                return []
                
        except Exception as e:
            print(f"❌ Error extracting tables from PDF with img2table: {str(e)}")
            print(f"💡 This PDF might need OCR processing instead")
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
            
            # Create Excel file in DocumentTableExtractor's output directory
            output_path = os.path.join(self.output_dir, f"{base_name}.xlsx")
            
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
            return formatted_tables
            
        except Exception as e:
            print(f"❌ Error extracting tables with NekkantiOCR: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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
            if cleanup_temp:
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
                print(f"   📊 Tables: {extractor.output_dir}/")
                if extractor.save_reconstructed_pdfs:
                    print(f"   📄 Reconstructed PDFs: {extractor.reconstructed_pdf_dir}/")
                    
            except Exception as e:
                print(f"❌ Error processing document: {str(e)}")
                
            finally:
                # Clean up temporary files
                extractor.cleanup_all_temp_files()
