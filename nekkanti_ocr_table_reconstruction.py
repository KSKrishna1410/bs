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
from img2table.document import Image as Img2TableImage, PDF as Img2TablePDF

warnings.filterwarnings("ignore")


class NekkantiOCR:
    def __init__(self, output_dir="ocr_outputs_reconstructed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu"
        )

    def detect_table_lines(self, image_path):
        """Detect table lines in the image using morphological operations."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 10)
        
        # Create kernels for horizontal and vertical line detection
        h_kernel_len = image.shape[1] // 40
        v_kernel_len = image.shape[0] // 40
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        
        # Extract horizontal lines
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        
        # Extract vertical lines
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines to create table mask
        table_mask = cv2.bitwise_or(horizontal, vertical)
        table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        return horizontal, vertical, table_mask

    def draw_table_lines_in_pdf(self, c, horizontal, vertical, table_mask, img_height):
        """Draw detected table lines in the PDF."""
        if horizontal is None or vertical is None or table_mask is None:
            return
            
        # Calculate scaling factors
        scale_x = c._pagesize[0] / self.original_width
        scale_y = c._pagesize[1] / self.original_height
        
        # Vertical offset to move lines up (adjust this value as needed)
        vertical_offset = 20  # pixels
        
        # Set line properties
        c.setStrokeColorRGB(0.5, 0.5, 0.5)  # Gray color for table lines
        c.setLineWidth(0.5)  # Line width for table lines
        
        # Draw horizontal lines
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20:  # Only draw lines that are reasonably long
                # Scale coordinates
                x1_scaled = x * scale_x
                x2_scaled = (x + w) * scale_x
                y_scaled = (y - vertical_offset) * scale_y  # Subtract offset to move up
                
                # Invert y coordinate for PDF
                y_pdf = img_height - y_scaled
                
                # Draw horizontal line
                c.line(x1_scaled, y_pdf, x2_scaled, y_pdf)
        
        # Draw vertical lines
        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20:  # Only draw lines that are reasonably long
                # Scale coordinates with horizontal offset for vertical lines
                horizontal_offset_vertical = 4  # Shift vertical lines 4 pixels to the right
                x_scaled = (x + horizontal_offset_vertical) * scale_x
                # Additional offset for vertical lines to move them up a bit more
                vertical_line_offset = vertical_offset + 3 # Extra 3 pixels up for vertical lines (reduced by 2 to shift down)
                y1_scaled = (y - vertical_line_offset) * scale_y  # Subtract offset to move up
                y2_scaled = (y + h - vertical_line_offset) * scale_y  # Subtract offset to move up
                
                # Invert y coordinates for PDF
                y1_pdf = img_height - y1_scaled
                y2_pdf = img_height - y2_scaled
                
                # Draw vertical line
                c.line(x_scaled, y1_pdf, x_scaled, y2_pdf)

    def convert_ndarray(self, obj):
        """Recursively convert NumPy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.convert_ndarray(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        return obj

    def _is_pdf(self, file_path):
        """Check if the input file is a PDF."""
        return file_path.lower().endswith('.pdf')
    
    def _pdf_to_images(self, pdf_path):
        """Convert PDF pages to temporary image files."""
        import fitz  # Make sure we have the right import
        
        # Ensure the output directory exists with proper permissions
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Using OCR output directory: {self.output_dir}")
        
        try:
            doc = fitz.open(pdf_path)
            temp_image_paths = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            print(f"📄 Converting PDF to images: {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image (with high DPI for better OCR)
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
                    
                    # Create temporary image path
                    temp_image_path = os.path.join(self.output_dir, f"{base_name}_temp_page{page_num + 1}.png")
                    
                    # Save the image
                    pix.save(temp_image_path)
                    
                    # Verify the file was created
                    if os.path.exists(temp_image_path):
                        temp_image_paths.append(temp_image_path)
                        print(f"✅ Created temp image: {os.path.basename(temp_image_path)}")
                    else:
                        print(f"❌ Failed to create temp image: {temp_image_path}")
                        
                except Exception as page_error:
                    print(f"❌ Error converting page {page_num + 1}: {page_error}")
                    continue
                    
            doc.close()
            
            if not temp_image_paths:
                raise RuntimeError(f"No temporary images were created from PDF: {pdf_path}")
                
            print(f"✅ Successfully created {len(temp_image_paths)} temporary images")
            return temp_image_paths
            
        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            raise

    def _cleanup_temp_files(self, temp_files):
        """Remove temporary files."""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️  Warning: Could not remove temp file {temp_file}: {e}")

    def ocr_and_reconstruct(self, input_path):
        temp_image_paths = []
        
        try:
            print(f"🔄 Starting OCR and reconstruction for: {os.path.basename(input_path)}")
            
            # Check if input is PDF or image
            if self._is_pdf(input_path):
                print(f"📄 Converting PDF to images for OCR processing...")
                temp_image_paths = self._pdf_to_images(input_path)
                
                if not temp_image_paths:
                    raise ValueError(f"Could not convert PDF pages to images: {input_path}")
                
                # Use first page for getting dimensions (assuming all pages have similar dimensions)
                primary_image_path = temp_image_paths[0]
            else:
                # It's an image file
                primary_image_path = input_path
            
            # Verify primary image exists and is readable
            if not os.path.exists(primary_image_path):
                raise FileNotFoundError(f"Primary image not found: {primary_image_path}")
            
            # Get original image dimensions
            img = cv2.imread(primary_image_path)
            if img is None:
                raise ValueError(f"Could not read image at {primary_image_path}")
            self.original_height = img.shape[0]
            self.original_width = img.shape[1]
            print(f"📐 Image dimensions: {self.original_width}x{self.original_height}")

            # Run OCR on the input (PaddleOCR can handle both images and PDFs)
            print(f"🔍 Running OCR on input...")
            result = self.ocr.predict(input_path)
            ocr_data = [self.convert_ndarray(dict(res)) for res in result]
            print(f"📝 OCR completed: {len(ocr_data)} pages processed")

            # Create reconstructed PDF
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            pdf_path = os.path.join(self.output_dir, f"{base_name}_reconstructed.pdf")
            
            print(f"📄 Creating reconstructed PDF: {os.path.basename(pdf_path)}")

            c = None

            for i, page_result in enumerate(ocr_data):
                rec_texts = page_result["rec_texts"]
                rec_polys = page_result["rec_polys"]

                if not rec_polys:
                    print(f"⚠️ No text found on page {i+1}")
                    continue

                # Use original image dimensions for PDF size
                img_width, img_height = self.original_width, self.original_height

                if c is None:
                    c = canvas.Canvas(pdf_path, pagesize=(img_width, img_height))
                else:
                    c.showPage()
                    c.setPageSize((img_width, img_height))

                # Detect and draw lines (use appropriate image for line detection)
                try:
                    if self._is_pdf(input_path) and i < len(temp_image_paths):
                        horizontal, vertical, table_mask = self.detect_table_lines(temp_image_paths[i])
                    else:
                        horizontal, vertical, table_mask = self.detect_table_lines(input_path)
                        
                    self.draw_table_lines_in_pdf(c, horizontal, vertical, table_mask, img_height)
                except Exception as line_error:
                    print(f"⚠️ Warning: Could not detect lines for page {i+1}: {line_error}")

                def invert_y(y):
                    return img_height - y

                # Add text to PDF
                text_count = 0
                for text, poly in zip(rec_texts, rec_polys):
                    try:
                        x = min(p[0] for p in poly)
                        y = min(p[1] for p in poly)
                        y_pdf = invert_y(y)
                        
                        # Add horizontal offset (adjust this value as needed)
                        horizontal_offset = 10  # pixels
                        x = x + horizontal_offset
                        
                        # Calculate the bounding box dimensions
                        height = max(abs(p[1] - poly[0][1]) for p in poly)
                        width = max(abs(p[0] - poly[0][0]) for p in poly)
                        
                        # Calculate average character width (assuming average character is about 60% of height)
                        avg_char_width = height * 0.6
                        
                        # Estimate number of characters that should fit in the width
                        num_chars = max(1, width / avg_char_width)
                        
                        # Calculate font size based on height with a small margin
                        height_based_size = int(height * 0.85)  # 85% of height to leave some margin
                        
                        # Calculate font size based on width and number of characters
                        width_based_size = int(width / num_chars * 0.85)  # 85% of width per character
                        
                        # Use the minimum of both sizes to ensure text fits
                        font_size = max(min(height_based_size, width_based_size), 6)
                        
                        # Set font and draw text
                        c.setFont("Helvetica", font_size)
                        c.drawString(x, y_pdf, text)
                        text_count += 1
                        
                    except Exception as text_error:
                        print(f"⚠️ Warning: Could not add text '{text}' to page {i+1}: {text_error}")
                        continue
                
                print(f"📝 Added {text_count} text elements to page {i+1}")

            if c:
                c.save()
                print(f"✅ PDF reconstruction completed: {os.path.basename(pdf_path)}")
                
                # Verify the PDF was created
                if not os.path.exists(pdf_path):
                    raise RuntimeError(f"Reconstructed PDF was not created: {pdf_path}")
                    
            else:
                raise RuntimeError("No pages were processed - could not create PDF")

            return ocr_data, pdf_path
            
        except Exception as e:
            print(f"❌ Error in OCR and reconstruction: {e}")
            raise
            
        finally:
            # Clean up temporary files
            if temp_image_paths:
                print(f"🧹 Cleaning up {len(temp_image_paths)} temporary image files...")
                self._cleanup_temp_files(temp_image_paths)

    def ocr(self, input_path):
        result = self.ocr.predict(input_path)
        return [self.convert_ndarray(dict(res)) for res in result]

    def extract_tables_from_pdf(self, pdf_path=None, save_to_excel=True):
        """
        Extract tables from a PDF using img2table.
        
        Args:
            pdf_path (str, optional): Path to the PDF file. If None, will try to use the last reconstructed PDF.
            save_to_excel (bool): Whether to save tables to Excel files.
            
        Returns:
            list: List of dictionaries containing table data and metadata.
        """
        if pdf_path is None:
            raise ValueError("PDF path must be provided")
            
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"📄 Processing PDF with img2table: {pdf_path}")
        
        try:
            # Use img2table to extract tables
            pdf = Img2TablePDF(pdf_path)
            tables = pdf.extract_tables(
                borderless_tables=True,
                implicit_columns=True,
                implicit_rows=True,
                min_confidence=50
            )
            
            print(f"📊 Found {len(tables)} page(s) with tables")
            
            all_tables = []
            
            # Only create Excel writer if saving is requested
            excel_path = None
            if save_to_excel:
                excel_filename = f"{base_name}_reconstructed_tables.xlsx"
                excel_path = os.path.join(self.output_dir, excel_filename)
            
            # Process tables from each page
            for page_num in range(len(tables)):
                page_tables = tables[page_num]
                
                if page_tables and len(page_tables) > 0:
                    print(f"📊 Found {len(page_tables)} table(s) on page {page_num + 1}")
                    
                    for table_idx in range(len(page_tables)):
                        try:
                            # Get DataFrame from img2table
                            table = page_tables[table_idx]
                            df = table.df.copy()
                            
                            if df.empty:
                                print(f"⚠️  Table {table_idx + 1} on page {page_num + 1} is empty")
                                continue
                            
                            # Post-process the DataFrame to merge single-cell rows
                            df = self._post_process_table(df)
                            
                            # Create table metadata
                            table_data = {
                                'page_number': page_num + 1,
                                'table_index': table_idx + 1,
                                'dataframe': df,
                                'shape': df.shape,
                                'columns': df.columns.tolist(),
                            }
                            
                            all_tables.append(table_data)
                            print(f"✅ Table {table_idx + 1} extracted and processed: {df.shape[0]} rows × {df.shape[1]} columns")
                            
                        except Exception as e:
                            print(f"❌ Error processing table {table_idx + 1} on page {page_num + 1}: {str(e)}")
                            continue
                else:
                    print(f"❌ No tables found on page {page_num + 1}")
            
            # Save all tables to a single Excel file with multiple sheets (only if requested)
            if all_tables and save_to_excel:
                try:
                    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                        for table_data in all_tables:
                            sheet_name = f"Page{table_data['page_number']}_Table{table_data['table_index']}"
                            # Excel sheet names cannot exceed 31 characters
                            if len(sheet_name) > 31:
                                sheet_name = f"P{table_data['page_number']}_T{table_data['table_index']}"
                            
                            table_data['dataframe'].to_excel(writer, sheet_name=sheet_name, index=False, header=True)
                            table_data['excel_path'] = excel_path
                            table_data['sheet_name'] = sheet_name
                            
                    print(f"💾 All tables saved to: {excel_path}")
                    print(f"📊 Total sheets created: {len(all_tables)}")
                    
                except Exception as e:
                    print(f"❌ Error saving Excel file: {str(e)}")
                    # Fallback to individual files if Excel writer fails
                    for table_data in all_tables:
                        try:
                            individual_filename = f"{base_name}_page{table_data['page_number']}_table{table_data['table_index']}.xlsx"
                            individual_path = os.path.join(self.output_dir, individual_filename)
                            table_data['dataframe'].to_excel(individual_path, index=False, header=True)
                            table_data['excel_path'] = individual_path
                            print(f"💾 Table saved individually to: {individual_path}")
                        except Exception as fallback_e:
                            print(f"❌ Error saving individual table: {str(fallback_e)}")
            
            if all_tables:
                print(f"🎉 Successfully extracted {len(all_tables)} table(s) from PDF")
            else:
                print("❌ No tables were extracted from the PDF")
                
            return all_tables
            
        except Exception as e:
            print(f"❌ Error extracting tables with img2table: {str(e)}")
            return []

    def extract_tables_dataframes_only(self, pdf_path):
        """
        Extract tables from PDF and return only DataFrames without saving to Excel.
        This method is useful when the caller wants to handle Excel saving themselves.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries containing table data and metadata (including DataFrames)
        """
        return self.extract_tables_from_pdf(pdf_path, save_to_excel=False)
    
    def _clean_extracted_text(self, text):
        """
        Clean extracted text by removing Unicode artifacts and OCR noise.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text is None:
            return text
        
        text = str(text)
        
        # Remove Unicode escape sequences like x005F_xFFFE_
        import re
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

    def _post_process_table(self, df):
        """
        Post-process the DataFrame to merge single-cell rows with the previous row.
        Based on the user's provided logic.
        """
        if df.empty or len(df) < 2:
            return df
            
        # First, clean all text in the DataFrame to remove Unicode artifacts
        for col in df.columns:
            df[col] = df[col].apply(self._clean_extracted_text)
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Create a list to store rows to drop
        rows_to_drop = []
        
        # Iterate over rows (starting from index 1)
        for i in range(1, len(df)):
            # Check if the row index exists before accessing
            if i < len(df) and i not in rows_to_drop:
                try:
                # Count non-empty cells in the current row
                    row_data = df.iloc[i].astype(str).str.strip()
                    # Replace 'None', 'nan', empty strings with pd.NA
                    row_data = row_data.replace(['None', 'nan', '', 'NaN'], pd.NA)
                    non_empty = row_data.dropna()
                
                    # If only one cell is non-empty or two cells are non-empty
                    if len(non_empty) == 1 or len(non_empty) == 2:
                        col_idx = non_empty.index[0]  # Get the column where the data is
                        
                        # Only merge if the previous row exists and is not marked for deletion
                        if (i-1) >= 0 and (i-1) not in rows_to_drop:
                    # Merge the content into the above row
                            prev_val = str(df.at[i-1, col_idx]) if pd.notna(df.at[i-1, col_idx]) else ""
                            curr_val = str(df.at[i, col_idx]) if pd.notna(df.at[i, col_idx]) else ""
                            
                            # Only merge if current value is not empty
                            if curr_val.strip():
                                if prev_val.strip():
                                    df.at[i-1, col_idx] = prev_val + " " + curr_val
                                else:
                                    df.at[i-1, col_idx] = curr_val
                                
                    # Mark the current row for deletion
                        rows_to_drop.append(i)
                
                except Exception as e:
                    print(f"⚠️ Warning: Error processing row {i} in post-processing: {e}")
                    continue
        
        # Drop the marked rows after the loop
        if rows_to_drop:
            df = df.drop(rows_to_drop)
        
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        
        return df

    def ocr_reconstruct_and_extract_tables(self, input_path, save_to_excel=True):
        """
        Complete pipeline: OCR the input, reconstruct PDF with lines, and extract tables.
        
        Args:
            input_path (str): Path to input image or PDF
            save_to_excel (bool): Whether to save extracted tables to Excel files in NekkantiOCR's output directory
            
        Returns:
            tuple: (ocr_data, pdf_path, extracted_tables)
        """
        print("🔄 Starting OCR and reconstruction...")
        
        # Step 1: OCR and reconstruct PDF
        ocr_data, pdf_path = self.ocr_and_reconstruct(input_path)
        
        print("🔍 Extracting tables from reconstructed PDF...")
        
        # Step 2: Extract tables from the reconstructed PDF
        extracted_tables = self.extract_tables_from_pdf(pdf_path, save_to_excel)
        
        return ocr_data, pdf_path, extracted_tables


if __name__ == "__main__":
    # input_path = "35 Invoices/3.1_JioMart-Invoice-1739802766186.pdf"  # or .png, .jpg
    # input_path = "35 Invoices/1.1_13.05.2024  Digital Track-1.png"  # or .png, .jpg
    # input_path = "35 Invoices/28.1_Sree Cammphor Works-1.png"  # or .png, .jpg
    # input_path = "35 Invoices/18.1_Dunzo Daily.pdf"
    # input_path = "35 Invoices/2.1_08.07.2024 Hathway-1.png"
    # input_path = "/Users/saikrishnakompelly/Desktop/glbyte_ocr/35 Invoices/23.1_medplus-1.png"
    # input_path = "35 Invoices/8.6_13.06.2024 Bajaj Electronics (1).pdf"
    input_path = "/Users/saikrishnakompelly/Desktop/glbyte_ocr/BankStatements SK2/hdfc.pdf"
    # input_path = "/Users/saikrishnakompelly/Desktop/glbyte_ocr/BankStatements SK2/Canarabank.pdf"
    # input_path = "Financial-statements/Insipirian Yearly Results.pdf"
    # input_path = "WhatsApp Image 2025-07-05 at 12.10.54.jpeg"
    # input_path = "test_wrapper/21.1_itemInvoiceDownload-1 (1)-1.png"
    # input_path = "35 Invoices/8.6_13.06.2024 Bajaj Electronics (1).pdf"
    # input_path = "35 Invoices/23.1_medplus-1.png"
    # input_path = "35 Invoices/21.1_itemInvoiceDownload-1 (1)-1.png"

    processor = NekkantiOCR()
    
    # Use the complete pipeline: OCR, reconstruct PDF, and extract tables
    ocr_results, pdf_path, extracted_tables = processor.ocr_reconstruct_and_extract_tables(input_path)
    # ocr_results, pdf_path = processor.ocr_and_reconstruct(input_path)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    json_path = os.path.join(processor.output_dir, f"{base_name}_res.json")
    with open(json_path, "w") as f:
        json.dump(ocr_results, f, indent=2)

    print(f"✅ JSON saved to: {json_path}")
    print(f"✅ Reconstructed PDF saved to: {pdf_path}")
    
    # Display information about extracted tables
    if extracted_tables:
        print(f"\n📊 Table Extraction Summary:")
        print(f"   Total tables extracted: {len(extracted_tables)}")
        for i, table_info in enumerate(extracted_tables):
            print(f"   Table {i+1}:")
            print(f"     - Page: {table_info['page_number']}")
            print(f"     - Size: {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
            if 'excel_path' in table_info:
                print(f"     - Excel file: {table_info['excel_path']}")
    else:
        print("\n❌ No tables were extracted from the document")
        
    # Example: You can also extract tables from any existing PDF
    # existing_pdf_path = "path/to/your/existing.pdf"
    # tables = processor.extract_tables_from_pdf(existing_pdf_path)