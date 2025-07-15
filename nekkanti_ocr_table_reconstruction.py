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
from img2table.document import Image as Img2TableImage, PDF as Img2TablePDF
import traceback
import sys
import cv2
import numpy as np
from img2table.tables.processing.bordered_tables.cells.identification import identify_cells
from img2table.tables.processing.bordered_tables.cells.dataframe import get_cells_dataframe

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
        import traceback
        import sys
        import cv2
        import numpy as np
        from img2table.tables.processing.bordered_tables.cells.identification import identify_cells
        from img2table.tables.processing.bordered_tables.cells.dataframe import get_cells_dataframe
        
        if pdf_path is None:
            raise ValueError("PDF path must be provided")
            
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"📄 Processing PDF with img2table: {pdf_path}")
        
        try:
            # Use img2table to extract tables
            print("🔍 Creating PDF object...")
            pdf = Img2TablePDF(pdf_path)
            
            print("🔍 Attempting table extraction with the following parameters:")
            print("   - borderless_tables: True")
            print("   - implicit_columns: True")
            print("   - implicit_rows: True")
            print("   - min_confidence: 50")
            
            try:
                # Patch the identify_cells function to handle division by zero
                original_identify_cells = identify_cells
                
                def safe_identify_cells(*args, **kwargs):
                    try:
                        return original_identify_cells(*args, **kwargs)
                    except ZeroDivisionError:
                        print("⚠️ Division by zero in cell identification - using fallback method")
                        # Get the arrays from kwargs
                        h_lines_arr = kwargs.get('h_lines_arr', args[0] if args else None)
                        v_lines_arr = kwargs.get('v_lines_arr', args[1] if len(args) > 1 else None)
                        
                        if h_lines_arr is None or v_lines_arr is None:
                            raise ValueError("Missing line arrays")
                        
                        # Create a simple grid-based cell array as fallback
                        height, width = h_lines_arr.shape
                        cells = np.zeros((height, width), dtype=int)
                        cell_id = 1
                        
                        # Identify cells based on line intersections
                        for i in range(height-1):
                            for j in range(width-1):
                                if h_lines_arr[i,j] and v_lines_arr[i,j]:
                                    cells[i:i+2, j:j+2] = cell_id
                                    cell_id += 1
                        
                        return cells
                
                # Replace the original function with our safe version
                identify_cells = safe_identify_cells
                
                tables = pdf.extract_tables(
                    borderless_tables=True,
                    implicit_columns=True,
                    implicit_rows=True,
                    min_confidence=50
                )
                
                # Restore the original function
                identify_cells = original_identify_cells
                
            except Exception as table_error:
                print(f"❌ Error extracting tables with img2table: {str(table_error)}")
                print("\n📋 Detailed error traceback:")
                traceback.print_exc()
                print("\n🔍 Stack trace analysis:")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_list = traceback.extract_tb(exc_traceback)
                for filename, line, func, text in tb_list:
                    print(f"  File: {filename}")
                    print(f"  Line: {line}")
                    print(f"  Function: {func}")
                    print(f"  Code: {text}\n")
                
                # Try alternative method with more conservative parameters
                try:
                    print("🔄 Attempting alternative extraction method...")
                    tables = pdf.extract_tables(
                        borderless_tables=True,
                        implicit_columns=True,
                        implicit_rows=True,
                        min_confidence=30,
                        min_row_height=10,
                        cell_margin=1.0,
                        line_scale=30,
                        threshold_blocksize=45,
                        threshold_constant=15
                    )
                except Exception as alt_error:
                    print(f"❌ Alternative method also failed: {str(alt_error)}")
                    return []
        
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
                            print(f"   🔍 Processing table {table_idx + 1} on page {page_num + 1}...")
                            # Get DataFrame from img2table
                            table = page_tables[table_idx]
                            
                            try:
                                df = table.df.copy()
                            except Exception as df_error:
                                print(f"❌ Error accessing table DataFrame: {str(df_error)}")
                                print("\n📋 Detailed error traceback:")
                                traceback.print_exc()
                                print("\n🔍 Stack trace analysis:")
                                exc_type, exc_value, exc_traceback = sys.exc_info()
                                tb_list = traceback.extract_tb(exc_traceback)
                                for filename, line, func, text in tb_list:
                                    print(f"  File: {filename}")
                                    print(f"  Line: {line}")
                                    print(f"  Function: {func}")
                                    print(f"  Code: {text}\n")
                                continue
                            
                            if df.empty:
                                print(f"⚠️  Table {table_idx + 1} on page {page_num + 1} is empty")
                                continue
                            
                            try:
                                print("   🔄 Applying post-processing...")
                                # Post-process the DataFrame to merge single-cell rows
                                df = self._post_process_table(df)
                            except Exception as post_error:
                                print(f"❌ Error in post-processing: {str(post_error)}")
                                print("\n📋 Detailed error traceback:")
                                traceback.print_exc()
                                print("\n🔍 Stack trace analysis:")
                                exc_type, exc_value, exc_traceback = sys.exc_info()
                                tb_list = traceback.extract_tb(exc_traceback)
                                for filename, line, func, text in tb_list:
                                    print(f"  File: {filename}")
                                    print(f"  Line: {line}")
                                    print(f"  Function: {func}")
                                    print(f"  Code: {text}\n")
                                continue
                            
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
                            print("\n📋 Detailed error traceback:")
                            traceback.print_exc()
                            print("\n🔍 Stack trace analysis:")
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            tb_list = traceback.extract_tb(exc_traceback)
                            for filename, line, func, text in tb_list:
                                print(f"  File: {filename}")
                                print(f"  Line: {line}")
                                print(f"  Function: {func}")
                                print(f"  Code: {text}\n")
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
                    print("\n📋 Detailed error traceback:")
                    traceback.print_exc()
                    print("\n🔍 Stack trace analysis:")
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb_list = traceback.extract_tb(exc_traceback)
                    for filename, line, func, text in tb_list:
                        print(f"  File: {filename}")
                        print(f"  Line: {line}")
                        print(f"  Function: {func}")
                        print(f"  Code: {text}\n")
                    
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
                            print("\n📋 Detailed error traceback:")
                            traceback.print_exc()
        
            if all_tables:
                print(f"🎉 Successfully extracted {len(all_tables)} table(s) from PDF")
            else:
                print("❌ No tables were extracted from the PDF")
                
            return all_tables
            
        except Exception as e:
            print(f"❌ Error in table extraction: {str(e)}")
            print("\n📋 Detailed error traceback:")
            traceback.print_exc()
            print("\n🔍 Stack trace analysis:")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_list = traceback.extract_tb(exc_traceback)
            for filename, line, func, text in tb_list:
                print(f"  File: {filename}")
                print(f"  Line: {line}")
                print(f"  Function: {func}")
                print(f"  Code: {text}\n")
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
        Enhanced post-processing to merge multiple consecutive rows that belong to the same logical row.
        Handles cases like Canara Bank where a single transaction spans multiple rows.
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
        
        # Enhanced logic to handle multiple consecutive rows belonging to same logical row
        i = 1
        while i < len(df):
            # Skip if this row is already marked for deletion
            if i in rows_to_drop:
                i += 1
                continue
                
            try:
                # Count non-empty cells in the current row
                row_data = df.iloc[i].astype(str).str.strip()
                # Replace 'None', 'nan', empty strings with pd.NA
                row_data = row_data.replace(['None', 'nan', '', 'NaN'], pd.NA)
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
                                
                            curr_row_data = df.iloc[j].astype(str).str.strip()
                            curr_row_data = curr_row_data.replace(['None', 'nan', '', 'NaN'], pd.NA)
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
                row_data = df.iloc[target_idx].astype(str).str.strip()
                row_data = row_data.replace(['None', 'nan', '', 'NaN'], pd.NA)
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
        source_row = df.iloc[source_idx].astype(str).str.strip()
        source_row = source_row.replace(['None', 'nan', '', 'NaN'], pd.NA)
        
        for col_idx, value in enumerate(source_row):
            if pd.notna(value) and str(value).strip():
                # Get the current value in the target row
                target_val = str(df.at[target_idx, df.columns[col_idx]]) if pd.notna(df.at[target_idx, df.columns[col_idx]]) else ""
                source_val = str(value).strip()
                
                # Merge the content
                if target_val.strip():
                    df.at[target_idx, df.columns[col_idx]] = target_val + " " + source_val
                else:
                    df.at[target_idx, df.columns[col_idx]] = source_val
    
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