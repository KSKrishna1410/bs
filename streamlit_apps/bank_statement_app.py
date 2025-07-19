#!/usr/bin/env python3
"""
Streamlit UI for Bank Statement OCR Processing

This app provides a user-friendly interface to upload bank statements,
process them through the OCR API, and display results in both raw and
structured formats.
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Optional, Dict, Any
import io
import time
import base64
from datetime import datetime
import os
import tempfile
from PIL import Image

# Authentication configuration
AUTH_CREDENTIALS = {
    "admin": os.getenv("ADMIN_PASSWORD", "admin123"),
    "user": os.getenv("USER_PASSWORD", "user123"),
    "demo": os.getenv("DEMO_PASSWORD", "demo123")
}

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def login_form():
    """Display login form"""
    st.title("🔐 Nekkanti OCR - Login")
    st.markdown("---")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Please Login to Continue")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button:
                    if username in AUTH_CREDENTIALS and AUTH_CREDENTIALS[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            # Demo credentials info
            with st.expander("📋 Demo Credentials"):
                st.markdown("""
                **Available Demo Accounts:**
                - Username: `admin`, Password: `admin123`
                - Username: `user`, Password: `user123`  
                - Username: `demo`, Password: `demo123`
                """)

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# Page configuration
st.set_page_config(
    page_title="Nekkanti OCR - Bank Statement Extractor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/ocr_process/"

def process_file_with_api(file_bytes: bytes, filename: str) -> Optional[Dict[Any, Any]]:
    """Process file through the API"""
    try:
        files = {'file': (filename, io.BytesIO(file_bytes), 'application/pdf')}
        data = {
            'output_dir': '',
            'doctype': 'BANKSTMT'
        }
        
        with st.spinner('Processing file through OCR API... This may take several minutes for large or complex files.'):
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=600)  # 10 minutes
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out after 10 minutes. The file might be too large or complex.")
        st.info("💡 **Tips to resolve timeout issues:**")
        st.info("• Try with a smaller file size")
        st.info("• Use a PDF instead of an image if possible")
        st.info("• Check if the file is corrupted")
        st.info("• For very large files, consider splitting them into smaller parts")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_headers(headers_data):
    """Display headers in a structured format"""
    st.subheader("📋 Extracted Headers")
    
    if not headers_data:
        st.warning("No headers found in the document")
        return
    
    # Group headers by category
    account_info = []
    bank_info = []
    statement_info = []
    balance_info = []
    
    for header in headers_data:
        if not isinstance(header, dict) or 'key' not in header or 'value' not in header:
            continue
            
        field_name = header['key']
        
        # Handle string representation of dictionary
        try:
            if isinstance(header['value'], str) and header['value'].startswith('{') and header['value'].endswith('}'):
                # Convert string representation to dict using ast.literal_eval
                import ast
                value_dict = ast.literal_eval(header['value'])
                value = value_dict.get('value', '')
                method = value_dict.get('method', '')
                confidence = value_dict.get('confidence', 0)
                validation_score = value_dict.get('validation_score', 0)
            else:
                value = header['value']
                method = header.get('method', '')
                confidence = header.get('confidence', 0)
                validation_score = header.get('validation_score', 0)
        except:
            # Fallback if parsing fails
            value = header['value']
            method = header.get('method', '')
            confidence = header.get('confidence', 0)
            validation_score = header.get('validation_score', 0)
            
        if not value:  # Skip empty values
            continue
            
        header_data = {
            'Field': field_name,
            'Value': value,
            'Method': method,
            'Confidence': f"{confidence:.2%}" if confidence > 0 else "N/A",
            'Validation': f"{validation_score:.2%}" if validation_score > 0 else "N/A"
        }
        
        # Categorize headers
        if field_name in ['Account Number', 'IFSC Code']:
            account_info.append(header_data)
        elif field_name.startswith('Bank'):
            bank_info.append(header_data)
        elif 'Date' in field_name:
            statement_info.append(header_data)
        elif 'Balance' in field_name:
            balance_info.append(header_data)
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        if account_info:
            st.markdown("**🏦 Account Information**")
            df = pd.DataFrame(account_info)
            st.dataframe(
                df,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                    "Method": st.column_config.TextColumn("Method", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Validation": st.column_config.TextColumn("Validation", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
        
        if statement_info:
            st.markdown("**📅 Statement Information**")
            df = pd.DataFrame(statement_info)
            st.dataframe(
                df,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                    "Method": st.column_config.TextColumn("Method", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Validation": st.column_config.TextColumn("Validation", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
    
    with col2:
        if bank_info:
            st.markdown("**🏢 Bank Information**")
            df = pd.DataFrame(bank_info)
            st.dataframe(
                df,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                    "Method": st.column_config.TextColumn("Method", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Validation": st.column_config.TextColumn("Validation", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
        
        if balance_info:
            st.markdown("**💰 Balance Information**")
            df = pd.DataFrame(balance_info)
            st.dataframe(
                df,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium"),
                    "Method": st.column_config.TextColumn("Method", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Validation": st.column_config.TextColumn("Validation", width="small")
                },
                use_container_width=True,
                hide_index=True
            )

def display_table_data(table_data: list) -> None:
    """Display table data in a structured format"""
    st.subheader("📊 Transaction Table")
    
    if not table_data:
        st.warning("No transaction data found in the document")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    if df.empty:
        st.warning("Transaction table is empty")
        return
    
    # Display table without metrics
    
    # Display the table
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="bank_statement_transactions.csv",
        mime="text/csv"
    )

def display_summary_metrics(response_data: dict) -> None:
    """Display summary metrics"""
    data = response_data.get('data', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Document Type", data.get('document_type', 'N/A'))
    
    with col2:
        st.metric("Page Count", data.get('page_cnt', 'N/A'))
    
    with col3:
        page_wise_data = data.get('pageWiseData', [])
        headers_count = len(page_wise_data[0].get('headerInfo', [])) if page_wise_data else 0
        st.metric("Headers Found", headers_count)
    
    with col4:
        table_rows = len(data.get('lineTabulaData', []))
        st.metric("Table Rows", table_rows)


def display_file_preview(file_bytes: bytes, filename: str) -> None:
    """Display file preview in Streamlit (PDF or Image)"""
    st.subheader("📄 File Preview")
    
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    try:
        if file_extension == 'pdf':
            try:
                # First try: Use PyMuPDF to convert PDF pages to images
                import fitz  # PyMuPDF
                
                # Create a temporary file to handle the PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name
                
                try:
                    # Open the PDF
                    pdf_document = fitz.open(temp_path)
                    
                    # Get number of pages
                    num_pages = len(pdf_document)
                    
                    if num_pages > 0:
                        # Create tabs for multiple pages
                        if num_pages > 1:
                            tabs = st.tabs([f"Page {i+1}" for i in range(num_pages)])
                        else:
                            tabs = [st.container()]  # Single container for one page
                        
                        # Display each page
                        for page_num in range(num_pages):
                            with tabs[page_num]:
                                # Get the page
                                page = pdf_document[page_num]
                                
                                # Convert page to image
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                                
                                # Convert to PIL Image
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                
                                # Display the image
                                st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                                
                                # Add download button for each page
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format='PNG')
                                st.download_button(
                                    label=f"📥 Download Page {page_num + 1} as Image",
                                    data=img_bytes.getvalue(),
                                    file_name=f"{filename}_page_{page_num + 1}.png",
                                    mime="image/png"
                                )
                    else:
                        st.warning("No pages found in the PDF")
                    
                    pdf_document.close()
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except Exception as pdf_error:
                st.warning(f"Could not render PDF preview: {str(pdf_error)}")
                st.info("Falling back to basic PDF preview...")
                
                # Fallback: Basic PDF display with iframe
                base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
                pdf_display = f"""
                <embed
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="100%"
                    height="800px"
                    type="application/pdf"
                >
                """
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Always provide download option
            st.download_button(
                label="📥 Download Original PDF",
                data=file_bytes,
                file_name=filename,
                mime="application/pdf"
            )
            
        elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']:
            # Display Image
            try:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(file_bytes))
                
                # Display image with improved quality
                st.image(image, caption=filename, use_container_width=True)
                
                # Add image info
                st.info(f"📊 Image Info: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
                
                # Add download button
                img_bytes = io.BytesIO()
                image.save(img_bytes, format=image.format or 'PNG')
                st.download_button(
                    label="📥 Download Image",
                    data=img_bytes.getvalue(),
                    file_name=filename,
                    mime=f"image/{image.format.lower() if image.format else 'png'}"
                )
            except Exception as img_error:
                st.error(f"Could not display image preview: {str(img_error)}")
                st.warning("Providing download option instead")
                st.download_button(
                    label="📥 Download Image",
                    data=file_bytes,
                    file_name=filename,
                    mime=f"image/{file_extension}"
                )
            
        else:
            # Fallback for unknown types
            st.warning(f"Preview not available for {file_extension.upper()} files")
            st.download_button(
                label="📥 Download file to view",
                data=file_bytes,
                file_name=filename,
                mime="application/octet-stream"
            )
        
    except Exception as e:
        st.error(f"Could not display file preview: {str(e)}")
        
        # Fallback: provide download button
        mime_type = "application/pdf" if file_extension == 'pdf' else f"image/{file_extension}"
        st.download_button(
            label="📥 Download file to view",
            data=file_bytes,
            file_name=filename,
            mime=mime_type
        )

def main():
    """Main application"""
    
    # Authentication check
    if not check_authentication():
        login_form()
        return
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.markdown(f"👤 **Welcome, {st.session_state.get('username', 'User')}!**")
        if st.button("🚪 Logout"):
            logout()
        st.markdown("---")

    # Header
    st.markdown('<div class="main-header">🏦 Nekkanti OCR - Bank Statement Extractor</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** your bank statement (PDF or image file)
        2. **Wait** while it's processed through our OCR engine
        3. **View** extracted headers and tables
        4. **Download** results in CSV format
        """)
    
    # File upload
    st.header("📤 Upload Bank Statement")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
        help="Upload your bank statement in PDF or image format (PNG, JPG, JPEG, BMP, TIFF, TIF)",
        on_change=None  # Remove any callback if present
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        current_file = getattr(st.session_state, 'current_file', None)
        if current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.file_processed = False
        
        # File info
        file_size_kb = uploaded_file.size / 1024
        file_size_mb = file_size_kb / 1024
        
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{file_size_mb:.2f} MB" if file_size_mb > 1 else f"{file_size_kb:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        # Warning for large files
        if file_size_mb > 10:
            st.warning("⚠️ **Large File Detected**")
            st.warning(f"File size: {file_size_mb:.2f} MB")
            st.warning("Large files may take longer to process and could timeout. Consider:")
            st.warning("• Compressing the PDF")
            st.warning("• Using a smaller resolution image")
            st.warning("• Splitting large documents into smaller parts")
        
        elif file_size_mb > 5:
            st.info(f"ℹ️ **Medium File Size**: {file_size_mb:.2f} MB - Processing may take 2-5 minutes")
        
        # Read file bytes for preview and processing
        file_bytes = uploaded_file.read()
        
        # Store file bytes in session state for preview
        st.session_state.uploaded_file_bytes = file_bytes
        st.session_state.uploaded_filename = uploaded_file.name
        
        # Display file info
        st.info("📄 **File Information**")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Display file preview
        st.markdown("---")
        display_file_preview(file_bytes, uploaded_file.name)
        
        # Auto-process if not already processed
        if not getattr(st.session_state, 'file_processed', False):
            with st.spinner("🔄 Processing file automatically..."):
                result = process_file_with_api(file_bytes, uploaded_file.name)
                
                if result:
                    # Store result in session state
                    st.session_state.processing_result = result
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.file_processed = True
                    st.success("✅ File processed successfully!")
                    # Force a rerun to update the UI
                    st.rerun()
    
    # Display results if available
    if hasattr(st.session_state, 'processing_result') and st.session_state.processing_result:
        st.header("📊 Processing Results")
        
        # Summary metrics
        st.subheader("📈 Summary")
        display_summary_metrics(st.session_state.processing_result)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["🎯 Structured View", "📄 Raw Response"])
        
        with tab1:
            # Structured view
            data = st.session_state.processing_result.get('data', {})
            
            # Headers
            page_wise_data = data.get('pageWiseData', [])
            if page_wise_data:
                headers_data = page_wise_data[0].get('headerInfo', {})
                display_headers(headers_data)
            
            st.markdown("---")
            
            # Table data
            table_data = data.get('lineTabulaData', [])
            display_table_data(table_data)
        
        with tab2:
            # Raw response
            st.subheader("🔍 Raw API Response")
            st.json(st.session_state.processing_result)
            
            # Download raw response
            json_str = json.dumps(st.session_state.processing_result, indent=2)
            st.download_button(
                label="📥 Download Raw Response",
                data=json_str,
                file_name=f"{st.session_state.processed_filename}_response.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main() 