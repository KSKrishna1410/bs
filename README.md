# 🏦 Nekkanti OCR

A comprehensive system for extracting headers and transaction tables from bank statements using OCR technology, featuring both API and web UI interfaces.

## 🚀 Features

### 📋 **Header Extraction**
- **Account Information**: Account Number, IFSC Code
- **Bank Details**: Bank Name, Address, Branch, City, State, Phone
- **Statement Information**: Statement dates, period coverage
- **Balance Information**: Opening and closing balances
- **IFSC Integration**: Automatic bank details lookup using IFSC codes

### 📊 **Table Extraction**
- **Transaction Detection**: Automatic identification of transaction tables
- **Multi-page Support**: Handles multi-page bank statements
- **Smart Consolidation**: Combines transaction tables from multiple pages
- **Data Validation**: Filters out irrelevant rows and headers

### 🔧 **Processing Methods**
- **OCR Pipeline**: Handles both readable and scanned PDFs
- **Multi-tier Account Detection**: Keyword → Regex → Label Proximity
- **Fragmented Number Handling**: Reconstructs account numbers split across text elements
- **Smart Validation**: Context-aware filtering to prevent false positives

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Core          │
│   (Frontend)    │────│   (API Layer)   │────│   Processors    │
│   Port: 8501    │    │   Port: 8888    │    │   (Backend)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Core Components**
1. **`comprehensive_bank_statement_processor.py`** - Main processing engine
2. **`bank_statement_header_extractor.py`** - Header extraction logic
3. **`bank_statement_extractor.py`** - Table extraction logic
4. **`api.py`** - FastAPI service layer
5. **`streamlit_app.py`** - Web UI interface

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.8+
- Virtual environment (recommended)

### **Setup**
```bash
# Clone the repository
git clone <repository-url>
cd glbyte_bs

# Create virtual environment
python -m venv venv-v5
source venv-v5/bin/activate  # On Windows: venv-v5\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 **Usage**

### **Option 1: Quick Demo (Recommended)**
```bash
python run_demo.py
```
This will start both the API server and Streamlit app automatically.

### **Option 2: Manual Setup**

#### **Start API Server**
```bash
python api.py
```
- API will be available at: http://localhost:8888
- API Documentation: http://localhost:8888/docs

#### **Start Streamlit App**
```bash
streamlit run streamlit_app.py --server.port=8501
```
- UI will be available at: http://localhost:8501

### **Option 3: Direct API Usage**
```bash
curl -X POST "http://localhost:8888/ocr_process/" \
  -F "file=@your_bank_statement.pdf" \
  -F "output_dir=" \
  -F "doctype=BANKSTMT"
```

## 📱 **Web UI Features**

### **📄 PDF Preview & Processing**
- **Instant Preview**: View uploaded PDF immediately upon selection
- **Side-by-side Layout**: PDF preview alongside extracted results
- **Embedded Viewer**: Native browser PDF display with fallback options

### **🎯 Structured View**
- **Header Display**: Organized by categories (Account, Bank, Statement, Balance)
- **Table Visualization**: Interactive transaction table with search and filter
- **Download Options**: Export to CSV format

### **📄 Raw Response View**
- **JSON Display**: Complete API response in formatted JSON
- **Download**: Save raw response as JSON file

### **📄 PDF Preview**
- **Live Preview**: Embedded PDF viewer showing uploaded document
- **Side-by-side View**: PDF preview alongside extracted data
- **Fallback Download**: Download option if preview fails

## 🔌 **API Reference**

### **Endpoints**

#### **POST /ocr_process/**
Process a bank statement file and extract headers and tables.

**Parameters:**
- `file` (required): PDF file to process
- `output_dir` (optional): Output directory (leave blank)
- `doctype` (required): Document type ("BANKSTMT")

**Response:**
```json
{
  "status_code": 200,
  "status": "Success",
  "data": {
    "processId": "uuid-string",
    "document_type": "BANKSTMT",
    "page_cnt": 4,
    "pageWiseData": [...],
    "lineTabulaData": [...]
  }
}
```

#### **GET /health**
Check API health status.

#### **GET /**
API information and available endpoints.

## 📊 **Performance Metrics**

### **Current Success Rates**
- **Account Number Detection**: 100% (7/7 test files)
- **Header Extraction**: 85.7% overall success rate
- **Table Extraction**: 90%+ accuracy on readable PDFs
- **IFSC Lookup**: 100% when valid IFSC codes are present

### **Processing Speed**
- **Readable PDFs**: 2-5 seconds per page
- **Scanned PDFs**: 10-15 seconds per page (with OCR)
- **Multi-page Documents**: Linear scaling with page count

## 🗂️ **Supported Banks**

Tested and verified with statements from:
- ✅ Axis Bank
- ✅ ICICI Bank
- ✅ HDFC Bank
- ✅ State Bank of India (SBI)
- ✅ IDFC First Bank
- ✅ Indian Overseas Bank (IOB)
- ✅ Kotak Mahindra Bank
- ✅ Union Bank of India
- ✅ Canara Bank

## 📁 **File Structure**

```
glbyte_bs/
├── api.py                              # FastAPI service
├── streamlit_app.py                    # Streamlit UI
├── run_demo.py                         # Demo runner script
├── comprehensive_bank_statement_processor.py  # Main processor
├── bank_statement_header_extractor.py  # Header extraction
├── bank_statement_extractor.py        # Table extraction
├── requirements.txt                    # Dependencies
├── README.md                          # This file
├── bankstmt_allkeys.csv              # Header keyword mappings
├── IFSC_master.csv                   # IFSC code database
└── BankStatements SK2/               # Sample input files
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **API Not Starting**
   ```bash
   # Check if port 8888 is available
   lsof -i :8888
   
   # Try different port
   uvicorn api:app --port 8889
   ```

2. **Streamlit Issues**
   ```bash
   # Clear cache
   streamlit cache clear
   
   # Try different port
   streamlit run streamlit_app.py --server.port=8502
   ```

3. **OCR Processing Errors**
   - Ensure PDF is not password-protected
   - Check file size (very large files may timeout)
   - Verify PDF format compatibility

### **Debug Mode**
Enable verbose logging by setting environment variable:
```bash
export DEBUG=1
python api.py
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **PaddleOCR** for OCR capabilities
- **img2table** for table extraction
- **FastAPI** for API framework
- **Streamlit** for UI framework
- **IFSC API** for bank code database

---

## 📞 **Support**

For issues, questions, or contributions:
- Create an issue on GitHub
- Email: support@example.com
- Documentation: [API Docs](http://localhost:8888/docs)

---

*Built with ❤️ for efficient bank statement processing*
