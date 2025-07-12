#!/usr/bin/env python3
"""
Test script for the Bank Statement OCR Processing API
"""

import requests
import json
import os
import time
import subprocess
import signal
import sys
from typing import Optional


def test_api_with_file(file_path: str, doctype: str = "BANKSTMT") -> Optional[dict]:
    """Test the API with a specific file"""
    
    url = "http://localhost:8000/ocr_process/"
    
    try:
        # Prepare the files and data
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'application/pdf')
            }
            data = {
                'output_dir': '',
                'doctype': doctype
            }
            
            # Make the API call
            response = requests.post(url, files=files, data=data)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Request Successful!")
            print(f"Response Status: {result.get('status')}")
            print(f"Response Status Code: {result.get('status_code')}")
            
            # Check if data is present
            if 'data' in result:
                data = result['data']
                print(f"Process ID: {data.get('processId')}")
                print(f"Document Type: {data.get('document_type')}")
                print(f"Page Count: {data.get('page_cnt')}")
                print(f"Headers Found: {len(data.get('pageWiseData', [{}])[0].get('headers', {}))}")
                print(f"Table Rows: {len(data.get('lineTabulaData', []))}")
                
            return result
        else:
            print(f"❌ API Request Failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on localhost:8000")
        return None
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return None


def start_api_server():
    """Start the API server in the background"""
    
    print("🚀 Starting API server...")
    
    # Start the server
    process = subprocess.Popen(
        [sys.executable, "api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API server is running successfully!")
            return process
        else:
            print("❌ API server failed to start properly")
            return None
    except:
        print("❌ API server is not responding")
        return None


def main():
    """Main test function"""
    
    print("=== Bank Statement OCR Processing API Test ===")
    print()
    
    # Start API server
    server_process = start_api_server()
    
    if not server_process:
        print("Failed to start API server")
        return
    
    try:
        # Test with a sample bank statement
        test_files = [
            "BankStatements SK2/axis_bank__statement_for_september_2024_unlocked.pdf",
            "BankStatements SK2/Bank Statement - ICICI BANK.pdf",
            "BankStatements SK2/Kotak.pdf"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"\n🔍 Testing with: {os.path.basename(file_path)}")
                result = test_api_with_file(file_path)
                
                if result:
                    print(f"✅ Successfully processed {os.path.basename(file_path)}")
                else:
                    print(f"❌ Failed to process {os.path.basename(file_path)}")
                break
        else:
            print("❌ No test files found")
            
    finally:
        # Clean up - stop the server
        if server_process:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            server_process.wait()
            print("✅ API server stopped")


if __name__ == "__main__":
    main() 