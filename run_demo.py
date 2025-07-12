#!/usr/bin/env python3
"""
Demo Runner Script

This script helps run both the API server and Streamlit app together
for easy demonstration of the bank statement processing system.
"""

import subprocess
import sys
import time
import os
import signal
import webbrowser
from threading import Thread
import requests

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_api_server():
    """Run the FastAPI server"""
    print("🚀 Starting API server...")
    try:
        # Start the API server
        api_process = subprocess.Popen(
            [sys.executable, "api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("⏳ Waiting for API server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_api_health():
                print("✅ API server is running at http://localhost:8000")
                return api_process
            time.sleep(1)
        
        print("❌ API server failed to start within 30 seconds")
        api_process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def run_streamlit_app():
    """Run the Streamlit app"""
    print("🎨 Starting Streamlit app...")
    try:
        # Start Streamlit
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for Streamlit to start
        time.sleep(3)
        print("✅ Streamlit app is running at http://localhost:8501")
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        return streamlit_process
        
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None

def main():
    """Main function to run both services"""
    print("🏦 Nekkanti OCR Demo")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ["api.py", "streamlit_app.py", "comprehensive_bank_statement_processor.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    api_process = None
    streamlit_process = None
    
    try:
        # Start API server
        api_process = run_api_server()
        if not api_process:
            print("❌ Failed to start API server")
            return
        
        # Start Streamlit app
        streamlit_process = run_streamlit_app()
        if not streamlit_process:
            print("❌ Failed to start Streamlit app")
            return
        
        print("\n🎉 Both services are running!")
        print("📊 API Server: http://localhost:8000")
        print("🎨 Streamlit App: http://localhost:8501")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both services...")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
            
            if streamlit_process.poll() is not None:
                print("❌ Streamlit app stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
    finally:
        # Clean up processes
        if api_process:
            print("🔄 Stopping API server...")
            api_process.terminate()
            api_process.wait()
        
        if streamlit_process:
            print("🔄 Stopping Streamlit app...")
            streamlit_process.terminate()
            streamlit_process.wait()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 