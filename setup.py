from setuptools import setup, find_packages

setup(
    name="bank_statements",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "PyMuPDF>=1.18.0",
        "paddleocr>=2.0.0",
        "img2table>=1.0.0",
        "reportlab>=3.6.0"
    ],
) 