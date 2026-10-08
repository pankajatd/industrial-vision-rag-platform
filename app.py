"""
Industrial Multi-Agent Vision & Diagnostic RAG Platform
=======================================================
Main entry point for Streamlit Cloud deployment.
Redirects execution to streamlit_app.py.
"""
import sys
from pathlib import Path

# Ensure root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run the primary Streamlit application
import streamlit_app
