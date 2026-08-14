"""
Pytest configuration and fixtures.

This file is automatically discovered and run by pytest before any tests.
It ensures the project root is in the Python path so modules can be imported.
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
