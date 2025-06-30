#!/usr/bin/env python3
"""
Simple test script to verify Gemini API key
"""

import google.generativeai as genai
import os

def test_gemini_api():
    """Test the Gemini API with the provided key"""
    
    # API key
    api_key = "AIzaSyD4aitYsivhcrJPkHsxFaQR9s2Rij_1pm4"
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test with flash-lite model
        print("Testing with gemini-1.5-flash-latest...")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Simple test
        response = model.generate_content("Hello, this is a test. Please respond with 'API working' if you can see this.")
        print(f"Response: {response.text}")
        print("✅ API key is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api() 