#!/usr/bin/env python3
"""
Analyze Gemini API call patterns and counts
"""

import pandas as pd
import json
from pathlib import Path
import re

def analyze_api_calls():
    """Analyze how many Gemini API calls were made"""
    
    print("🔍 Analyzing Gemini API Call Patterns")
    print("=" * 50)
    
    # Check output files
    output_dir = Path("output")
    
    # Analyze gemini_classified_data.csv
    gemini_file = output_dir / "gemini_classified_data.csv"
    if gemini_file.exists():
        df = pd.read_csv(gemini_file)
        print(f"\n📊 Gemini Classified Data Analysis:")
        print(f"   Total rows processed: {len(df)}")
        
        # Check if there's a method column indicating LLM vs keyword
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            print(f"   LLM classifications: {method_counts.get('llm', 0)}")
            print(f"   Keyword classifications: {method_counts.get('keyword', 0)}")
        else:
            print("   ⚠️  No 'method' column found - can't determine LLM vs keyword usage")
    
    # Analyze batch_classified_data.csv
    batch_file = output_dir / "batch_classified_data.csv"
    if batch_file.exists():
        df = pd.read_csv(batch_file)
        print(f"\n📦 Batch Classified Data Analysis:")
        print(f"   Total rows processed: {len(df)}")
        
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            print(f"   LLM classifications: {method_counts.get('llm', 0)}")
            print(f"   Keyword classifications: {method_counts.get('keyword', 0)}")
        else:
            print("   ⚠️  No 'method' column found")
    
    # Analyze analysis_results.json
    analysis_file = output_dir / "analysis_results.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        print(f"\n📈 Analysis Results:")
        print(f"   Total generations: {analysis.get('total_generations', 'N/A')}")
        print(f"   Unique users: {analysis.get('unique_users', 'N/A')}")
    
    # Check code patterns for API call limits
    print(f"\n🔧 Code Analysis:")
    
    # Check process_with_gemini.py
    gemini_code = Path("process_with_gemini.py")
    if gemini_code.exists():
        with open(gemini_code, 'r') as f:
            content = f.read()
        
        # Find daily limit
        daily_limit_match = re.search(r'max_requests_per_day\s*=\s*(\d+)', content)
        if daily_limit_match:
            daily_limit = daily_limit_match.group(1)
            print(f"   Daily API limit set to: {daily_limit} requests")
        
        # Find rate limiting
        rate_limit_match = re.search(r'time\.sleep\((\d+)\)', content)
        if rate_limit_match:
            sleep_time = rate_limit_match.group(1)
            print(f"   Rate limiting: {sleep_time} seconds between requests")
    
    # Check text_classifier.py
    text_classifier = Path("text_classifier.py")
    if text_classifier.exists():
        with open(text_classifier, 'r') as f:
            content = f.read()
        
        # Find rate limiting
        rate_limit_match = re.search(r'time\.sleep\((\d+)\)', content)
        if rate_limit_match:
            sleep_time = rate_limit_match.group(1)
            print(f"   Text classifier rate limiting: {sleep_time} seconds between requests")
    
    # Check batch_classifier.py
    batch_classifier = Path("batch_classifier.py")
    if batch_classifier.exists():
        with open(batch_classifier, 'r') as f:
            content = f.read()
        
        print(f"   Batch classifier found - processes multiple texts per API call")
    
    # Estimate actual API calls based on data
    print(f"\n📊 API Call Estimation:")
    
    if gemini_file.exists():
        df = pd.read_csv(gemini_file)
        total_rows = len(df)
        
        # If we have method column, count actual LLM calls
        if 'method' in df.columns:
            llm_calls = df['method'].value_counts().get('llm', 0)
            keyword_calls = df['method'].value_counts().get('keyword', 0)
            print(f"   Actual Gemini API calls made: {llm_calls}")
            print(f"   Keyword fallback calls: {keyword_calls}")
        else:
            # Estimate based on typical patterns
            print(f"   Estimated Gemini API calls: {min(total_rows, 50)} (assuming daily limit)")
            print(f"   Note: This is an estimate - check logs for actual count")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Add logging to track actual API calls made")
    print(f"   2. Consider batch processing to reduce API calls")
    print(f"   3. Monitor quota usage in Google AI Studio dashboard")
    print(f"   4. Implement better fallback strategies")

if __name__ == "__main__":
    analyze_api_calls() 