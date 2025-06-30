#!/usr/bin/env python3
"""
Detailed analysis of Gemini API calls made
"""

import pandas as pd
import json
from pathlib import Path

def detailed_api_analysis():
    """Detailed analysis of API calls made"""
    
    print("🔍 DETAILED GEMINI API CALL ANALYSIS")
    print("=" * 60)
    
    output_dir = Path("output")
    
    # Analyze all output files
    files_to_analyze = [
        "gemini_classified_data.csv",
        "batch_classified_data.csv", 
        "openai_batch_classified_data.csv",
        "classified_data.csv"
    ]
    
    total_api_calls = 0
    total_keyword_calls = 0
    total_rows = 0
    
    for filename in files_to_analyze:
        file_path = output_dir / filename
        if file_path.exists():
            print(f"\n📁 Analyzing: {filename}")
            df = pd.read_csv(file_path)
            total_rows += len(df)
            
            print(f"   Total rows: {len(df)}")
            
            if 'method' in df.columns:
                method_counts = df['method'].value_counts()
                llm_calls = method_counts.get('llm', 0) or 0
                keyword_calls = method_counts.get('keyword', 0) or 0
                
                total_api_calls += llm_calls
                total_keyword_calls += keyword_calls
                
                print(f"   LLM API calls: {llm_calls}")
                print(f"   Keyword fallback: {keyword_calls}")
                
                if llm_calls > 0:
                    print(f"   API call percentage: {llm_calls/len(df)*100:.1f}%")
                else:
                    print(f"   ⚠️  No API calls made - all keyword classification")
            else:
                print(f"   ⚠️  No 'method' column - cannot determine API usage")
    
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    print(f"Total rows processed across all files: {total_rows}")
    print(f"Total Gemini API calls made: {total_api_calls}")
    print(f"Total keyword fallback calls: {total_keyword_calls}")
    
    if total_rows > 0:
        api_percentage = (total_api_calls / total_rows) * 100
        print(f"Overall API usage: {api_percentage:.1f}%")
    
    # Check code limits
    print(f"\n🔧 CODE CONFIGURATION")
    print("=" * 60)
    
    # Check process_with_gemini.py limits
    gemini_file = Path("process_with_gemini.py")
    if gemini_file.exists():
        with open(gemini_file, 'r') as f:
            content = f.read()
        
        if 'max_requests_per_day = 50' in content:
            print(f"Daily API limit: 50 requests")
        if 'time.sleep(2)' in content:
            print(f"Rate limiting: 2 seconds between requests")
    
    # Check batch_classifier.py
    batch_file = Path("batch_classifier.py")
    if batch_file.exists():
        print(f"Batch processing available: Yes")
        print(f"   - Can process multiple texts per API call")
        print(f"   - More efficient than individual calls")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if total_api_calls == 0:
        print("❌ No Gemini API calls were made!")
        print("   Possible reasons:")
        print("   - API quota exceeded")
        print("   - API key issues")
        print("   - Fallback to keyword classification")
        print("   - Code errors")
    else:
        print(f"✅ {total_api_calls} Gemini API calls were made")
        print(f"   - {total_keyword_calls} texts used keyword fallback")
        
        if total_api_calls < total_rows:
            print(f"   - Consider batch processing to reduce API calls")
            print(f"   - Current efficiency: {total_api_calls}/{total_rows} texts per call")
    
    print(f"\n🔍 NEXT STEPS")
    print("=" * 60)
    print("1. Check Google AI Studio dashboard for quota usage")
    print("2. Review logs for any API errors")
    print("3. Consider upgrading to paid tier for higher limits")
    print("4. Implement better batch processing")
    print("5. Add more detailed logging for API call tracking")

if __name__ == "__main__":
    detailed_api_analysis() 