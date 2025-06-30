#!/usr/bin/env python3
"""
Simple test script for the improved Text Classifier

This script demonstrates the text classification functionality
with both LLM and keyword-based fallback methods.
"""

import os
from dotenv import load_dotenv
from text_classifier import TextClassifier, DataProcessor, UserPatternAnalyzer, ReportGenerator

# Load environment variables
load_dotenv()

def test_text_classification():
    """
    Test the text classifier with sample texts
    """
    print("🧪 Testing Text Classifier")
    print("=" * 50)
    
    # Sample texts for testing
    test_texts = [
        {
            "text": "Once upon a time, there was a brave knight who lived in a magical kingdom. He embarked on a quest to save the princess from an evil dragon.",
            "expected_primary": "entertainment",
            "expected_secondary": "storybook"
        },
        {
            "text": "Chapter 1: Introduction to Machine Learning. This chapter covers the fundamental concepts of supervised learning algorithms.",
            "expected_primary": "learning/productivity", 
            "expected_secondary": "textbook_or_pdf"
        },
        {
            "text": "Just posted on r/technology about the latest AI developments. The community had some interesting discussions about the future of automation.",
            "expected_primary": "online_content",
            "expected_secondary": "reddit_post"
        },
        {
            "text": "Weekly Newsletter: Top 10 Productivity Tips for Remote Workers. Subscribe to get more insights delivered to your inbox every Monday.",
            "expected_primary": "learning/productivity",
            "expected_secondary": "newsletter_substack"
        }
    ]
    
    # Initialize classifier
    print("🔧 Initializing Text Classifier...")
    
    # Try to get API key from environment
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if gemini_api_key:
        print("✅ Using Gemini LLM for classification")
        classifier = TextClassifier(gemini_api_key=gemini_api_key, use_llm=True)
    else:
        print("⚠️  No Gemini API key found. Using keyword-based classification")
        classifier = TextClassifier(use_llm=False)
    
    print("\n📝 Testing Classification:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {test_case['text'][:100]}...")
        
        # Classify the text
        primary, secondary = classifier.classify_text(test_case['text'])
        
        print(f"Result: Primary={primary}, Secondary={secondary}")
        print(f"Expected: Primary={test_case['expected_primary']}, Secondary={test_case['expected_secondary']}")
        
        # Check if results match expectations
        primary_match = primary == test_case['expected_primary']
        secondary_match = secondary == test_case['expected_secondary']
        
        if primary_match and secondary_match:
            print("✅ Perfect match!")
        elif primary_match:
            print("✅ Primary category matches")
        elif secondary_match:
            print("✅ Secondary category matches")
        else:
            print("❌ No matches, but classification completed")
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")

def test_sample_data_processing():
    """
    Test the text classifier with the sample_data.csv file
    """
    print("\n📊 Testing Sample Data Processing")
    print("=" * 50)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        processor = DataProcessor(chunk_size=5)  # Small chunks for testing
        analyzer = UserPatternAnalyzer(bias_threshold=0.2)
        report_generator = ReportGenerator()
        
        # Get API key from environment
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            print("✅ Using Gemini LLM for classification")
            processor.classifier = TextClassifier(gemini_api_key=gemini_api_key, use_llm=True)
        else:
            print("⚠️  No Gemini API key found. Using keyword-based classification")
            processor.classifier = TextClassifier(use_llm=False)
        
        # Load sample data
        print("📁 Loading sample_data.csv...")
        df = processor.load_data("sample_data.csv")
        print(f"✅ Loaded {len(df)} rows from sample_data.csv")
        
        # Process data
        print("🔄 Processing data...")
        processed_df = processor.process_data_in_chunks(df)
        print(f"✅ Processed {len(processed_df)} rows")
        
        # Analyze patterns
        print("📈 Analyzing user patterns...")
        analysis_results = analyzer.analyze_user_patterns(processed_df)
        
        # Show summary
        print(f"\n📊 Analysis Summary:")
        print(f"Total generations: {analysis_results['total_generations']}")
        print(f"Unique users: {analysis_results['unique_users']}")
        
        # Show primary categories
        print(f"\n🎯 Primary Categories:")
        for category, count in analysis_results['primary_category_stats']['generations'].items():
            percentage = (count / analysis_results['total_generations']) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Show secondary categories
        print(f"\n📚 Secondary Categories:")
        for category, count in analysis_results['secondary_category_stats']['generations'].items():
            percentage = (count / analysis_results['total_generations']) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Show classification method stats
        print(f"\n🔧 Classification Method Statistics:")
        print(f"  LLM classifications: {processor.llm_processed_count}")
        print(f"  Keyword classifications: {processor.fallback_processed_count}")
        
        # Generate reports
        print("\n📄 Generating reports...")
        report_generator.generate_report(processed_df, analysis_results, processor)
        print("✅ Reports generated in output/ directory")
        
        print("\n" + "=" * 50)
        print("🎉 Sample data processing completed!")
        
    except FileNotFoundError:
        print("❌ sample_data.csv not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error during sample data processing: {e}")

def test_single_text():
    """
    Test classification of a single text input
    """
    print("\n🔍 Single Text Classification Test")
    print("=" * 50)
    
    # Get user input
    user_text = input("Enter text to classify (or press Enter for sample): ").strip()
    
    if not user_text:
        user_text = "The latest research shows that machine learning algorithms can significantly improve productivity in various industries."
        print(f"Using sample text: {user_text}")
    
    # Initialize classifier
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    classifier = TextClassifier(gemini_api_key=gemini_api_key, use_llm=bool(gemini_api_key))
    
    # Classify
    print("\n🔍 Classifying...")
    primary, secondary = classifier.classify_text(user_text)
    
    print(f"\n📊 Classification Results:")
    print(f"Primary Category: {primary}")
    print(f"Secondary Category: {secondary}")
    
    # Provide explanation
    category_explanations = {
        'entertainment': 'Content for leisure, fun, or creative purposes',
        'learning/productivity': 'Educational content, tutorials, or work-related materials',
        'online_content': 'Social media posts, forum discussions, or online articles',
        'storybook': 'Fictional narratives, stories, or creative writing',
        'textbook_or_pdf': 'Academic content, technical documentation, or reference materials',
        'newsletter_substack': 'Email newsletters, subscriptions, or regular updates',
        'reddit_post': 'Social media content, forum posts, or community discussions',
        'other': 'Content that doesn\'t fit the specific categories above'
    }
    
    print(f"\n💡 Explanation:")
    print(f"Primary: {category_explanations.get(primary, 'Unknown category')}")
    print(f"Secondary: {category_explanations.get(secondary, 'Unknown category')}")

if __name__ == "__main__":
    print("🚀 Text Classifier Test Suite")
    print("=" * 50)
    
    try:
        # Run automated tests
        test_text_classification()
        
        # Run sample data processing test
        test_sample_data_processing()
        
        # Run interactive test
        test_single_text()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Make sure you have all dependencies installed:")
        print("pip install -r requirements.txt") 