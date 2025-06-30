#!/usr/bin/env python3
"""
Text Classifier and User Pattern Analysis Script

This script processes a table of text data and:
1. Classifies text content into primary and secondary categories using Gemini LLM
2. Analyzes user patterns and usage distribution
3. Processes data in chunks for efficient testing
4. Provides comprehensive reporting and analysis

Author: Shivani Bajaj
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier:
    """
    A text classifier that uses Gemini LLM for accurate content classification
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None, use_llm: bool = True):
        """
        Initialize the classifier
        
        Args:
            gemini_api_key: Gemini API key (if None, will try to get from environment)
            use_llm: If True, use LLM; if False, use fallback keyword classification
        """
        self.use_llm = use_llm
        self.model = None
        
        # Define classification categories
        self.primary_categories = ['entertainment', 'learning/productivity', 'online_content']
        self.secondary_categories = ['storybook', 'textbook_or_pdf', 'newsletter_substack', 'reddit_post', 'other']
        
        if self.use_llm:
            self._setup_gemini(gemini_api_key)
        
        # Fallback keyword rules (kept for backup)
        self.keyword_rules = {
            'primary': {
                'entertainment': [
                    'story', 'novel', 'fiction', 'fantasy', 'adventure', 'mystery', 'romance',
                    'character', 'plot', 'chapter', 'book', 'tale', 'narrative', 'fairy tale',
                    'princess', 'dragon', 'magic', 'hero', 'quest', 'journey'
                ],
                'learning/productivity': [
                    'chapter', 'section', 'learning', 'education', 'textbook',
                    'academic', 'research', 'analysis', 'method', 'technique', 'guide',
                    'tutorial', 'instruction', 'knowledge', 'skill', 'training'
                ],
                'online_content': [
                    'reddit', 'post', 'comment', 'thread', 'subreddit', 'upvote', 'downvote',
                    'moderator', 'community', 'discussion', 'forum', 'social media',
                    'tweet', 'thread', 'viral', 'trending'
                ]
            },
            'secondary': {
                'storybook': [
                    'story', 'book', 'tale', 'narrative', 'chapter', 'character', 'plot',
                    'beginning', 'ending', 'once upon a time', 'happily ever after'
                ],
                'textbook_or_pdf': [
                    'chapter', 'section', 'page', 'figure', 'table', 'reference',
                    'bibliography', 'index', 'glossary', 'appendix', 'footnote'
                ],
                'newsletter_substack': [
                    'newsletter', 'substack', 'subscribe', 'email', 'digest', 'weekly',
                    'monthly', 'update', 'news', 'insights', 'analysis', 'perspective'
                ],
                'reddit_post': [
                    'reddit', 'r/', 'subreddit', 'upvote', 'downvote', 'karma', 'moderator',
                    'community', 'thread', 'post', 'comment', 'edit:', 'update:'
                ]
            }
        }
    
    def _setup_gemini(self, api_key: Optional[str] = None) -> None:
        """
        Setup Gemini API
        
        Args:
            api_key: Gemini API key
        """
        try:
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Try to get from environment variable
                env_api_key = os.getenv('GEMINI_API_KEY')
                if env_api_key:
                    genai.configure(api_key=env_api_key)
                else:
                    logger.warning("No Gemini API key provided. Falling back to keyword classification.")
                    self.use_llm = False
                    return
            
            # Test the API with flash-lite model for higher rate limits
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content("Hello")
            self.model = model
            logger.info("Gemini API setup successful with flash-lite model (30 requests/minute)")
            
        except Exception as e:
            logger.error(f"Failed to setup Gemini API: {e}")
            logger.warning("Falling back to keyword classification")
            self.use_llm = False
    
    def _create_classification_prompt(self, text: str) -> str:
        """
        Create a prompt for text classification
        
        Args:
            text: Text to classify
            
        Returns:
            Classification prompt
        """
        prompt = f"""
You are a content classifier. Analyze the following text and classify it into the specified categories.

TEXT TO CLASSIFY:
{text[:2000]}  # Limit text length for API

CLASSIFICATION TASK:
1. Primary Category (choose one):
   - entertainment: Stories, novels, fiction, creative content, leisure reading
   - learning/productivity: Educational content, textbooks, academic papers, how-to guides, professional development
   - online_content: Social media posts, forum discussions, Reddit posts, online discussions

2. Secondary Category (choose one):
   - storybook: Fiction stories, novels, creative narratives, fairy tales
   - textbook_or_pdf: Academic content, technical documentation, research papers, educational materials
   - newsletter_substack: Newsletters, email digests, subscription content, regular updates
   - reddit_post: Reddit content, forum posts, social media discussions
   - other: Content that doesn't fit the above categories

RESPONSE FORMAT:
Return ONLY a JSON object with this exact format:
{{
    "primary_category": "category_name",
    "secondary_category": "category_name"
}}

Focus on the actual content and purpose, not just keywords. Consider the context and intent of the text.
"""
        return prompt
    
    def _classify_with_llm(self, text: str) -> Tuple[str, str]:
        """
        Classify text using Gemini LLM
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (primary_category, secondary_category)
        """
        if self.model is None:
            return self._fallback_classify(text)
        
        try:
            prompt = f"""
            Classify the following text into content categories. 
            
            Text: "{text[:1000]}"  # Limit text length
            
            Choose ONE primary category from:
            - entertainment (fun, stories, games, media)
            - learning/productivity (educational, work, skills, tutorials)
            - online_content (social media, blogs, forums, news)
            
            Choose ONE secondary category from:
            - storybook (narratives, fiction, bedtime stories)
            - textbook_or_pdf (academic, technical, reference)
            - newsletter_substack (email newsletters, subscriptions)
            - reddit_post (social media posts, discussions)
            - other (anything else)
            
            Respond in this exact format:
            Primary: [category]
            Secondary: [category]
            """
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse the response
            primary = "other"
            secondary = "other"
            
            for line in result.split('\n'):
                if line.startswith('Primary:'):
                    primary = line.replace('Primary:', '').strip().lower()
                elif line.startswith('Secondary:'):
                    secondary = line.replace('Secondary:', '').strip().lower()
            
            # Validate categories
            valid_primary = ['entertainment', 'learning/productivity', 'online_content']
            valid_secondary = ['storybook', 'textbook_or_pdf', 'newsletter_substack', 'reddit_post', 'other']
            
            if primary not in valid_primary:
                primary = "other"
            if secondary not in valid_secondary:
                secondary = "other"
                
            return primary, secondary
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "429" in error_msg:
                logger.warning(f"Gemini API quota exceeded: {e}")
                logger.info("Falling back to keyword classification for this text")
                return self._fallback_classify(text)
            else:
                logger.error(f"Gemini API error: {e}")
                return self._fallback_classify(text)
    
    def _fallback_classify(self, text: str) -> Tuple[str, str]:
        """
        Fallback to keyword-based classification
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (primary_category, secondary_category)
        """
        text_lower = text.lower()
        
        # Primary classification
        primary_scores = {}
        for category, keywords in self.keyword_rules['primary'].items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            primary_scores[category] = score
        
        # Find the category with the highest score
        if primary_scores and any(primary_scores.values()):
            primary_category = max(primary_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_category = 'entertainment'
        
        # Secondary classification
        secondary_scores = {}
        for category, keywords in self.keyword_rules['secondary'].items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            secondary_scores[category] = score
        
        # Find the category with the highest score
        if secondary_scores and any(secondary_scores.values()):
            secondary_category = max(secondary_scores.items(), key=lambda x: x[1])[0]
        else:
            secondary_category = 'other'
        
        return primary_category, secondary_category
    
    def classify_text(self, text: str) -> Tuple[str, str]:
        """
        Classify a single text into primary and secondary categories
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (primary_category, secondary_category)
        """
        if self.use_llm and self.model is not None:
            # Use LLM classification with rate limiting
            result = self._classify_with_llm(text)
            # Add delay to respect API rate limits (30 requests per minute = 2 seconds between requests)
            time.sleep(2)
            return result
        else:
            # Use fallback keyword classification
            return self._fallback_classify(text)


class DataProcessor:
    """
    Handles data loading, processing, and chunked operations
    """
    
    def __init__(self, chunk_size: int = 100):
        """
        Initialize the data processor
        
        Args:
            chunk_size: Number of rows to process in each chunk
        """
        self.chunk_size = chunk_size
        self.classifier = TextClassifier(use_llm=False)  # Start with fallback for simplicity
        self.llm_processed_count = 0
        self.fallback_processed_count = 0
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Validate required columns
        required_columns = ['email', 'text', 'voice']
        # Also check for common variations
        column_variations = {
            'email': ['email', 'Email', 'EMAIL'],
            'text': ['text', 'Text', 'TEXT'],
            'voice': ['voice', 'Voice', 'Voice Name', 'VOICE']
        }
        
        # Check if we have the required columns (with variations)
        found_columns = {}
        for required_col in required_columns:
            found = False
            for variation in column_variations[required_col]:
                if variation in df.columns:
                    found_columns[variation] = required_col  # Map original name to standard name
                    found = True
                    break
            if not found:
                missing_columns = [col for col in required_columns if col not in [found_columns.get(k, k) for k in df.columns]]
                raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
        
        # Rename columns to standard format
        df = df.rename(columns=found_columns)
        
        # Verify the renaming worked
        logger.info(f"Column mapping: {found_columns}")
        logger.info(f"Final columns: {list(df.columns)}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chunk of data and classify texts
        
        Args:
            chunk: DataFrame chunk to process
            
        Returns:
            Processed chunk with classification columns
        """
        results = []
        
        for idx, row in chunk.iterrows():
            try:
                # Track which classification method was used
                was_llm_before = self.classifier.use_llm
                
                # Convert row text to string to ensure compatibility
                text_content = str(row['text'])
                primary_cat, secondary_cat = self.classifier.classify_text(text_content)
                
                # Count based on whether LLM was used
                if was_llm_before and self.classifier.use_llm:
                    self.llm_processed_count += 1
                else:
                    self.fallback_processed_count += 1
                
                results.append({
                    'email': row['email'],
                    'text': text_content,
                    'voice': row['voice'],
                    'primary_category': primary_cat,
                    'secondary_category': secondary_cat
                })
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                self.fallback_processed_count += 1
                results.append({
                    'email': row['email'],
                    'text': str(row['text']),
                    'voice': row['voice'],
                    'primary_category': 'other',
                    'secondary_category': 'other'
                })
        
        return pd.DataFrame(results)
    
    def process_data_in_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire dataset in chunks
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame with classification results
        """
        all_results = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Processing {len(df)} rows in {total_chunks} chunks of size {self.chunk_size}")
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} rows)")
            
            processed_chunk = self.process_chunk(chunk)
            all_results.append(processed_chunk)
            
            # Print sample results for validation
            if chunk_num <= 3:  # Show first 3 chunks
                sample_rows = processed_chunk.sample(min(3, len(processed_chunk)))
                logger.info(f"Sample results from chunk {chunk_num}:")
                for _, row in sample_rows.iterrows():
                    logger.info(f"  Email: {row['email'][:30]}... | Primary: {row['primary_category']} | Secondary: {row['secondary_category']}")
        
        result_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Completed processing {len(result_df)} rows")
        
        return result_df


class UserPatternAnalyzer:
    """
    Analyzes user patterns and usage distribution
    """
    
    def __init__(self, bias_threshold: float = 0.2):
        """
        Initialize the analyzer
        
        Args:
            bias_threshold: Threshold for flagging disproportionate usage (default: 20%)
        """
        self.bias_threshold = bias_threshold
    
    def analyze_user_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze user patterns and generate comprehensive reports
        
        Args:
            df: Processed DataFrame with classification results
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing user patterns...")
        
        # Create summary table
        summary_df = df.groupby(['email', 'primary_category', 'secondary_category']).size().reset_index(name='count')
        
        # Calculate statistics
        total_generations = len(df)
        unique_users = df['email'].nunique()
        
        # Primary category analysis
        primary_stats = df['primary_category'].value_counts()
        primary_user_counts = df.groupby('primary_category')['email'].nunique()
        
        # Secondary category analysis
        secondary_stats = df['secondary_category'].value_counts()
        secondary_user_counts = df.groupby('secondary_category')['email'].nunique()
        
        # Identify potential bias
        bias_flags = self._identify_bias(df)
        
        # Top users by category
        top_users_by_category = self._get_top_users_by_category(df)
        
        results = {
            'summary_table': summary_df,
            'total_generations': total_generations,
            'unique_users': unique_users,
            'primary_category_stats': {
                'generations': primary_stats.to_dict(),
                'unique_users': primary_user_counts.to_dict()
            },
            'secondary_category_stats': {
                'generations': secondary_stats.to_dict(),
                'unique_users': secondary_user_counts.to_dict()
            },
            'bias_flags': bias_flags,
            'top_users_by_category': top_users_by_category
        }
        
        return results
    
    def _identify_bias(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify cases where users have disproportionate usage
        
        Args:
            df: Processed DataFrame
            
        Returns:
            List of bias flags
        """
        bias_flags = []
        
        # Check primary categories
        for category in df['primary_category'].unique():
            category_data = df[df['primary_category'] == category]
            total_in_category = len(category_data)
            
            # Ensure we're working with a DataFrame
            if isinstance(category_data, pd.DataFrame):
                user_counts = category_data['email'].value_counts()
                for user, count in user_counts.items():
                    percentage = count / total_in_category
                    if percentage > self.bias_threshold:
                        bias_flags.append({
                            'category_type': 'primary',
                            'category': category,
                            'user': user,
                            'count': count,
                            'total_in_category': total_in_category,
                            'percentage': percentage
                        })
        
        # Check secondary categories
        for category in df['secondary_category'].unique():
            category_data = df[df['secondary_category'] == category]
            total_in_category = len(category_data)
            
            # Ensure we're working with a DataFrame
            if isinstance(category_data, pd.DataFrame):
                user_counts = category_data['email'].value_counts()
                for user, count in user_counts.items():
                    percentage = count / total_in_category
                    if percentage > self.bias_threshold:
                        bias_flags.append({
                            'category_type': 'secondary',
                            'category': category,
                            'user': user,
                            'count': count,
                            'total_in_category': total_in_category,
                            'percentage': percentage
                        })
        
        return bias_flags
    
    def _get_top_users_by_category(self, df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Get top users by category
        
        Args:
            df: Processed DataFrame
            top_n: Number of top users to return
            
        Returns:
            Dictionary of top users by category
        """
        top_users = {}
        
        # Primary categories
        for category in df['primary_category'].unique():
            category_data = df[df['primary_category'] == category]
            if isinstance(category_data, pd.DataFrame):
                top_users[f'primary_{category}'] = category_data['email'].value_counts().head(top_n).to_dict()
        
        # Secondary categories
        for category in df['secondary_category'].unique():
            category_data = df[df['secondary_category'] == category]
            if isinstance(category_data, pd.DataFrame):
                top_users[f'secondary_{category}'] = category_data['email'].value_counts().head(top_n).to_dict()
        
        return top_users


class ReportGenerator:
    """
    Generates comprehensive reports and saves results
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the report generator
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, df: pd.DataFrame, analysis_results: Dict, processor: DataProcessor) -> None:
        """
        Generate comprehensive report
        
        Args:
            df: Processed DataFrame
            analysis_results: Results from user pattern analysis
            processor: DataProcessor instance
        """
        logger.info("Generating comprehensive report...")
        
        # Save processed data
        output_file = self.output_dir / "classified_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved classified data to {output_file}")
        
        # Save summary table
        summary_file = self.output_dir / "user_summary.csv"
        analysis_results['summary_table'].to_csv(summary_file, index=False)
        logger.info(f"Saved user summary to {summary_file}")
        
        # Save unique user analysis
        unique_user_file = self.output_dir / "unique_user_analysis.csv"
        self._save_unique_user_analysis(df, unique_user_file)
        logger.info(f"Saved unique user analysis to {unique_user_file}")
        
        # Save comprehensive analysis tables
        comprehensive_file = self.output_dir / "comprehensive_analysis.csv"
        self._save_comprehensive_analysis(df, analysis_results, comprehensive_file)
        logger.info(f"Saved comprehensive analysis to {comprehensive_file}")
        
        # Print comprehensive report
        self._print_report(df, analysis_results, processor)
        
        # Save detailed analysis as JSON
        analysis_file = self.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"Saved analysis results to {analysis_file}")
    
    def _save_unique_user_analysis(self, df: pd.DataFrame, output_file: Path) -> None:
        """
        Save unique user analysis to CSV file
        
        Args:
            df: Processed DataFrame
            output_file: Path to save the analysis
        """
        # Get unique users per primary category
        primary_unique_users = df.groupby('primary_category')['email'].nunique().reset_index()
        primary_unique_users.columns = ['category', 'unique_users']
        primary_unique_users['category_type'] = 'primary'
        
        # Get unique users per secondary category
        secondary_unique_users = df.groupby('secondary_category')['email'].nunique().reset_index()
        secondary_unique_users.columns = ['category', 'unique_users']
        secondary_unique_users['category_type'] = 'secondary'
        
        # Combine and save
        unique_user_analysis = pd.concat([primary_unique_users, secondary_unique_users], ignore_index=True)
        unique_user_analysis.to_csv(output_file, index=False)
    
    def _save_comprehensive_analysis(self, df: pd.DataFrame, analysis_results: Dict, output_file: Path) -> None:
        """
        Save comprehensive analysis tables to CSV file
        
        Args:
            df: Processed DataFrame
            analysis_results: Analysis results
            output_file: Path to save the analysis
        """
        # Create primary category table
        primary_stats = analysis_results['primary_category_stats']
        primary_data = []
        for category in primary_stats['generations'].keys():
            generations = primary_stats['generations'][category]
            unique_users = primary_stats['unique_users'][category]
            gen_percentage = (generations / analysis_results['total_generations']) * 100
            user_percentage = (unique_users / analysis_results['unique_users']) * 100
            primary_data.append({
                'Category': category,
                'Generations': generations,
                'Unique Users': unique_users,
                'Gen %': f"{gen_percentage:.1f}%",
                'User %': f"{user_percentage:.1f}%"
            })
        
        primary_df = pd.DataFrame(primary_data)
        
        # Create secondary category table
        secondary_stats = analysis_results['secondary_category_stats']
        secondary_data = []
        for category in secondary_stats['generations'].keys():
            generations = secondary_stats['generations'][category]
            unique_users = secondary_stats['unique_users'][category]
            gen_percentage = (generations / analysis_results['total_generations']) * 100
            user_percentage = (unique_users / analysis_results['unique_users']) * 100
            secondary_data.append({
                'Category': category,
                'Generations': generations,
                'Unique Users': unique_users,
                'Gen %': f"{gen_percentage:.1f}%",
                'User %': f"{user_percentage:.1f}%"
            })
        
        secondary_df = pd.DataFrame(secondary_data)
        
        # Combine and save
        comprehensive_data = pd.concat([primary_df, secondary_df], ignore_index=True)
        comprehensive_data.to_csv(output_file, index=False)
    
    def _print_report(self, df: pd.DataFrame, analysis_results: Dict, processor: DataProcessor) -> None:
        """
        Print comprehensive report to console
        
        Args:
            df: Processed DataFrame
            analysis_results: Analysis results
            processor: DataProcessor instance
        """
        print("\n" + "="*80)
        print("TEXT CLASSIFICATION AND USER PATTERN ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total TTS Generations: {analysis_results['total_generations']:,}")
        print(f"   Unique Users: {analysis_results['unique_users']:,}")
        print(f"   Average generations per user: {analysis_results['total_generations'] / analysis_results['unique_users']:.1f}")
        
        # Create comprehensive analysis table
        print(f"\n📋 COMPREHENSIVE ANALYSIS TABLE:")
        self._print_analysis_table(df, analysis_results)
        
        # Bias flags
        if analysis_results['bias_flags']:
            print(f"\n⚠️  POTENTIAL BIAS FLAGS (>{analysis_results['bias_flags'][0]['percentage']*100:.0f}% threshold):")
            for flag in analysis_results['bias_flags']:
                print(f"   {flag['category_type']:8} | {flag['category']:20} | {flag['user']:30} | {flag['count']:3} generations ({flag['percentage']*100:5.1f}%)")
        else:
            print(f"\n✅ No significant bias detected (threshold: {analysis_results['bias_flags'][0]['percentage']*100:.0f}%)")
        
        # Sample classified rows
        print(f"\n📝 SAMPLE CLASSIFIED ROWS:")
        sample_rows = df.sample(min(5, len(df)))
        for _, row in sample_rows.iterrows():
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"   Email: {row['email']}")
            print(f"   Text: {text_preview}")
            print(f"   Primary: {row['primary_category']} | Secondary: {row['secondary_category']}")
            print(f"   Voice: {row['voice']}")
            print()
        
        # Statistics about classification methods
        print(f"\n📊 CLASSIFICATION METHOD STATISTICS:")
        print(f"   Total rows processed: {len(df):,}")
        print(f"   Rows processed with Gemini LLM: {processor.llm_processed_count:,}")
        print(f"   Rows processed with keyword classification: {processor.fallback_processed_count:,}")
        print(f"   Percentage of rows processed with Gemini LLM: {processor.llm_processed_count / len(df) * 100:.1f}%")
        print(f"   Percentage of rows processed with keyword classification: {processor.fallback_processed_count / len(df) * 100:.1f}%")
        
        print("="*80)
        print("Report generation complete!")
        print("="*80)
    
    def _print_analysis_table(self, df: pd.DataFrame, analysis_results: Dict) -> None:
        """
        Print analysis in table format
        
        Args:
            df: Processed DataFrame
            analysis_results: Analysis results
        """
        # Create primary category table
        primary_stats = analysis_results['primary_category_stats']
        primary_data = []
        for category in primary_stats['generations'].keys():
            generations = primary_stats['generations'][category]
            unique_users = primary_stats['unique_users'][category]
            gen_percentage = (generations / analysis_results['total_generations']) * 100
            user_percentage = (unique_users / analysis_results['unique_users']) * 100
            primary_data.append({
                'Category': category,
                'Generations': generations,
                'Unique Users': unique_users,
                'Gen %': f"{gen_percentage:.1f}%",
                'User %': f"{user_percentage:.1f}%"
            })
        
        primary_df = pd.DataFrame(primary_data)
        
        # Create secondary category table
        secondary_stats = analysis_results['secondary_category_stats']
        secondary_data = []
        for category in secondary_stats['generations'].keys():
            generations = secondary_stats['generations'][category]
            unique_users = secondary_stats['unique_users'][category]
            gen_percentage = (generations / analysis_results['total_generations']) * 100
            user_percentage = (unique_users / analysis_results['unique_users']) * 100
            secondary_data.append({
                'Category': category,
                'Generations': generations,
                'Unique Users': unique_users,
                'Gen %': f"{gen_percentage:.1f}%",
                'User %': f"{user_percentage:.1f}%"
            })
        
        secondary_df = pd.DataFrame(secondary_data)
        
        # Print tables
        print(f"\n🎯 PRIMARY CATEGORIES:")
        print(primary_df.to_string(index=False, justify='left'))
        
        print(f"\n📚 SECONDARY CATEGORIES:")
        print(secondary_df.to_string(index=False, justify='left'))
        
        # Print summary insights
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Most popular primary category: {primary_df.loc[primary_df['Generations'].idxmax(), 'Category']} ({primary_df.loc[primary_df['Generations'].idxmax(), 'Gen %']})")
        print(f"   • Most users engaged with: {primary_df.loc[primary_df['Unique Users'].idxmax(), 'Category']} ({primary_df.loc[primary_df['Unique Users'].idxmax(), 'User %']} of users)")
        print(f"   • Most diverse secondary category: {secondary_df.loc[secondary_df['Unique Users'].idxmax(), 'Category']} ({secondary_df.loc[secondary_df['Unique Users'].idxmax(), 'User %']} of users)")
        print(f"   • Least popular secondary category: {secondary_df.loc[secondary_df['Generations'].idxmin(), 'Category']} ({secondary_df.loc[secondary_df['Generations'].idxmin(), 'Gen %']})")


def main():
    """
    Main function to run the text classification and analysis pipeline
    """
    # Configuration
    input_file = "sample_data.csv"  # Use sample_data.csv file in current directory
    chunk_size = 10  # Smaller chunks to process all with Gemini
    bias_threshold = 0.2  # 20%
    
    # Gemini API Configuration
    gemini_api_key = os.getenv('GEMINI_API_KEY')  # Get from environment variable
    use_llm = True  # Set to False to use keyword classification instead
    
    try:
        # Initialize components
        processor = DataProcessor(chunk_size=chunk_size)
        analyzer = UserPatternAnalyzer(bias_threshold=bias_threshold)
        report_generator = ReportGenerator()
        
        # Initialize classifier with LLM
        if use_llm and gemini_api_key:
            processor.classifier = TextClassifier(gemini_api_key=gemini_api_key, use_llm=True)
            logger.info("Using Gemini LLM for classification (flash-lite model, 30 req/min)")
        else:
            processor.classifier = TextClassifier(use_llm=False)
            logger.info("Using keyword-based classification")
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = processor.load_data(input_file)
        
        # Process data in chunks
        logger.info("Starting data processing...")
        processed_df = processor.process_data_in_chunks(df)
        
        # Analyze user patterns
        logger.info("Starting user pattern analysis...")
        analysis_results = analyzer.analyze_user_patterns(processed_df)
        
        # Generate report
        logger.info("Generating final report...")
        report_generator.generate_report(processed_df, analysis_results, processor)
        
        logger.info("Pipeline completed successfully!")
        
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found. Please provide a valid CSV or Excel file.")
        print(f"\nTo use this script:")
        print(f"1. Place your data file in the same directory as this script")
        print(f"2. Update the 'input_file' variable in main() to match your filename")
        print(f"3. Ensure your file has columns: email, text, voice")
        print(f"4. Run the script again")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main() 