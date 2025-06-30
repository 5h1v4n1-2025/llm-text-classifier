# Daruma - Text-to-Speech Application

A Flask-based web application that converts text to speech using Google Gemini AI for content analysis and ElevenLabs for voice synthesis.

## Features

- **Intelligent Content Analysis**: Uses Google Gemini AI to analyze text content and identify characters
- **Multi-Voice Generation**: Automatically assigns appropriate voices to different characters
- **Script Generation**: Creates natural dialogue and narration from text input
- **Audio Synthesis**: Generates high-quality speech using ElevenLabs API
- **Web Interface**: Simple and intuitive web interface for text input

## Text Classifier Component

The project includes an advanced text classification system that can:

- **Primary Categories**: entertainment, learning/productivity, online_content
- **Secondary Categories**: storybook, textbook_or_pdf, newsletter_substack, reddit_post, other
- **Dual Classification Methods**: 
  - Gemini LLM for intelligent classification
  - Keyword-based fallback for reliability
- **Batch Processing**: Process large datasets efficiently
- **User Pattern Analysis**: Analyze usage patterns and identify biases

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# ElevenLabs API Key  
# Get your API key from: https://elevenlabs.io/speech-synthesis
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Text Classifier Usage

### Quick Start with Sample Data

The text classifier comes with `sample_data.csv` containing 20 diverse text samples for testing:

```bash
# Run the main classifier with sample data
python text_classifier.py

# Or run the test suite
python test_text_classifier.py
```

The sample data includes various content types:
- **Stories**: Fantasy narratives, fairy tales
- **Educational**: Machine learning chapters, academic content
- **Social Media**: Reddit posts, forum discussions
- **Newsletters**: Email digests, updates

### Basic Usage

```python
from text_classifier import TextClassifier

# Initialize with Gemini API key
classifier = TextClassifier(gemini_api_key="your_api_key", use_llm=True)

# Classify text
primary, secondary = classifier.classify_text("Your text here")
print(f"Primary: {primary}, Secondary: {secondary}")
```

### Fallback Mode

```python
# Use keyword-based classification (no API key needed)
classifier = TextClassifier(use_llm=False)
primary, secondary = classifier.classify_text("Your text here")
```

### Batch Processing

```python
from text_classifier import DataProcessor, UserPatternAnalyzer, ReportGenerator

# Initialize components
processor = DataProcessor(chunk_size=100)
analyzer = UserPatternAnalyzer(bias_threshold=0.2)
report_generator = ReportGenerator()

# Load and process data
df = processor.load_data("sample_data.csv")  # or your own CSV file
processed_df = processor.process_data_in_chunks(df)

# Analyze patterns
analysis_results = analyzer.analyze_user_patterns(processed_df)

# Generate reports
report_generator.generate_report(processed_df, analysis_results, processor)
```

### Testing

Run the comprehensive test suite:

```bash
python test_text_classifier.py
```

This will:
- Test individual text classification
- Process the sample_data.csv file
- Generate analysis reports
- Allow interactive testing

## API Endpoints

### POST /generate-audio

Convert text to speech with character analysis.

**Request Body:**
```json
{
  "text": "Your text to convert to speech"
}
```

**Response:** Audio file (MP3)

## Classification Categories

### Primary Categories
- **entertainment**: Stories, novels, fiction, creative content, leisure reading
- **learning/productivity**: Educational content, textbooks, academic papers, how-to guides
- **online_content**: Social media posts, forum discussions, Reddit posts, online discussions

### Secondary Categories
- **storybook**: Fiction stories, novels, creative narratives, fairy tales
- **textbook_or_pdf**: Academic content, technical documentation, research papers
- **newsletter_substack**: Newsletters, email digests, subscription content
- **reddit_post**: Reddit content, forum posts, social media discussions
- **other**: Content that doesn't fit the above categories

## File Structure

```
daruma/
├── app.py                 # Main Flask application
├── text_classifier.py     # Text classification system
├── test_text_classifier.py # Test suite
├── sample_data.csv        # Sample data for testing (20 diverse texts)
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Your API keys (not in git)
├── README.md             # This file
└── templates/
    └── index.html        # Web interface
```

## Sample Data

The `sample_data.csv` file contains 20 diverse text samples covering all classification categories:

- **8 Entertainment texts**: Fantasy stories, fairy tales, magical narratives
- **6 Learning/Productivity texts**: Academic chapters, educational content
- **6 Online Content texts**: Reddit posts, social media discussions
- **Multiple users**: 8 different users with varying content preferences

This provides a perfect starting point for testing the classification system and understanding how it works with different content types.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on GitHub.
