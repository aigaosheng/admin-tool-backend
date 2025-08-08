I'll create a CV parser that can extract resume metadata using both Ollama and Gemini APIs with LangChain. This will be a comprehensive solution that can parse various resume formats and extract structured information.I've created a comprehensive CV parser that can extract resume metadata using both Ollama and Gemini APIs with LangChain. Here's what the solution includes:

## Key Features:

1. **Multiple Model Support**: Works with both Ollama and Gemini APIs
2. **File Format Support**: Handles PDF, DOCX, and TXT files
3. **Structured Data Extraction**: Extracts personal info, skills, experience, education, projects, certifications, and more
4. **JSON Output**: Saves extracted data in structured JSON format
5. **Error Handling**: Robust error handling for file parsing and API calls

## Installation Requirements:

```bash
pip install langchain langchain-google-genai PyPDF2 python-docx pydantic
```

## Setup Instructions:

### For Ollama:
1. Install Ollama from https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Make sure Ollama is running

### For Gemini:
1. Get a Google API key from Google AI Studio
2. Set environment variable: `export GOOGLE_API_KEY="your-api-key"`

## Usage Examples:

```python
# Using Ollama
parser = CVParser(model_provider="ollama", model_name="llama2")

# Using Gemini
parser = CVParser(model_provider="gemini")

# Parse resume
metadata = parser.parse_resume("resume.pdf")

# Display summary
parser.print_resume_summary(metadata)

# Save to JSON
parser.save_to_json(metadata, "output.json")
```

## Data Structure:

The parser extracts information into structured dataclasses:
- **PersonalInfo**: Name, email, phone, LinkedIn, GitHub, etc.
- **Education**: Degree, institution, graduation year, GPA
- **Experience**: Job title, company, duration, responsibilities
- **Projects**: Name, description, technologies used
- **Skills, Certifications, Languages, Achievements**: Lists of relevant items

## Key Benefits:

1. **Flexible**: Switch between different LLM providers easily
2. **Structured Output**: Consistent JSON format for easy integration
3. **Extensible**: Easy to add new extraction fields or file formats
4. **Production Ready**: Includes error handling and validation

The parser uses a carefully crafted prompt to ensure consistent JSON output and handles various resume formats and structures. You can customize the extraction fields or add additional processing logic as needed for your specific use case.