# Deal Summary & QA Bot

A command-line tool for processing deal content to generate social media summaries and perform quality assurance validation. Uses OpenRouter by default for access to free AI models.

## Features

- Generate concise social media summaries (180-220 characters)
- Quality assurance validation for price, URLs, keywords, and duplicates
- Support for single text input or batch CSV processing
- **OpenRouter integration with free tier models** (default)
- OpenAI API compatibility
- Comprehensive error handling and validation
- Markdown report generation

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/example/deal-summary-qa-bot.git
cd deal-summary-qa-bot

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install deal-summary-qa-bot
```

### Requirements

- Python 3.8 or higher
- OpenRouter API key (free tier available) or OpenAI API key

## Quick Start

1. Sign up at [OpenRouter](https://openrouter.ai/) to get your free API key
2. Process your first deal:

```bash
# Process single text (uses OpenRouter by default)
python -m src.cli --text "Amazing deal! 50% off laptops at TechStore" --api_key YOUR_OPENROUTER_KEY

# Process CSV file
python -m src.cli --input_csv deals.csv --api_key YOUR_OPENROUTER_KEY
```

## Usage

### OpenRouter (Default - Free Tier Available)

```bash
# Basic usage (OpenRouter is default)
python -m src.cli --text "Deal content..." --api_key YOUR_OPENROUTER_KEY

# Use a specific free model
python -m src.cli --text "Deal text..." --model "meta-llama/llama-3.2-3b-instruct:free" --api_key YOUR_KEY

# Process CSV with free model
python -m src.cli --input_csv deals.csv --model "meta-llama/llama-3.2-3b-instruct:free" --api_key YOUR_KEY
```

### OpenAI API (Alternative)

```bash
# Use OpenAI instead of OpenRouter
python -m src.cli --text "Amazing deal! 50% off laptops" --provider openai --api_key YOUR_OPENAI_KEY

# Process CSV file with OpenAI
python -m src.cli --input_csv deals.csv --provider openai --api_key YOUR_OPENAI_KEY
```

### CSV File Format

Your CSV file must contain a `text` column with the deal content:

```csv
text,category
"Amazing laptop deal! Get 50% off premium gaming laptops at TechStore.",electronics
"Free shipping on all orders over €25! Shop now at FashionHub.",fashion
```

### Command Line Options

```
usage: cli.py [-h] (--text TEXT | --input_csv INPUT_CSV)
              [--keywords KEYWORDS [KEYWORDS ...]] [--out OUT] 
              [--api_key API_KEY] [--model MODEL] [--base_url BASE_URL]
              [--provider {openai,openrouter}]

options:
  --text TEXT           Direct text input for processing a single deal
  --input_csv INPUT_CSV Path to CSV file containing deals (must have 'text' column)
  --keywords KEYWORDS   Required keywords for validation (default: deal % euro)
  --out OUT             Output file path (default: deal_analysis_report.md)
  --api_key API_KEY     API key (can also be set via OPENROUTER_API_KEY or OPENAI_API_KEY env var)
  --model MODEL         Model to use (default: openai/gpt-4o-mini for OpenRouter)
  --base_url BASE_URL   Base URL for API (default: https://openrouter.ai/api/v1)
  --provider PROVIDER   API provider preset (default: openrouter)
```

## Examples

### Basic Usage
```bash
# Simple text processing (OpenRouter default)
python -m src.cli --text "Great deal on laptops! 50% off at TechStore with free shipping." --api_key sk-or-v1-...

# Process example CSV file
python -m src.cli --input_csv examples/sample_deals.csv --out my_report.md --api_key sk-or-v1-...

# CSV processing with custom keywords
python -m src.cli --input_csv examples/sample_deals.csv --keywords deal discount euro --api_key sk-or-v1-...

# Use free tier model
python -m src.cli --text "Deal content..." --model "meta-llama/llama-3.2-3b-instruct:free" --api_key sk-or-v1-...

# Process single deal from example file
python -m src.cli --text "$(cat examples/single_deal_example.txt)" --api_key sk-or-v1-...
```

### Environment Variable
```bash
# Set OpenRouter API key as environment variable
export OPENROUTER_API_KEY="sk-or-v1-your-openrouter-key"
python -m src.cli --text "Deal content..."

# Or use OPENAI_API_KEY (also supported)
export OPENAI_API_KEY="sk-or-v1-your-openrouter-key"
python -m src.cli --text "Deal content..."
```

## Output

The tool generates a Markdown report containing:

- Original deal text
- Generated social media summary with character count
- Quality assurance results in JSON format
- Processing status and error handling

### Sample Output

```markdown
# Deal Summary & QA Report

## Deal 1

**Original Text:**
```
Amazing laptop deal! Get 50% off premium gaming laptops at TechStore.
```

**Generated Summary:**
> Unlock unbeatable performance with TechStore's 50% off premium gaming laptops! Don't miss out—shop now!

*Length: 180 characters*

**Quality Assurance Results:**
```json
{
  "has_price": false,
  "price_value": null,
  "has_source_url": false,
  "source_url": null,
  "within_length_limit": true,
  "missing_keywords": ["euro"],
  "duplicate_suspect": false,
  "notes": "Content missing price and source URL."
}
```
```

## API Providers

### OpenRouter (Default)
- **Free tier available** - perfect for getting started
- Access to multiple models including Llama, Claude, and GPT
- Sign up at: https://openrouter.ai/
- Popular free models:
  - `meta-llama/llama-3.2-3b-instruct:free`
  - `meta-llama/llama-3.2-1b-instruct:free`
  - `qwen/qwen-2-7b-instruct:free`
- Paid models available for higher quality

### OpenAI (Alternative)
- Requires paid API key
- High-quality GPT models (gpt-4o-mini, gpt-4o, etc.)
- Sign up at: https://platform.openai.com/
- Use `--provider openai` to switch from default OpenRouter

## API Documentation

### Core Classes

#### `DealAnalysisPipeline`

Main pipeline class for processing deal content.

```python
from src.pipeline import DealAnalysisPipeline

# Initialize pipeline
pipeline = DealAnalysisPipeline(
    api_key="your-api-key",
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

# Analyze single deal
result = pipeline.analyze_deal(
    deal_text="Amazing deal! 50% off laptops at TechStore",
    required_keywords=["deal", "%", "euro"]
)
```

#### `QAResult`

Pydantic model for structured QA validation results.

```python
from src.schemas import QAResult

# Example QA result structure
qa_result = QAResult(
    has_price=True,
    price_value=199.99,
    has_source_url=True,
    source_url="https://example.com",
    within_length_limit=True,
    missing_keywords=[],
    duplicate_suspect=False,
    notes="Valid deal content"
)
```

### Error Handling

The application includes comprehensive error handling:

- `PipelineError`: General pipeline processing errors
- `LLMAPIError`: API communication errors with retry logic
- `StructuredOutputError`: JSON parsing and validation errors
- `InputValidationError`: Input format and validation errors
- `FileSystemError`: File I/O related errors

### Configuration

#### Environment Variables

```bash
# OpenRouter (default)
export OPENROUTER_API_KEY="sk-or-v1-your-key"

# OpenAI (alternative)
export OPENAI_API_KEY="sk-your-openai-key"

# Optional: Custom model
export DEFAULT_MODEL="meta-llama/llama-3.2-3b-instruct:free"
```

#### Programmatic Configuration

```python
from src.pipeline import DealAnalysisPipeline

# Custom configuration
pipeline = DealAnalysisPipeline(
    api_key="your-key",
    model="custom-model",
    base_url="https://custom-api.com/v1",
    temperature=0.1,
    max_retries=3,
    timeout=30
)
```

## Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/example/deal-summary-qa-bot.git
cd deal-summary-qa-bot
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_pipeline.py -v

# Run performance tests
pytest tests/test_performance.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
deal-summary-qa-bot/
├── src/
│   ├── __init__.py
│   ├── __main__.py          # Entry point for python -m src
│   ├── cli.py               # Command-line interface
│   ├── pipeline.py          # Core processing pipeline
│   ├── prompts.py           # LLM prompt templates
│   └── schemas.py           # Pydantic data models
├── tests/
│   ├── test_cli.py          # CLI testing
│   ├── test_pipeline.py     # Pipeline testing
│   ├── test_prompts.py      # Prompt testing
│   ├── test_schemas.py      # Schema validation testing
│   ├── test_performance.py  # Performance testing
│   └── test_data/           # Test datasets
├── examples/                # Example files and outputs
│   ├── sample_deals.csv     # Example CSV input
│   ├── single_deal_example.txt # Example single deal
│   ├── sample_output.md     # Example generated report
│   └── README.md            # Examples documentation
├── pyproject.toml          # Project configuration
├── REQUIREMENTS_VALIDATION.md # Requirements compliance report
└── README.md               # This file
```

## Examples

See the `examples/` directory for:
- Sample CSV files with various deal types
- Example single deal text
- Sample generated reports
- Usage examples and documentation

Run the examples:
```bash
# Process sample deals
python -m src.cli --input_csv examples/sample_deals.csv --api_key YOUR_KEY

# Process single example
python -m src.cli --text "$(cat examples/single_deal_example.txt)" --api_key YOUR_KEY
```