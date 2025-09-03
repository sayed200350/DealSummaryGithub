# Examples

This directory contains example files and sample outputs to help you get started with the Deal Summary & QA Bot.

## Files

### Input Examples

- **`sample_deals.csv`** - Example CSV file with 5 different deal types (electronics, fashion, food)
- **`single_deal_example.txt`** - Example of a single deal text for direct processing

### Output Examples

- **`sample_output.md`** - Example of generated Markdown report showing summaries and QA results

## Usage Examples

### Process the sample CSV file

```bash
# Using OpenRouter (default, free tier available)
python -m src.cli --input_csv examples/sample_deals.csv --out examples/my_report.md --api_key YOUR_OPENROUTER_KEY

# Using a free model
python -m src.cli --input_csv examples/sample_deals.csv --model "meta-llama/llama-3.2-3b-instruct:free" --api_key YOUR_KEY
```

### Process a single deal

```bash
# Read from file and process
python -m src.cli --text "$(cat examples/single_deal_example.txt)" --api_key YOUR_OPENROUTER_KEY

# Or process directly
python -m src.cli --text "Amazing deal! 50% off laptops at TechStore for €999.99. Visit https://techstore.com/deals" --api_key YOUR_KEY
```

### Custom keywords

```bash
# Use custom validation keywords
python -m src.cli --input_csv examples/sample_deals.csv --keywords deal discount euro price --api_key YOUR_KEY
```

## Expected Output

When you run the examples, you should see:

1. **Summary Generation**: 180-220 character social media ready summaries
2. **QA Validation**: JSON structured results checking for:
   - Price presence and extraction
   - Source URL validation
   - Length limits (200 words)
   - Required keyword presence
   - Duplicate detection
3. **Markdown Report**: Formatted output with both summaries and QA results

## CSV Format Requirements

Your CSV files must have a `text` column containing the deal content:

```csv
text,category,source
"Deal content here...",electronics,store-name
"Another deal...",fashion,another-store
```

Additional columns (like `category`, `source`) are optional and will be ignored by the processing pipeline.