# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create directory structure: src/, tests/, and root files
  - Initialize pyproject.toml with Python dependencies (langchain-openai, pydantic, etc.)
  - Create README.md with basic project description and usage
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2. Implement core data models and validation
  - Create schemas.py with QAResult Pydantic model
  - Implement field validation for URLs, prices, and boolean flags
  - Write unit tests for schema validation with valid and invalid data
  - _Requirements: 2.2, 2.3, 6.2, 6.3_

- [x] 3. Create prompt templates and configuration
  - Implement prompts.py with system and user prompt templates
  - Define SUMMARY_SYSTEM prompt with 180-220 character rules and formatting requirements
  - Define QA_SYSTEM prompt with validation rules for price, URL, length, keywords, and duplicates
  - Write tests to verify prompt template rendering with sample data
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.6_

- [x] 4. Build LangChain pipeline integration
  - Create pipeline.py with OpenAI model initialization
  - Implement summary chain using ChatPromptTemplate and StrOutputParser
  - Implement QA chain using structured output with Pydantic model binding
  - Write analyze_deal function to orchestrate both chains
  - _Requirements: 1.5, 2.7, 6.1, 6.4_

- [x] 5. Implement CLI interface and input handling
  - Create cli.py with argument parsing for text and CSV inputs
  - Implement CSV file processing with 'text' column extraction
  - Add support for custom keyword specification with defaults
  - Write input validation and error handling for missing files or invalid formats
  - _Requirements: 3.1, 3.2, 3.5, 4.1, 4.2_

- [x] 6. Build report generation and output formatting
  - Implement Markdown report generation in cli.py
  - Create structured output with summary and QA sections for each processed item
  - Add JSON formatting with proper indentation for QA results
  - Implement file output with configurable output paths
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Add keyword validation and duplicate detection logic
  - Implement case-insensitive keyword matching in QA validation
  - Create duplicate detection algorithm using price+product pattern matching
  - Add missing keyword tracking and reporting
  - Write unit tests for keyword validation and duplicate detection scenarios
  - _Requirements: 2.5, 2.6, 4.3, 4.4_

- [x] 8. Implement comprehensive error handling
  - Add try-catch blocks for LLM API failures with appropriate error messages
  - Implement input validation with clear error messages for missing or invalid data
  - Add file system error handling for permissions and path issues
  - Create graceful degradation for structured output parsing failures
  - _Requirements: 6.4, 6.5_

- [x] 9. Create comprehensive test suite
  - Write unit tests for all schema models with edge cases
  - Create integration tests for LangChain pipeline with mocked LLM responses
  - Implement CLI testing with various argument combinations
  - Add end-to-end tests with sample CSV files and expected outputs
  - _Requirements: 6.2, 6.3_

- [x] 10. Add batch processing optimization and validation
  - Implement efficient CSV processing for large files
  - Add progress indication for batch operations
  - Create validation for required CSV column structure
  - Write performance tests with large datasets
  - _Requirements: 3.3, 3.4_

- [x] 11. Finalize project configuration and documentation
  - Complete pyproject.toml with all dependencies and project metadata
  - Update README.md with installation instructions, usage examples, and API documentation
  - Add example CSV files and sample outputs for demonstration
  - Create requirements validation against all acceptance criteria
  - _Requirements: 5.5, 3.5_