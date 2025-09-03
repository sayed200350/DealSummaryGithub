# Requirements Document

## Introduction

The Deal Summary & QA Bot is a Python-based application that processes raw deal/article text and produces social-media-ready summaries along with structured quality assurance reports. The system validates content against specific criteria including price presence, source URLs, length limits, required keywords, and duplicate detection. Built with Python, LangChain, and OpenAI's structured outputs, it provides a robust pipeline for content processing and validation.

## Requirements

### Requirement 1

**User Story:** As a content manager, I want to input raw deal text and receive a concise social-media summary, so that I can quickly publish engaging content without manual editing.

#### Acceptance Criteria

1. WHEN raw deal text is provided THEN the system SHALL generate a summary between 180-220 characters
2. WHEN generating a summary THEN the system SHALL include the brand or product name
3. WHEN generating a summary THEN the system SHALL include exactly one call-to-action
4. WHEN generating a summary THEN the system SHALL exclude emojis from the output
5. WHEN the summary is generated THEN the system SHALL use an engaging, concise copywriting tone

### Requirement 2

**User Story:** As a quality assurance specialist, I want to validate deal content against predefined rules, so that I can ensure all published content meets our standards.

#### Acceptance Criteria

1. WHEN validating content THEN the system SHALL check for price presence (EUR or identifiable number)
2. WHEN a price is found THEN the system SHALL extract the numeric price value
3. WHEN validating content THEN the system SHALL verify the presence of a source URL (http/https)
4. WHEN validating content THEN the system SHALL check if content exceeds 200 words
5. WHEN validating content THEN the system SHALL verify presence of required keywords (case-insensitive)
6. WHEN validating content THEN the system SHALL detect potential duplicates using price+product pattern matching
7. WHEN validation is complete THEN the system SHALL return results in strict JSON schema format

### Requirement 3

**User Story:** As a developer, I want to process multiple deals from CSV files or single text inputs, so that I can handle both batch and individual processing workflows.

#### Acceptance Criteria

1. WHEN provided with a CSV file THEN the system SHALL process all rows with a 'text' column
2. WHEN provided with direct text input THEN the system SHALL process the single text string
3. WHEN processing multiple items THEN the system SHALL generate individual reports for each item
4. WHEN processing is complete THEN the system SHALL output a consolidated Markdown report
5. IF no input method is specified THEN the system SHALL display an error message

### Requirement 4

**User Story:** As a system administrator, I want configurable keyword validation, so that I can adapt the QA rules to different content types and requirements.

#### Acceptance Criteria

1. WHEN running the system THEN the system SHALL accept custom required keywords as parameters
2. WHEN no keywords are specified THEN the system SHALL use default keywords: "deal", "%", "euro"
3. WHEN validating keywords THEN the system SHALL perform case-insensitive matching
4. WHEN keywords are missing THEN the system SHALL list all missing keywords in the QA report

### Requirement 5

**User Story:** As a content team member, I want structured output reports, so that I can easily review and act on validation results.

#### Acceptance Criteria

1. WHEN processing is complete THEN the system SHALL generate a Markdown report file
2. WHEN generating reports THEN the system SHALL include both summary and QA sections for each item
3. WHEN displaying QA results THEN the system SHALL format JSON output with proper indentation
4. WHEN the report is generated THEN the system SHALL save it to a specified output file
5. WHEN the process completes THEN the system SHALL confirm the output file location

### Requirement 6

**User Story:** As a developer, I want reliable structured data extraction, so that downstream systems can programmatically process the QA results.

#### Acceptance Criteria

1. WHEN using OpenAI structured outputs THEN the system SHALL enforce strict JSON schema conformance
2. WHEN generating QA results THEN the system SHALL use Pydantic models for data validation
3. WHEN LangChain processes requests THEN the system SHALL maintain consistent output formatting
4. WHEN structured output fails THEN the system SHALL provide clear error messages
5. WHEN the system processes requests THEN the system SHALL use appropriate temperature settings for consistency