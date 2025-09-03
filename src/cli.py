"""
Command-line interface for the Deal Summary & QA Bot.

This module provides CLI functionality for processing deal content from various
input sources (direct text, CSV files) and generating structured reports.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .pipeline import DealAnalysisPipeline, PipelineError, LLMAPIError, StructuredOutputError


class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass


class InputValidationError(CLIError):
    """Exception for input validation failures."""
    pass


class FileSystemError(CLIError):
    """Exception for file system related errors."""
    pass


class DealSummaryQACLI:
    """
    Command-line interface for the Deal Summary & QA Bot.
    
    Handles argument parsing, input validation, batch processing,
    and report generation.
    """
    
    def __init__(self):
        """Initialize the CLI with default configuration."""
        self.pipeline = None
        self.default_keywords = ["deal", "%", "euro"]
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure the argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="Process deal content to generate summaries and QA reports",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process single text input with OpenRouter (default)
  python -m src.cli --text "Amazing deal on laptops! 50% off at TechStore..." --api_key YOUR_OPENROUTER_KEY
  
  # Process CSV file with OpenRouter (free tier available)
  python -m src.cli --input_csv deals.csv --api_key YOUR_OPENROUTER_KEY
  
  # Process with custom OpenRouter model (free tier)
  python -m src.cli --text "Deal text..." --model "meta-llama/llama-3.2-3b-instruct:free" --api_key YOUR_KEY
  
  # Process with OpenAI directly
  python -m src.cli --text "Deal text..." --provider openai --model gpt-4o-mini --api_key YOUR_OPENAI_KEY
  
  # Process CSV with custom keywords and output
  python -m src.cli --input_csv deals.csv --keywords deal discount euro --out report.md --api_key YOUR_KEY
            """
        )
        
        # Input options (mutually exclusive)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            "--text",
            type=str,
            help="Direct text input for processing a single deal"
        )
        input_group.add_argument(
            "--input_csv",
            type=str,
            help="Path to CSV file containing deals (must have 'text' column)"
        )
        
        # Configuration options
        parser.add_argument(
            "--keywords",
            nargs="+",
            default=None,
            help="Required keywords for validation (default: deal %% euro)"
        )
        parser.add_argument(
            "--out",
            type=str,
            default="deal_analysis_report.md",
            help="Output file path for the generated report (default: deal_analysis_report.md)"
        )
        parser.add_argument(
            "--api_key",
            type=str,
            help="API key (can also be set via OPENROUTER_API_KEY or OPENAI_API_KEY environment variable)"
        )
        parser.add_argument(
            "--model",
            type=str,
            default="openai/gpt-4o-mini",
            help="Model to use for processing (default: openai/gpt-4o-mini for OpenRouter)"
        )
        parser.add_argument(
            "--base_url",
            type=str,
            default="https://openrouter.ai/api/v1",
            help="Base URL for API (default: https://openrouter.ai/api/v1 for OpenRouter)"
        )
        parser.add_argument(
            "--provider",
            type=str,
            choices=["openai", "openrouter"],
            default="openrouter",
            help="API provider preset (default: openrouter)"
        )
        
        return parser
    
    def validate_csv_file(self, csv_path: str) -> Dict[str, Any]:
        """
        Validate CSV file exists and has required structure with comprehensive analysis.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Dictionary with validation results and file statistics
            
        Raises:
            InputValidationError: If file doesn't exist or lacks required columns
            FileSystemError: If file system access fails
        """
        # Input validation
        if not isinstance(csv_path, str) or not csv_path.strip():
            raise InputValidationError("CSV file path must be a non-empty string")
        
        csv_path = csv_path.strip()
        
        # Check file existence
        if not os.path.exists(csv_path):
            # Provide helpful suggestions
            suggestions = []
            if '/' not in csv_path and '\\' not in csv_path:
                suggestions.append(f"Try using a full path: {os.path.abspath(csv_path)}")
            if not csv_path.endswith('.csv'):
                suggestions.append("Make sure the file has a .csv extension")
            
            error_msg = f"CSV file not found: {csv_path}"
            if suggestions:
                error_msg += f"\nSuggestions:\n- " + "\n- ".join(suggestions)
            
            raise InputValidationError(error_msg)
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(csv_path):
            raise InputValidationError(f"Path is not a file: {csv_path}")
        
        # Check file permissions
        if not os.access(csv_path, os.R_OK):
            raise FileSystemError(f"CSV file is not readable (permission denied): {csv_path}")
        
        validation_results = {
            'file_path': os.path.abspath(csv_path),
            'file_size_bytes': 0,
            'file_size_mb': 0.0,
            'estimated_rows': 0,
            'columns': [],
            'has_text_column': False,
            'sample_text_lengths': [],
            'warnings': [],
            'performance_estimate': 'unknown'
        }
        
        try:
            # Get file size and basic stats
            file_size = os.path.getsize(csv_path)
            validation_results['file_size_bytes'] = file_size
            validation_results['file_size_mb'] = file_size / (1024 * 1024)
            
            if file_size == 0:
                raise InputValidationError(f"CSV file is empty: {csv_path}")
            
            # Enhanced size limits with warnings
            if file_size > 500 * 1024 * 1024:  # 500MB hard limit
                raise InputValidationError(
                    f"CSV file is too large ({validation_results['file_size_mb']:.1f}MB). "
                    f"Maximum size is 500MB for batch processing."
                )
            elif file_size > 100 * 1024 * 1024:  # 100MB warning
                validation_results['warnings'].append(
                    f"Large file detected ({validation_results['file_size_mb']:.1f}MB). "
                    f"Processing may take significant time and memory."
                )
            
            with open(csv_path, 'r', encoding='utf-8') as file:
                try:
                    reader = csv.DictReader(file)
                    fieldnames = reader.fieldnames
                    
                    if not fieldnames:
                        raise InputValidationError("CSV file has no header row")
                    
                    validation_results['columns'] = list(fieldnames)
                    
                    # Check for required 'text' column
                    if 'text' not in fieldnames:
                        available_columns = ', '.join(fieldnames)
                        raise InputValidationError(
                            f"CSV file must contain a 'text' column.\n"
                            f"Available columns: {available_columns}\n"
                            f"Expected format: text,other_column1,other_column2"
                        )
                    
                    validation_results['has_text_column'] = True
                    
                    # Sample first few rows for analysis and count total rows
                    sample_rows = []
                    empty_text_count = 0
                    row_count = 0
                    
                    # For very large files (>10MB), use sampling
                    file_size_mb = validation_results['file_size_mb']
                    use_sampling = file_size_mb > 10
                    
                    for i, row in enumerate(reader):
                        row_count += 1
                        text_content = row.get('text', '').strip()
                        
                        if not text_content:
                            empty_text_count += 1
                        else:
                            validation_results['sample_text_lengths'].append(len(text_content))
                        
                        # Collect sample for analysis (first 10 rows)
                        if i < 10:
                            sample_rows.append(row)
                        
                        # For very large files, estimate after sampling
                        if use_sampling and i >= 1000:  # Sample first 1000 rows for very large files
                            # Estimate total rows based on file size and average row size
                            if row_count > 0:
                                # Calculate average bytes per row from sample
                                avg_text_length = sum(validation_results['sample_text_lengths']) / len(validation_results['sample_text_lengths']) if validation_results['sample_text_lengths'] else 100
                                # Rough estimate: text + CSV overhead (commas, quotes, other columns)
                                estimated_bytes_per_row = avg_text_length + 50  # Add overhead for CSV structure
                                
                                total_file_bytes = validation_results['file_size_bytes']
                                estimated_total_rows = int(total_file_bytes / estimated_bytes_per_row)
                                validation_results['estimated_rows'] = max(estimated_total_rows, row_count)
                            else:
                                validation_results['estimated_rows'] = row_count
                            break
                    else:
                        # Read all rows (small to medium files)
                        validation_results['estimated_rows'] = row_count
                    
                    if validation_results['estimated_rows'] == 0:
                        raise InputValidationError("CSV file contains no data rows (only header)")
                    
                    # Analyze content quality
                    if empty_text_count > 0:
                        empty_percentage = (empty_text_count / min(row_count, 100)) * 100
                        if empty_percentage > 50:
                            validation_results['warnings'].append(
                                f"High percentage of empty text fields ({empty_percentage:.1f}%) detected in sample"
                            )
                        elif empty_percentage > 10:
                            validation_results['warnings'].append(
                                f"Some empty text fields ({empty_percentage:.1f}%) detected in sample"
                            )
                    
                    # Analyze text length distribution
                    if validation_results['sample_text_lengths']:
                        avg_length = sum(validation_results['sample_text_lengths']) / len(validation_results['sample_text_lengths'])
                        max_length = max(validation_results['sample_text_lengths'])
                        min_length = min(validation_results['sample_text_lengths'])
                        
                        if avg_length > 2000:
                            validation_results['warnings'].append(
                                f"Very long text content detected (avg: {avg_length:.0f} chars). "
                                f"This may increase processing time and API costs."
                            )
                        elif avg_length < 50:
                            validation_results['warnings'].append(
                                f"Very short text content detected (avg: {avg_length:.0f} chars). "
                                f"Results may be less meaningful."
                            )
                        
                        if max_length > 10000:
                            validation_results['warnings'].append(
                                f"Extremely long text detected ({max_length} chars). "
                                f"Consider preprocessing to split or truncate content."
                            )
                    
                    # Performance estimation
                    estimated_rows = validation_results['estimated_rows']
                    if estimated_rows < 10:
                        validation_results['performance_estimate'] = 'fast'
                    elif estimated_rows < 100:
                        validation_results['performance_estimate'] = 'moderate'
                    elif estimated_rows < 1000:
                        validation_results['performance_estimate'] = 'slow'
                    else:
                        validation_results['performance_estimate'] = 'very_slow'
                        validation_results['warnings'].append(
                            f"Large dataset ({estimated_rows} estimated rows). "
                            f"Consider processing in smaller batches for better error recovery."
                        )
                        
                except csv.Error as e:
                    raise InputValidationError(f"Invalid CSV format: {e}")
                    
        except UnicodeDecodeError as e:
            raise InputValidationError(
                f"Unable to read CSV file due to encoding issue: {csv_path}\n"
                f"Try saving the file with UTF-8 encoding. Error: {e}"
            )
        except PermissionError:
            raise FileSystemError(f"Permission denied accessing CSV file: {csv_path}")
        except OSError as e:
            raise FileSystemError(f"File system error accessing CSV file: {csv_path}. Error: {e}")
        
        # Print validation summary
        print(f"📋 CSV Validation Summary:")
        print(f"   File: {validation_results['file_path']}")
        print(f"   Size: {validation_results['file_size_mb']:.2f} MB")
        print(f"   Estimated rows: {validation_results['estimated_rows']}")
        print(f"   Columns: {', '.join(validation_results['columns'])}")
        
        if validation_results['sample_text_lengths']:
            avg_length = sum(validation_results['sample_text_lengths']) / len(validation_results['sample_text_lengths'])
            print(f"   Average text length: {avg_length:.0f} characters")
        
        performance_messages = {
            'fast': '🚀 Expected processing time: < 1 minute',
            'moderate': '⏱️  Expected processing time: 1-5 minutes',
            'slow': '🐌 Expected processing time: 5-30 minutes',
            'very_slow': '⏳ Expected processing time: 30+ minutes'
        }
        
        if validation_results['performance_estimate'] in performance_messages:
            print(f"   {performance_messages[validation_results['performance_estimate']]}")
        
        # Print warnings
        for warning in validation_results['warnings']:
            print(f"   ⚠️  {warning}")
        
        return validation_results
    
    def read_csv_deals(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Read deal content from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of dictionaries containing deal data
            
        Raises:
            InputValidationError: If file content is invalid
            FileSystemError: If file reading fails
        """
        deals = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    try:
                        text_content = row.get('text', '').strip()
                        
                        if not text_content:
                            print(f"Warning: Row {row_num} has empty 'text' field, skipping")
                            continue
                        
                        # Validate text content length
                        if len(text_content) > 10000:  # 10KB limit per text
                            print(f"Warning: Row {row_num} text is very long ({len(text_content)} chars), may cause processing issues")
                        
                        deals.append({
                            'text': text_content,
                            'row_number': row_num
                        })
                        
                    except Exception as e:
                        print(f"Warning: Error processing row {row_num}: {e}, skipping")
                        continue
                
                if not deals:
                    raise InputValidationError(
                        "No valid deals found in CSV file. "
                        "Make sure the 'text' column contains non-empty values."
                    )
                    
        except UnicodeDecodeError as e:
            raise InputValidationError(f"Unable to read CSV file due to encoding issue: {e}")
        except PermissionError:
            raise FileSystemError(f"Permission denied reading CSV file: {csv_path}")
        except OSError as e:
            raise FileSystemError(f"File system error reading CSV file: {e}")
        except csv.Error as e:
            raise InputValidationError(f"CSV parsing error: {e}")
        except Exception as e:
            if isinstance(e, (InputValidationError, FileSystemError)):
                raise
            raise FileSystemError(f"Unexpected error reading CSV file: {e}")
        
        return deals
    
    def validate_output_path(self, output_path: str) -> None:
        """
        Validate output file path is writable.
        
        Args:
            output_path: Path where report will be written
            
        Raises:
            InputValidationError: If path format is invalid
            FileSystemError: If path is not writable
        """
        # Input validation
        if not isinstance(output_path, str) or not output_path.strip():
            raise InputValidationError("Output path must be a non-empty string")
        
        output_path = output_path.strip()
        
        # Validate file extension
        if not output_path.endswith('.md'):
            print(f"Warning: Output file doesn't have .md extension: {output_path}")
        
        try:
            output_path_abs = os.path.abspath(output_path)
            output_dir = os.path.dirname(output_path_abs)
            
            # Check if directory exists and is writable
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Created output directory: {output_dir}")
                except OSError as e:
                    raise FileSystemError(
                        f"Cannot create output directory: {output_dir}\n"
                        f"Error: {e}\n"
                        f"Make sure you have write permissions to the parent directory."
                    )
            
            # Check directory write permissions
            if not os.access(output_dir, os.W_OK):
                raise FileSystemError(
                    f"Output directory is not writable: {output_dir}\n"
                    f"Check directory permissions or try a different location."
                )
            
            # Check if file exists and is writable
            if os.path.exists(output_path_abs):
                if not os.access(output_path_abs, os.W_OK):
                    raise FileSystemError(
                        f"Output file exists but is not writable: {output_path_abs}\n"
                        f"Check file permissions or choose a different filename."
                    )
                else:
                    print(f"Warning: Output file exists and will be overwritten: {output_path_abs}")
            
            # Check available disk space (warn if less than 10MB)
            try:
                stat = os.statvfs(output_dir)
                free_space = stat.f_bavail * stat.f_frsize
                if free_space < 10 * 1024 * 1024:  # 10MB
                    print(f"Warning: Low disk space ({free_space / (1024*1024):.1f}MB available)")
            except (AttributeError, OSError):
                # statvfs not available on Windows, skip disk space check
                pass
                
        except OSError as e:
            raise FileSystemError(f"File system error validating output path: {e}")
    
    def process_single_text(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Process a single text input.
        
        Args:
            text: Deal text to process
            keywords: Required keywords for validation
            
        Returns:
            Processing results dictionary
            
        Raises:
            InputValidationError: If input is invalid
            LLMAPIError: If LLM API calls fail
            PipelineError: If processing fails
        """
        # Input validation
        if not isinstance(text, str):
            raise InputValidationError("Text input must be a string")
        if not text.strip():
            raise InputValidationError("Text input cannot be empty")
        
        text = text.strip()
        
        # Validate text length
        if len(text) > 10000:  # 10KB limit
            raise InputValidationError(f"Text input is too long ({len(text)} chars). Maximum is 10,000 characters.")
        
        if len(text) < 10:
            print("Warning: Text input is very short, may not produce meaningful results")
        
        try:
            result = self.pipeline.analyze_deal(text, keywords)
            return {
                'input': text,
                'summary': result['summary'],
                'qa': result['qa']
            }
        except (PipelineError, LLMAPIError, StructuredOutputError) as e:
            # Re-raise pipeline errors as-is
            raise
        except Exception as e:
            raise PipelineError(f"Unexpected error processing text: {e}")
    
    def process_csv_batch(self, csv_path: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple deals from CSV file with optimized batch processing.
        
        Args:
            csv_path: Path to CSV file
            keywords: Required keywords for validation
            
        Returns:
            List of processing results
            
        Raises:
            InputValidationError: If CSV file is invalid
            FileSystemError: If file access fails
        """
        import time
        from datetime import datetime, timedelta
        
        deals = self.read_csv_deals(csv_path)
        results = []
        
        print(f"Processing {len(deals)} deals from {csv_path}...")
        
        # Track processing statistics
        success_count = 0
        api_error_count = 0
        other_error_count = 0
        start_time = time.time()
        
        # Progress tracking for large batches
        batch_size = max(1, len(deals) // 20)  # Update progress every 5%
        last_progress_update = 0
        
        for i, deal in enumerate(deals, 1):
            try:
                # Show progress for large batches
                if len(deals) > 50 and (i - last_progress_update) >= batch_size:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(deals) - i) / rate if rate > 0 else 0
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    
                    progress_percent = (i / len(deals)) * 100
                    print(f"Progress: {progress_percent:.1f}% ({i}/{len(deals)}) - "
                          f"Rate: {rate:.1f} deals/sec - ETA: {eta}")
                    last_progress_update = i
                elif len(deals) <= 50:
                    print(f"Processing deal {i}/{len(deals)}...", end=' ')
                
                # Process the deal with existing results for duplicate detection
                result = self.pipeline.analyze_deal(deal['text'], keywords, results)
                
                results.append({
                    'input': deal['text'],
                    'row_number': deal['row_number'],
                    'summary': result['summary'],
                    'qa': result['qa']
                })
                
                success_count += 1
                if len(deals) <= 50:
                    print("✓")
                
                # Memory management for very large batches
                if len(results) % 1000 == 0:
                    print(f"Processed {len(results)} deals, continuing...")
                
            except LLMAPIError as e:
                api_error_count += 1
                error_msg = f"API Error: {str(e)}"
                if len(deals) <= 50:
                    print(f"✗ ({error_msg})")
                
                results.append({
                    'input': deal['text'],
                    'row_number': deal['row_number'],
                    'error': error_msg,
                    'error_type': 'api'
                })
                
                # Enhanced error handling for batch processing
                if api_error_count >= 5 and success_count == 0:
                    print(f"\n⚠️  Multiple consecutive API errors detected. This might indicate:")
                    print("   - Invalid API key or insufficient credits")
                    print("   - Network connectivity issues")
                    print("   - API service outage")
                    print("   Consider checking your API configuration and trying again later.")
                    
                    # For large batches, offer to continue or abort
                    if len(deals) > 100:
                        print("   For large batches, you may want to:")
                        print("   - Check your API status and try again")
                        print("   - Process in smaller chunks")
                        print("   - Use a different API provider")
                
            except (PipelineError, StructuredOutputError) as e:
                other_error_count += 1
                error_msg = f"Processing Error: {str(e)}"
                if len(deals) <= 50:
                    print(f"✗ ({error_msg})")
                
                results.append({
                    'input': deal['text'],
                    'row_number': deal['row_number'],
                    'error': error_msg,
                    'error_type': 'processing'
                })
                
            except Exception as e:
                other_error_count += 1
                error_msg = f"Unexpected Error: {str(e)}"
                if len(deals) <= 50:
                    print(f"✗ ({error_msg})")
                
                results.append({
                    'input': deal['text'],
                    'row_number': deal['row_number'],
                    'error': error_msg,
                    'error_type': 'unexpected'
                })
        
        # Print comprehensive summary statistics
        total_processed = len(deals)
        elapsed_time = time.time() - start_time
        avg_rate = total_processed / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nBatch processing complete:")
        print(f"  📊 Total processed: {total_processed} deals")
        print(f"  ⏱️  Processing time: {timedelta(seconds=int(elapsed_time))}")
        print(f"  🚀 Average rate: {avg_rate:.2f} deals/second")
        print(f"  ✅ Successful: {success_count}/{total_processed} ({success_count/total_processed*100:.1f}%)")
        
        if api_error_count > 0:
            print(f"  🔌 API Errors: {api_error_count}/{total_processed} ({api_error_count/total_processed*100:.1f}%)")
        if other_error_count > 0:
            print(f"  ⚠️  Other Errors: {other_error_count}/{total_processed} ({other_error_count/total_processed*100:.1f}%)")
        
        # Performance recommendations for future runs
        if total_processed > 100:
            if avg_rate < 0.5:
                print(f"  💡 Performance tip: Processing rate is slow. Consider:")
                print(f"     - Using a faster API provider or model")
                print(f"     - Processing in smaller batches")
                print(f"     - Checking network connectivity")
            elif avg_rate > 2.0:
                print(f"  🎉 Excellent processing rate! Your setup is well optimized.")
        
        return results
    
    def generate_markdown_report(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate Markdown report from processing results.
        
        Args:
            results: List of processing results
            output_path: Path to write the report
            
        Raises:
            FileSystemError: If report writing fails
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                f.write("# Deal Summary & QA Report\n\n")
                f.write(f"Generated on: {timestamp}\n")
                f.write(f"Total items processed: {len(results)}\n\n")
                
                # Write summary statistics
                successful = [r for r in results if 'error' not in r]
                failed = [r for r in results if 'error' in r]
                
                f.write("## Processing Summary\n\n")
                f.write(f"- ✅ **Successful:** {len(successful)}\n")
                f.write(f"- ❌ **Failed:** {len(failed)}\n")
                
                if failed:
                    # Group errors by type
                    error_types = {}
                    for result in failed:
                        error_type = result.get('error_type', 'unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    f.write("\n**Error Breakdown:**\n")
                    for error_type, count in error_types.items():
                        f.write(f"- {error_type.title()}: {count}\n")
                
                f.write("\n---\n\n")
                
                # Write individual results
                for i, result in enumerate(results, 1):
                    f.write(f"## Deal {i}")
                    
                    # Add row number for CSV processing
                    if 'row_number' in result:
                        f.write(f" (CSV Row {result['row_number']})")
                    
                    f.write("\n\n")
                    
                    # Handle processing errors
                    if 'error' in result:
                        error_type = result.get('error_type', 'unknown')
                        f.write(f"**Status:** ❌ Processing Failed ({error_type})\n\n")
                        f.write(f"**Error:** {result['error']}\n\n")
                        f.write("**Original Text:**\n")
                        f.write(f"```\n{result['input'][:1000]}{'...' if len(result['input']) > 1000 else ''}\n```\n\n")
                        continue
                    
                    # Original text (truncate if very long)
                    f.write("**Original Text:**\n")
                    input_text = result['input']
                    if len(input_text) > 1000:
                        f.write(f"```\n{input_text[:1000]}...\n```\n")
                        f.write(f"*Text truncated for display (full length: {len(input_text)} characters)*\n\n")
                    else:
                        f.write(f"```\n{input_text}\n```\n\n")
                    
                    # Summary
                    f.write("**Generated Summary:**\n")
                    f.write(f"> {result['summary']}\n\n")
                    f.write(f"*Length: {len(result['summary'])} characters*\n\n")
                    
                    # QA Results
                    f.write("**Quality Assurance Results:**\n")
                    f.write("```json\n")
                    
                    try:
                        # Convert QAResult to dict for JSON serialization
                        qa_dict = result['qa'].model_dump() if hasattr(result['qa'], 'model_dump') else result['qa']
                        f.write(json.dumps(qa_dict, indent=2, default=str))
                    except Exception as e:
                        f.write(f'{{"error": "Failed to serialize QA results: {str(e)}"}}')
                    
                    f.write("\n```\n\n")
                    f.write("---\n\n")
                
        except PermissionError:
            raise FileSystemError(f"Permission denied writing report to: {output_path}")
        except OSError as e:
            if "No space left on device" in str(e):
                raise FileSystemError(f"Insufficient disk space to write report: {output_path}")
            else:
                raise FileSystemError(f"File system error writing report: {e}")
        except UnicodeEncodeError as e:
            raise FileSystemError(f"Text encoding error writing report: {e}")
        except Exception as e:
            raise FileSystemError(f"Unexpected error writing report: {e}")
    
    def run(self, args: Optional[List[str]] = None) -> None:
        """
        Main CLI execution method.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
        """
        parser = self.create_parser()
        
        try:
            parsed_args = parser.parse_args(args)
        except SystemExit as e:
            # argparse calls sys.exit on error, catch it to provide better error handling
            if e.code != 0:
                print("❌ Invalid command line arguments. Use --help for usage information.", file=sys.stderr)
            raise
        
        try:
            # Handle provider presets
            base_url = parsed_args.base_url
            model_name = parsed_args.model
            
            if parsed_args.provider == "openrouter":
                # OpenRouter is now the default
                if not base_url:
                    base_url = "https://openrouter.ai/api/v1"
                # Ensure model is in OpenRouter format
                if model_name == "gpt-4o-mini":
                    model_name = "openai/gpt-4o-mini"
                elif not "/" in model_name and model_name.startswith("gpt"):
                    model_name = f"openai/{model_name}"
                print("Using OpenRouter API...")
            elif parsed_args.provider == "openai":
                # Override defaults for direct OpenAI usage
                base_url = None  # Use OpenAI's default
                if model_name.startswith("openai/"):
                    model_name = model_name.replace("openai/", "")
                print("Using OpenAI API...")
            
            # Initialize pipeline with comprehensive error handling
            print("Initializing Deal Summary & QA Bot...")
            try:
                self.pipeline = DealAnalysisPipeline(
                    api_key=parsed_args.api_key,
                    model_name=model_name,
                    base_url=base_url
                )
            except PipelineError as e:
                print(f"❌ Pipeline initialization failed: {e}", file=sys.stderr)
                if "API key" in str(e):
                    print("💡 Get your API key from:", file=sys.stderr)
                    print("   - OpenRouter (recommended): https://openrouter.ai/keys", file=sys.stderr)
                    print("   - OpenAI: https://platform.openai.com/api-keys", file=sys.stderr)
                sys.exit(1)
            
            # Validate output path early
            try:
                self.validate_output_path(parsed_args.out)
            except (InputValidationError, FileSystemError) as e:
                print(f"❌ Output path validation failed: {e}", file=sys.stderr)
                sys.exit(1)
            
            # Set default keywords if none provided
            keywords = parsed_args.keywords if parsed_args.keywords else self.default_keywords
            
            # Process input based on type
            results = []
            try:
                if parsed_args.text:
                    print("Processing single text input...")
                    results = [self.process_single_text(parsed_args.text, keywords)]
                    
                elif parsed_args.input_csv:
                    print("Validating CSV file...")
                    validation_results = self.validate_csv_file(parsed_args.input_csv)
                    
                    # Show processing confirmation for large files
                    if validation_results['performance_estimate'] in ['slow', 'very_slow']:
                        estimated_time = {
                            'slow': '5-30 minutes',
                            'very_slow': '30+ minutes'
                        }[validation_results['performance_estimate']]
                        
                        print(f"\n⏳ Large dataset detected. Estimated processing time: {estimated_time}")
                        print("💡 Tips for large datasets:")
                        print("   - Ensure stable internet connection")
                        print("   - Monitor API rate limits and costs")
                        print("   - Consider processing during off-peak hours")
                        print("   - Keep the terminal open during processing")
                        print()
                    
                    results = self.process_csv_batch(parsed_args.input_csv, keywords)
                
                else:
                    # This shouldn't happen due to mutually_exclusive_group(required=True)
                    raise CLIError("No input method specified")
                    
            except (InputValidationError, FileSystemError) as e:
                print(f"❌ Input processing failed: {e}", file=sys.stderr)
                sys.exit(1)
            except LLMAPIError as e:
                print(f"❌ API error during processing: {e}", file=sys.stderr)
                print("💡 Possible solutions:", file=sys.stderr)
                print("   - Check your API key and account credits", file=sys.stderr)
                print("   - Verify network connectivity", file=sys.stderr)
                print("   - Try a different model (OpenRouter has free tier models)", file=sys.stderr)
                print("   - Try again later if the service is experiencing issues", file=sys.stderr)
                sys.exit(1)
            
            # Generate report
            try:
                print(f"Generating report: {parsed_args.out}")
                self.generate_markdown_report(results, parsed_args.out)
            except FileSystemError as e:
                print(f"❌ Report generation failed: {e}", file=sys.stderr)
                sys.exit(1)
            
            # Success message with statistics
            successful_results = [r for r in results if 'error' not in r]
            failed_results = [r for r in results if 'error' in r]
            
            print(f"\n✅ Report generated successfully!")
            print(f"📄 Output file: {os.path.abspath(parsed_args.out)}")
            print(f"📊 Processed: {len(successful_results)} successful, {len(failed_results)} failed")
            
            if failed_results:
                print(f"⚠️  {len(failed_results)} deals failed processing - check report for details")
                
                # Provide specific guidance based on error types
                error_types = set(r.get('error_type', 'unknown') for r in failed_results)
                if 'api' in error_types:
                    print("💡 API errors detected - consider checking your API configuration")
                if 'processing' in error_types:
                    print("💡 Processing errors detected - some input text may be malformed")
            
            # Exit with appropriate code
            sys.exit(0 if not failed_results else 1)
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user", file=sys.stderr)
            sys.exit(130)  # Standard exit code for SIGINT
        except (CLIError, PipelineError, LLMAPIError, StructuredOutputError) as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}", file=sys.stderr)
            print("💡 This is likely a bug. Please report it with the full error message.", file=sys.stderr)
            sys.exit(1)


def main():
    """Entry point for the CLI application."""
    cli = DealSummaryQACLI()
    cli.run()


if __name__ == "__main__":
    main()