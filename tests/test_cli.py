"""
Tests for the CLI interface and input handling functionality.

This module tests argument parsing, input validation, CSV processing,
and error handling for the Deal Summary & QA Bot CLI.
"""

import argparse
import csv
import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.cli import DealSummaryQACLI, CLIError, InputValidationError, FileSystemError
from src.pipeline import PipelineError
from src.schemas import QAResult


class TestDealSummaryQACLI(unittest.TestCase):
    """Test cases for the DealSummaryQACLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = DealSummaryQACLI()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_parser(self):
        """Test argument parser creation and configuration."""
        parser = self.cli.create_parser()
        
        # Test parser is created
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Test required mutually exclusive group
        with self.assertRaises(SystemExit):
            parser.parse_args([])  # No arguments should fail
        
        # Test text input
        args = parser.parse_args(["--text", "test deal content"])
        self.assertEqual(args.text, "test deal content")
        self.assertIsNone(args.input_csv)
        
        # Test CSV input
        args = parser.parse_args(["--input_csv", "deals.csv"])
        self.assertEqual(args.input_csv, "deals.csv")
        self.assertIsNone(args.text)
        
        # Test mutually exclusive (should fail)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--text", "test", "--input_csv", "file.csv"])
    
    def test_parser_default_values(self):
        """Test parser default values."""
        parser = self.cli.create_parser()
        args = parser.parse_args(["--text", "test"])
        
        self.assertIsNone(args.keywords)  # Default is None, handled in run()
        self.assertEqual(args.out, "deal_analysis_report.md")
        self.assertEqual(args.model, "openai/gpt-4o-mini")
        self.assertIsNone(args.api_key)
        self.assertEqual(args.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(args.provider, "openrouter")
    
    def test_parser_custom_values(self):
        """Test parser with custom values."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            "--text", "test deal",
            "--keywords", "discount", "sale", "offer",
            "--out", "custom_report.md",
            "--api_key", "test-key"
        ])
        
        self.assertEqual(args.keywords, ["discount", "sale", "offer"])
        self.assertEqual(args.out, "custom_report.md")
        self.assertEqual(args.api_key, "test-key")
    
    def test_validate_csv_file_not_exists(self):
        """Test CSV validation with non-existent file."""
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file("nonexistent.csv")
        
        self.assertIn("CSV file not found", str(context.exception))
    
    def test_validate_csv_file_not_file(self):
        """Test CSV validation with directory instead of file."""
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file(self.temp_dir)
        
        self.assertIn("Path is not a file", str(context.exception))
    
    def test_validate_csv_file_empty(self):
        """Test CSV validation with empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.csv")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file(empty_file)
        
        self.assertIn("CSV file is empty", str(context.exception))
    
    def test_validate_csv_file_missing_text_column(self):
        """Test CSV validation with missing 'text' column."""
        csv_file = os.path.join(self.temp_dir, "no_text_column.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["title", "description"])
            writer.writerow(["Deal 1", "Great deal"])
        
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file(csv_file)
        
        self.assertIn("CSV file must contain a 'text' column", str(context.exception))
        self.assertIn("Available columns: title, description", str(context.exception))
    
    def test_validate_csv_file_no_data_rows(self):
        """Test CSV validation with header only."""
        csv_file = os.path.join(self.temp_dir, "header_only.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
        
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file(csv_file)
        
        self.assertIn("CSV file contains no data rows", str(context.exception))
    
    def test_validate_csv_file_valid(self):
        """Test CSV validation with valid file."""
        csv_file = os.path.join(self.temp_dir, "valid.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "category"])
            writer.writerow(["Great deal on laptops!", "electronics"])
        
        # Should not raise any exception
        self.cli.validate_csv_file(csv_file)
    
    def test_validate_csv_file_encoding_error(self):
        """Test CSV validation with encoding issues."""
        csv_file = os.path.join(self.temp_dir, "bad_encoding.csv")
        # Write binary data that's not valid UTF-8
        with open(csv_file, 'wb') as f:
            f.write(b'\xff\xfe\x00\x00invalid utf-8')
        
        with self.assertRaises(CLIError) as context:
            self.cli.validate_csv_file(csv_file)
        
        self.assertIn("encoding issue", str(context.exception))
    
    def test_read_csv_deals_valid(self):
        """Test reading deals from valid CSV file."""
        csv_file = os.path.join(self.temp_dir, "deals.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "category"])
            writer.writerow(["Great deal on laptops!", "electronics"])
            writer.writerow(["50% off shoes", "fashion"])
        
        deals = self.cli.read_csv_deals(csv_file)
        
        self.assertEqual(len(deals), 2)
        self.assertEqual(deals[0]['text'], "Great deal on laptops!")
        self.assertEqual(deals[0]['row_number'], 2)
        self.assertEqual(deals[1]['text'], "50% off shoes")
        self.assertEqual(deals[1]['row_number'], 3)
    
    def test_read_csv_deals_with_empty_rows(self):
        """Test reading CSV with empty text fields."""
        csv_file = os.path.join(self.temp_dir, "deals_with_empty.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["Valid deal"])
            writer.writerow([""])  # Empty text
            writer.writerow(["   "])  # Whitespace only
            writer.writerow(["Another valid deal"])
        
        with patch('builtins.print') as mock_print:
            deals = self.cli.read_csv_deals(csv_file)
        
        # Should only get valid deals
        self.assertEqual(len(deals), 2)
        self.assertEqual(deals[0]['text'], "Valid deal")
        self.assertEqual(deals[1]['text'], "Another valid deal")
        
        # Should print warnings for empty rows
        mock_print.assert_any_call("Warning: Row 3 has empty 'text' field, skipping")
        mock_print.assert_any_call("Warning: Row 4 has empty 'text' field, skipping")
    
    def test_read_csv_deals_no_valid_deals(self):
        """Test reading CSV with no valid deals."""
        csv_file = os.path.join(self.temp_dir, "no_valid_deals.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow([""])
            writer.writerow(["   "])
        
        with self.assertRaises(CLIError) as context:
            self.cli.read_csv_deals(csv_file)
        
        self.assertIn("No valid deals found in CSV file", str(context.exception))
    
    def test_validate_output_path_valid(self):
        """Test output path validation with valid path."""
        output_path = os.path.join(self.temp_dir, "report.md")
        
        # Should not raise any exception
        self.cli.validate_output_path(output_path)
    
    def test_validate_output_path_create_directory(self):
        """Test output path validation that creates directory."""
        new_dir = os.path.join(self.temp_dir, "new_subdir")
        output_path = os.path.join(new_dir, "report.md")
        
        # Directory doesn't exist yet
        self.assertFalse(os.path.exists(new_dir))
        
        # Should create directory
        self.cli.validate_output_path(output_path)
        
        # Directory should now exist
        self.assertTrue(os.path.exists(new_dir))
    
    def test_validate_output_path_not_writable_file(self):
        """Test output path validation with non-writable existing file."""
        output_path = os.path.join(self.temp_dir, "readonly.md")
        
        # Create file and make it read-only
        with open(output_path, 'w') as f:
            f.write("test")
        os.chmod(output_path, 0o444)  # Read-only
        
        try:
            with self.assertRaises(CLIError) as context:
                self.cli.validate_output_path(output_path)
            
            self.assertIn("Output file exists but is not writable", str(context.exception))
        finally:
            # Restore write permissions for cleanup
            os.chmod(output_path, 0o644)
    
    def test_process_single_text_empty(self):
        """Test processing empty text input."""
        with self.assertRaises(CLIError) as context:
            self.cli.process_single_text("", ["deal"])
        
        self.assertIn("Text input cannot be empty", str(context.exception))
        
        with self.assertRaises(CLIError) as context:
            self.cli.process_single_text("   ", ["deal"])
        
        self.assertIn("Text input cannot be empty", str(context.exception))
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_process_single_text_success(self, mock_pipeline_class):
        """Test successful single text processing."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock QA result
        mock_qa_result = QAResult(
            has_price=True,
            price_value=29.99,
            has_source_url=True,
            source_url="https://example.com",
            within_length_limit=True,
            missing_keywords=[],
            duplicate_suspect=False
        )
        
        mock_pipeline.analyze_deal.return_value = {
            "summary": "Great laptop deal - 50% off premium models!",
            "qa": mock_qa_result
        }
        
        self.cli.pipeline = mock_pipeline
        
        result = self.cli.process_single_text("Great deal on laptops!", ["deal"])
        
        self.assertEqual(result['input'], "Great deal on laptops!")
        self.assertEqual(result['summary'], "Great laptop deal - 50% off premium models!")
        self.assertEqual(result['qa'], mock_qa_result)
        
        mock_pipeline.analyze_deal.assert_called_once_with("Great deal on laptops!", ["deal"])
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_process_single_text_pipeline_error(self, mock_pipeline_class):
        """Test single text processing with pipeline error."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.analyze_deal.side_effect = Exception("API error")
        
        self.cli.pipeline = mock_pipeline
        
        with self.assertRaises(PipelineError) as context:
            self.cli.process_single_text("test deal", ["deal"])
        
        self.assertIn("Unexpected error processing text: API error", str(context.exception))
    
    def test_generate_markdown_report_single_success(self):
        """Test Markdown report generation for single successful result."""
        mock_qa_result = QAResult(
            has_price=True,
            price_value=29.99,
            has_source_url=True,
            source_url="https://example.com",
            within_length_limit=True,
            missing_keywords=[],
            duplicate_suspect=False,
            notes="Valid deal content"
        )
        
        results = [{
            'input': "Great deal on laptops!",
            'summary': "Amazing laptop sale - 50% off!",
            'qa': mock_qa_result
        }]
        
        output_path = os.path.join(self.temp_dir, "test_report.md")
        self.cli.generate_markdown_report(results, output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("# Deal Summary & QA Report", content)
        self.assertIn("Total items processed: 1", content)
        self.assertIn("## Deal 1", content)
        self.assertIn("Great deal on laptops!", content)
        self.assertIn("Amazing laptop sale - 50% off!", content)
        self.assertIn("Length: 30 characters", content)
        self.assertIn('"has_price": true', content)
        self.assertIn('"price_value": 29.99', content)
    
    def test_generate_markdown_report_with_errors(self):
        """Test Markdown report generation with processing errors."""
        results = [
            {
                'input': "Valid deal text",
                'summary': "Great summary",
                'qa': QAResult(
                    has_price=False,
                    has_source_url=False,
                    within_length_limit=True,
                    missing_keywords=["deal"],
                    duplicate_suspect=False
                )
            },
            {
                'input': "Failed deal text",
                'row_number': 3,
                'error': "API timeout"
            }
        ]
        
        output_path = os.path.join(self.temp_dir, "error_report.md")
        self.cli.generate_markdown_report(results, output_path)
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Total items processed: 2", content)
        self.assertIn("## Deal 1", content)
        self.assertIn("## Deal 2 (CSV Row 3)", content)
        self.assertIn("**Status:** ❌ Processing Failed", content)
        self.assertIn("**Error:** API timeout", content)
        self.assertIn("Failed deal text", content)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_cli_text_input_integration(self, mock_pipeline_class):
        """Test complete CLI workflow with text input."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_qa_result = QAResult(
            has_price=True,
            price_value=19.99,
            has_source_url=True,
            source_url="https://shop.example.com",
            within_length_limit=True,
            missing_keywords=[],
            duplicate_suspect=False
        )
        
        mock_pipeline.analyze_deal.return_value = {
            "summary": "Awesome deal alert! 50% off electronics today only - don't miss out on these incredible savings at TechStore! Limited time offer with free shipping included.",
            "qa": mock_qa_result
        }
        
        output_path = os.path.join(self.temp_dir, "integration_report.md")
        
        cli = DealSummaryQACLI()
        
        # Test CLI run
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as context:
                cli.run([
                    "--text", "Great electronics deal with 50% discount!",
                    "--keywords", "deal", "discount",
                    "--out", output_path
                ])
            # Should exit with code 0 (success)
            self.assertEqual(context.exception.code, 0)
        
        # Verify output messages
        output = mock_stdout.getvalue()
        self.assertIn("Initializing Deal Summary & QA Bot", output)
        self.assertIn("Processing single text input", output)
        self.assertIn("Report generated successfully", output)
        self.assertIn("Processed: 1 successful, 0 failed", output)
        
        # Verify report file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify pipeline was called correctly
        mock_pipeline.analyze_deal.assert_called_once_with(
            "Great electronics deal with 50% discount!",
            ["deal", "discount"]
        )
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_cli_csv_input_integration(self, mock_pipeline_class):
        """Test complete CLI workflow with CSV input."""
        # Create test CSV
        csv_file = os.path.join(self.temp_dir, "test_deals.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "category"])
            writer.writerow(["50% off laptops today!", "electronics"])
            writer.writerow(["Free shipping on all orders", "general"])
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            return {
                "summary": f"Summary for: {text[:20]}...",
                "qa": QAResult(
                    has_price="%" in text,
                    price_value=50.0 if "%" in text else None,
                    has_source_url=False,
                    within_length_limit=True,
                    missing_keywords=[],
                    duplicate_suspect=False
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        output_path = os.path.join(self.temp_dir, "csv_report.md")
        
        cli = DealSummaryQACLI()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as context:
                cli.run([
                    "--input_csv", csv_file,
                    "--out", output_path
                ])
            # Should exit with code 0 because both deals are successful
            self.assertEqual(context.exception.code, 0)
        
        # Verify output
        output = mock_stdout.getvalue()
        self.assertIn("Processing 2 deals from", output)
        self.assertIn("Processing deal 1/2", output)
        self.assertIn("Processing deal 2/2", output)
        self.assertIn("Processed: 2 successful, 0 failed", output)
        
        # Verify both deals were processed
        self.assertEqual(mock_pipeline.analyze_deal.call_count, 2)
        
        # Verify report file
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Total items processed: 2", content)
        self.assertIn("50% off laptops today!", content)
        self.assertIn("Free shipping on all orders", content)


class TestEndToEndWorkflows(unittest.TestCase):
    """Comprehensive end-to-end tests with sample CSV files and expected outputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cli = DealSummaryQACLI()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_csv(self, filename: str, data: List[Dict[str, str]]) -> str:
        """Create a sample CSV file for testing."""
        csv_path = os.path.join(self.temp_dir, filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        return csv_path
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_end_to_end_csv_processing_with_mixed_results(self, mock_pipeline_class):
        """Test complete CSV processing workflow with mixed success/failure results."""
        # Create sample CSV with various deal types
        sample_data = [
            {
                "text": "Sony WH-1000XM4 headphones at €199.99 with 50% off! Visit https://store.com/deal",
                "category": "electronics"
            },
            {
                "text": "Free shipping on Nike shoes this weekend!",
                "category": "fashion"
            },
            {
                "text": "MacBook Pro with student discount €1899 at https://edu-store.com",
                "category": "electronics"
            }
        ]
        
        csv_path = self.create_sample_csv("mixed_deals.csv", sample_data)
        output_path = os.path.join(self.temp_dir, "mixed_results.md")
        
        # Mock pipeline with different results for each deal
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            if "Sony" in text:
                return {
                    "summary": "Amazing Sony headphones deal! WH-1000XM4 now 50% off at €199.99 with premium noise cancellation. Limited time offer - get yours today!",
                    "qa": QAResult(
                        has_price=True,
                        price_value=199.99,
                        has_source_url=True,
                        source_url="https://store.com/deal",
                        within_length_limit=True,
                        missing_keywords=[],
                        duplicate_suspect=False,
                        notes="Complete deal with all required elements"
                    )
                }
            elif "Nike" in text:
                return {
                    "summary": "Nike shoes with free shipping this weekend! Comfortable running shoes perfect for athletes and casual wear. Don't miss this limited offer!",
                    "qa": QAResult(
                        has_price=False,
                        price_value=None,
                        has_source_url=False,
                        source_url=None,
                        within_length_limit=True,
                        missing_keywords=["euro", "%"],
                        duplicate_suspect=False,
                        notes="Missing price and URL information"
                    )
                }
            elif "MacBook" in text:
                return {
                    "summary": "MacBook Pro student discount available! M2 chip, 16GB RAM, 512GB SSD for €1899. Perfect for creative professionals and students. Apply now!",
                    "qa": QAResult(
                        has_price=True,
                        price_value=1899.0,
                        has_source_url=True,
                        source_url="https://edu-store.com",
                        within_length_limit=True,
                        missing_keywords=["%"],
                        duplicate_suspect=False,
                        notes="Good deal but missing percentage discount"
                    )
                }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        # Run CLI processing
        with patch('sys.stdout', new_callable=StringIO):
            with self.assertRaises(SystemExit) as context:
                self.cli.run([
                    "--input_csv", csv_path,
                    "--keywords", "deal", "%", "euro",
                    "--out", output_path
                ])
            # Should exit with code 0 (all successful)
            self.assertEqual(context.exception.code, 0)
        
        # Verify report was generated
        self.assertTrue(os.path.exists(output_path))
        
        # Verify report content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check report structure
        self.assertIn("# Deal Summary & QA Report", content)
        self.assertIn("Total items processed: 3", content)
        self.assertIn("✅ **Successful:** 3", content)
        self.assertIn("❌ **Failed:** 0", content)
        
        # Check individual deals
        self.assertIn("## Deal 1", content)
        self.assertIn("## Deal 2", content)
        self.assertIn("## Deal 3", content)
        
        # Check specific content
        self.assertIn("Sony WH-1000XM4", content)
        self.assertIn("Nike shoes", content)
        self.assertIn("MacBook Pro", content)
        
        # Check QA results are included
        self.assertIn('"has_price": true', content)
        self.assertIn('"has_price": false', content)
        self.assertIn('"missing_keywords":', content)
        
        # Verify all deals were processed
        self.assertEqual(mock_pipeline.analyze_deal.call_count, 3)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_end_to_end_with_processing_errors(self, mock_pipeline_class):
        """Test end-to-end processing with some deals failing."""
        sample_data = [
            {"text": "Valid deal with price €29.99 at https://store.com"},
            {"text": "Another valid deal"},
            {"text": "Deal that will fail processing"}
        ]
        
        csv_path = self.create_sample_csv("error_deals.csv", sample_data)
        output_path = os.path.join(self.temp_dir, "error_results.md")
        
        # Mock pipeline with mixed success/failure
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            if "fail" in text:
                raise PipelineError("Simulated processing error")
            elif "€29.99" in text:
                return {
                    "summary": "Great deal alert! Premium product now €29.99 with fast shipping. Limited time offer - order now for best price!",
                    "qa": QAResult(
                        has_price=True,
                        price_value=29.99,
                        has_source_url=True,
                        source_url="https://store.com",
                        within_length_limit=True,
                        missing_keywords=[],
                        duplicate_suspect=False
                    )
                }
            else:
                return {
                    "summary": "Another great deal available! Check out this amazing offer with excellent value. Don't miss out on this opportunity!",
                    "qa": QAResult(
                        has_price=False,
                        has_source_url=False,
                        within_length_limit=True,
                        missing_keywords=["deal", "%", "euro"],
                        duplicate_suspect=False
                    )
                }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        # Run CLI processing
        with patch('sys.stdout', new_callable=StringIO):
            with self.assertRaises(SystemExit) as context:
                self.cli.run([
                    "--input_csv", csv_path,
                    "--out", output_path
                ])
            # Should exit with code 1 (some failures)
            self.assertEqual(context.exception.code, 1)
        
        # Verify report content includes errors
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("✅ **Successful:** 2", content)
        self.assertIn("❌ **Failed:** 1", content)
        self.assertIn("**Status:** ❌ Processing Failed", content)
        self.assertIn("Simulated processing error", content)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_end_to_end_edge_cases_handling(self, mock_pipeline_class):
        """Test handling of edge cases like very long text, special characters, etc."""
        sample_data = [
            {"text": "Short"},
            {"text": "Very long text " + "word " * 300},  # Exceeds 200 words
            {"text": "Special chars: €29.99 & 50% off! Visit https://test.com?ref=deal&utm=special"},
            {"text": "DEAL IN ALL CAPS WITH EXCITEMENT!!!"},
            {"text": ""}  # Empty text (should be skipped)
        ]
        
        csv_path = self.create_sample_csv("edge_cases.csv", sample_data)
        output_path = os.path.join(self.temp_dir, "edge_results.md")
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            word_count = len(text.split())
            return {
                "summary": f"Summary for text with {word_count} words. Great deal available with excellent value proposition!",
                "qa": QAResult(
                    has_price="€" in text or "%" in text,
                    price_value=29.99 if "€29.99" in text else None,
                    has_source_url="http" in text,
                    source_url="https://test.com?ref=deal&utm=special" if "test.com" in text else None,
                    within_length_limit=word_count <= 200,
                    missing_keywords=[] if "deal" in text.lower() else ["deal"],
                    duplicate_suspect=False,
                    notes=f"Processed text with {word_count} words"
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        # Run CLI processing
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit):
                self.cli.run([
                    "--input_csv", csv_path,
                    "--out", output_path
                ])
        
        # Check that empty row warning was printed
        output = mock_stdout.getvalue()
        self.assertIn("Warning: Row", output)
        self.assertIn("empty 'text' field", output)
        
        # Verify report handles edge cases
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should process 4 valid deals (empty one skipped)
        self.assertIn("Total items processed: 4", content)
        
        # Check that long text is truncated in display
        self.assertIn("Text truncated for display", content)
        
        # Check special characters are handled
        self.assertIn("€29.99", content)
        self.assertIn("https://test.com", content)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_end_to_end_duplicate_detection(self, mock_pipeline_class):
        """Test end-to-end duplicate detection across multiple deals."""
        sample_data = [
            {"text": "Sony WH-1000XM4 headphones for €199.99 at https://store1.com"},
            {"text": "Sony WH-1000XM4 wireless headphones at €199.99 from https://store2.com"},  # Potential duplicate
            {"text": "Different product for €199.99 at https://store3.com"},  # Same price, different product
            {"text": "Sony headphones for €299.99 at https://store4.com"}  # Same product, different price
        ]
        
        csv_path = self.create_sample_csv("duplicate_test.csv", sample_data)
        output_path = os.path.join(self.temp_dir, "duplicate_results.md")
        
        # Mock pipeline with duplicate detection logic
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            # Simulate duplicate detection based on existing deals
            is_duplicate = False
            if existing_deals and "Sony WH-1000XM4" in text and "€199.99" in text:
                # Check if we already have a similar Sony deal at same price
                for deal in existing_deals:
                    if ('qa' in deal and deal['qa'].price_value == 199.99 and 
                        'Sony WH-1000XM4' in deal['input']):
                        is_duplicate = True
                        break
            
            # Extract proper URL from text
            source_url = None
            if "https://" in text:
                url_part = text.split("https://")[1].split()[0]
                source_url = f"https://{url_part}"
            
            return {
                "summary": f"Deal summary for {text[:30]}... Great value and quality product available now!",
                "qa": QAResult(
                    has_price=True,
                    price_value=199.99 if "€199.99" in text else 299.99,
                    has_source_url=source_url is not None,
                    source_url=source_url,
                    within_length_limit=True,
                    missing_keywords=[],
                    duplicate_suspect=is_duplicate,
                    notes="Duplicate detected" if is_duplicate else "Unique deal"
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        # Run CLI processing
        with patch('sys.stdout', new_callable=StringIO):
            with self.assertRaises(SystemExit):
                self.cli.run([
                    "--input_csv", csv_path,
                    "--out", output_path
                ])
        
        # Verify duplicate detection in report
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have both duplicate and non-duplicate results
        self.assertIn('"duplicate_suspect": true', content)
        self.assertIn('"duplicate_suspect": false', content)
        self.assertIn("Duplicate detected", content)
        self.assertIn("Unique deal", content)
    
    def test_end_to_end_argument_combinations(self):
        """Test various CLI argument combinations."""
        # Test different keyword combinations
        test_cases = [
            {
                "args": ["--text", "Test deal", "--keywords", "deal"],
                "description": "Single keyword"
            },
            {
                "args": ["--text", "Test deal", "--keywords", "deal", "discount", "sale"],
                "description": "Multiple keywords"
            },
            {
                "args": ["--text", "Test deal", "--out", "custom_report.md"],
                "description": "Custom output file"
            },
            {
                "args": ["--text", "Test deal", "--model", "gpt-4"],
                "description": "Custom model"
            },
            {
                "args": ["--text", "Test deal", "--provider", "openrouter"],
                "description": "OpenRouter provider"
            }
        ]
        
        for case in test_cases:
            with self.subTest(case["description"]):
                parser = self.cli.create_parser()
                
                # Should parse without errors
                try:
                    args = parser.parse_args(case["args"])
                    self.assertIsNotNone(args)
                except SystemExit:
                    self.fail(f"Argument parsing failed for {case['description']}")
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_end_to_end_large_csv_performance(self, mock_pipeline_class):
        """Test performance with larger CSV files."""
        # Create a larger dataset
        large_data = []
        for i in range(50):  # 50 deals
            large_data.append({
                "text": f"Deal {i+1}: Amazing product with {i*10}% discount! Price: €{i+10}.99. Visit https://store{i}.com/deal{i}"
            })
        
        csv_path = self.create_sample_csv("large_deals.csv", large_data)
        output_path = os.path.join(self.temp_dir, "large_results.md")
        
        # Mock pipeline with fast responses
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_deal(text, keywords, existing_deals=None):
            deal_num = int(text.split("Deal ")[1].split(":")[0])
            return {
                "summary": f"Great deal #{deal_num}! Amazing product with excellent value. Limited time offer - don't miss out on this opportunity!",
                "qa": QAResult(
                    has_price=True,
                    price_value=float(deal_num + 9),
                    has_source_url=True,
                    source_url=f"https://store{deal_num-1}.com/deal{deal_num-1}",
                    within_length_limit=True,
                    missing_keywords=[],
                    duplicate_suspect=False,
                    notes=f"Deal #{deal_num} processed successfully"
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_deal
        
        # Measure processing time
        import time
        start_time = time.time()
        
        with patch('sys.stdout', new_callable=StringIO):
            with self.assertRaises(SystemExit):
                self.cli.run([
                    "--input_csv", csv_path,
                    "--out", output_path
                ])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all deals were processed
        self.assertEqual(mock_pipeline.analyze_deal.call_count, 50)
        
        # Verify report was generated
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("Total items processed: 50", content)
        self.assertIn("✅ **Successful:** 50", content)
        
        # Performance should be reasonable (less than 5 seconds for mocked responses)
        self.assertLess(processing_time, 5.0, "Processing took too long for 50 mocked deals")


if __name__ == '__main__':
    unittest.main()