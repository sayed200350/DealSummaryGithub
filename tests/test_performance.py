"""
Performance tests for batch processing with large datasets.

This module tests the performance characteristics of the Deal Summary & QA Bot
when processing large CSV files and validates optimization features.
"""

import csv
import os
import tempfile
import time
import unittest
from io import StringIO
from typing import List, Dict, Any
from unittest.mock import Mock, patch

import pytest

from src.cli import DealSummaryQACLI, InputValidationError, FileSystemError
from src.pipeline import PipelineError, LLMAPIError
from src.schemas import QAResult


class TestBatchProcessingPerformance(unittest.TestCase):
    """Test performance characteristics of batch processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = DealSummaryQACLI()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_large_csv(self, filename: str, num_rows: int) -> str:
        """Create a large CSV file for performance testing."""
        csv_path = os.path.join(self.temp_dir, filename)
        
        # Sample deal texts of varying lengths
        sample_texts = [
            "Amazing deal on electronics! Get 50% off premium headphones at €99.99. Visit https://store.com/deal",
            "Free shipping on all orders this weekend! No minimum purchase required.",
            "MacBook Pro with student discount - save 25% on M2 models. Perfect for creative work and studies. Apply at https://edu-store.com with valid student ID.",
            "Coffee lovers special: Premium Italian beans at 40% off, only €12.99/kg. Rich flavor with chocolate notes.",
            "Smartphone clearance sale! Samsung Galaxy models starting from €299. Limited stock available.",
            "Fashion week special: Designer clothes up to 70% off. Exclusive collection from top brands.",
            "Home appliances mega sale! Dishwashers, washing machines, and more at unbeatable prices.",
            "Gaming laptop deal: High-performance RTX graphics, 16GB RAM, 1TB SSD for €1299. Perfect for gamers.",
            "Organic food festival: Fresh produce, artisanal breads, and local specialties. This weekend only!",
            "Travel deals: European city breaks from €199 including flights and hotel. Book now for summer!"
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'category', 'id'])
            
            for i in range(num_rows):
                text = sample_texts[i % len(sample_texts)]
                # Add variation to make each row unique
                text = f"Deal #{i+1}: {text}"
                category = ['electronics', 'fashion', 'food', 'travel', 'home'][i % 5]
                writer.writerow([text, category, i+1])
        
        return csv_path
    
    def test_csv_validation_performance_small_file(self):
        """Test CSV validation performance with small files."""
        csv_path = self.create_large_csv("small_file.csv", 50)
        
        start_time = time.time()
        validation_results = self.cli.validate_csv_file(csv_path)
        validation_time = time.time() - start_time
        
        # Validation should be very fast for small files
        self.assertLess(validation_time, 1.0, "Small file validation should complete in under 1 second")
        
        # Check validation results
        self.assertEqual(validation_results['estimated_rows'], 50)
        self.assertEqual(validation_results['performance_estimate'], 'moderate')
        self.assertTrue(validation_results['has_text_column'])
        self.assertIn('text', validation_results['columns'])
    
    def test_csv_validation_performance_medium_file(self):
        """Test CSV validation performance with medium files."""
        csv_path = self.create_large_csv("medium_file.csv", 500)
        
        start_time = time.time()
        validation_results = self.cli.validate_csv_file(csv_path)
        validation_time = time.time() - start_time
        
        # Validation should still be fast for medium files
        self.assertLess(validation_time, 2.0, "Medium file validation should complete in under 2 seconds")
        
        # Check validation results
        self.assertEqual(validation_results['estimated_rows'], 500)
        self.assertEqual(validation_results['performance_estimate'], 'slow')
        self.assertTrue(validation_results['has_text_column'])
    
    def test_csv_validation_performance_large_file(self):
        """Test CSV validation performance with large files."""
        csv_path = self.create_large_csv("large_file.csv", 2000)
        
        start_time = time.time()
        validation_results = self.cli.validate_csv_file(csv_path)
        validation_time = time.time() - start_time
        
        # Validation should use sampling for large files and remain fast
        self.assertLess(validation_time, 3.0, "Large file validation should complete in under 3 seconds using sampling")
        
        # Check validation results
        self.assertEqual(validation_results['estimated_rows'], 2000)
        self.assertEqual(validation_results['performance_estimate'], 'very_slow')
        self.assertTrue(validation_results['has_text_column'])
        self.assertGreater(len(validation_results['warnings']), 0, "Large files should generate warnings")
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_batch_processing_progress_indication(self, mock_pipeline_class):
        """Test progress indication during batch processing."""
        # Create medium-sized file to test progress updates
        csv_path = self.create_large_csv("progress_test.csv", 100)
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock successful processing
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
            "summary": "Test summary",
            "qa": mock_qa_result
        }
        
        self.cli.pipeline = mock_pipeline
        
        # Capture output to verify progress messages
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            results = self.cli.process_csv_batch(csv_path, ["deal"])
        
        output = mock_stdout.getvalue()
        
        # Verify progress indicators are present
        self.assertIn("Processing 100 deals", output)
        self.assertIn("Batch processing complete", output)
        self.assertIn("Total processed: 100 deals", output)
        self.assertIn("Processing time:", output)
        self.assertIn("Average rate:", output)
        self.assertIn("deals/second", output)
        
        # Verify all deals were processed
        self.assertEqual(len(results), 100)
        self.assertEqual(mock_pipeline.analyze_deal.call_count, 100)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_batch_processing_large_file_progress(self, mock_pipeline_class):
        """Test progress indication for large files with percentage updates."""
        # Create large file to trigger percentage-based progress
        csv_path = self.create_large_csv("large_progress_test.csv", 200)
        
        # Mock pipeline with slight delay to simulate real processing
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_with_delay(text, keywords, existing_deals=None):
            time.sleep(0.001)  # Small delay to simulate processing
            return {
                "summary": f"Summary for: {text[:30]}...",
                "qa": QAResult(
                    has_price=True,
                    price_value=19.99,
                    has_source_url=False,
                    within_length_limit=True,
                    missing_keywords=[],
                    duplicate_suspect=False
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_with_delay
        self.cli.pipeline = mock_pipeline
        
        # Capture output to verify progress messages
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            start_time = time.time()
            results = self.cli.process_csv_batch(csv_path, ["deal"])
            processing_time = time.time() - start_time
        
        output = mock_stdout.getvalue()
        
        # Verify percentage-based progress for large files
        self.assertIn("Progress:", output)
        self.assertIn("Rate:", output)
        self.assertIn("ETA:", output)
        self.assertIn("deals/sec", output)
        
        # Verify performance statistics
        self.assertIn("Average rate:", output)
        self.assertIn("Processing time:", output)
        
        # Verify all deals were processed
        self.assertEqual(len(results), 200)
        
        # Performance should be reasonable (not too slow due to mocking)
        self.assertLess(processing_time, 10.0, "Mocked processing should complete quickly")
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_batch_processing_memory_management(self, mock_pipeline_class):
        """Test memory management during large batch processing."""
        # Create file large enough to test memory management features
        csv_path = self.create_large_csv("memory_test.csv", 1500)
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_pipeline.analyze_deal.return_value = {
            "summary": "Memory test summary",
            "qa": QAResult(
                has_price=False,
                has_source_url=False,
                within_length_limit=True,
                missing_keywords=["deal"],
                duplicate_suspect=False
            )
        }
        
        self.cli.pipeline = mock_pipeline
        
        # Monitor memory usage during processing
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            results = self.cli.process_csv_batch(csv_path, ["deal"])
        
        output = mock_stdout.getvalue()
        
        # Verify memory management messages for large batches
        self.assertIn("Processed 1000 deals, continuing", output)
        
        # Verify all deals were processed
        self.assertEqual(len(results), 1500)
        
        # Verify results structure is maintained
        for result in results[:10]:  # Check first 10 results
            self.assertIn('input', result)
            self.assertIn('summary', result)
            self.assertIn('qa', result)
            self.assertIn('row_number', result)
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_batch_processing_error_handling_performance(self, mock_pipeline_class):
        """Test error handling performance with mixed success/failure scenarios."""
        csv_path = self.create_large_csv("error_performance_test.csv", 100)
        
        # Mock pipeline with mixed results
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def mock_analyze_with_errors(text, keywords, existing_deals=None):
            # Simulate different error types based on deal number
            if "Deal #10:" in text or "Deal #20:" in text:
                raise LLMAPIError("Simulated API error")
            elif "Deal #15:" in text or "Deal #25:" in text:
                raise PipelineError("Simulated processing error")
            else:
                return {
                    "summary": f"Success: {text[:30]}...",
                    "qa": QAResult(
                        has_price=True,
                        price_value=25.99,
                        has_source_url=True,
                        source_url="https://test.com",
                        within_length_limit=True,
                        missing_keywords=[],
                        duplicate_suspect=False
                    )
                }
        
        mock_pipeline.analyze_deal.side_effect = mock_analyze_with_errors
        self.cli.pipeline = mock_pipeline
        
        # Process with errors
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            start_time = time.time()
            results = self.cli.process_csv_batch(csv_path, ["deal"])
            processing_time = time.time() - start_time
        
        output = mock_stdout.getvalue()
        
        # Verify error handling doesn't significantly impact performance
        self.assertLess(processing_time, 5.0, "Error handling should not significantly slow processing")
        
        # Verify error statistics are reported
        self.assertIn("API Errors:", output)
        self.assertIn("Other Errors:", output)
        self.assertIn("Successful:", output)
        
        # Verify all deals were processed (including errors)
        self.assertEqual(len(results), 100)
        
        # Count successful vs failed results
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        self.assertEqual(len(successful), 96)  # 100 - 4 errors
        self.assertEqual(len(failed), 4)  # 4 simulated errors
    
    def test_csv_validation_with_various_file_sizes(self):
        """Test CSV validation performance across different file sizes."""
        test_cases = [
            (10, "tiny"),
            (100, "small"),
            (500, "medium"),
            (1000, "large"),
            (2500, "very_large")
        ]
        
        validation_times = {}
        
        for num_rows, size_label in test_cases:
            csv_path = self.create_large_csv(f"{size_label}_file.csv", num_rows)
            
            start_time = time.time()
            validation_results = self.cli.validate_csv_file(csv_path)
            validation_time = time.time() - start_time
            
            validation_times[size_label] = validation_time
            
            # Verify validation results are reasonable
            self.assertEqual(validation_results['estimated_rows'], num_rows)
            self.assertTrue(validation_results['has_text_column'])
            
            # Validation should remain fast even for large files due to sampling
            self.assertLess(validation_time, 5.0, f"Validation for {size_label} file should be under 5 seconds")
        
        # Verify that validation time doesn't grow linearly with file size
        # (due to sampling optimization)
        tiny_time = validation_times['tiny']
        large_time = validation_times['very_large']
        
        # Large file should not take more than 50x longer than tiny file
        # (allowing for some variance in small timing measurements)
        max_ratio = 50 if tiny_time > 0.001 else 100  # More lenient for very fast operations
        self.assertLess(large_time, tiny_time * max_ratio, 
                       f"Large file validation should not scale linearly due to sampling. "
                       f"Tiny: {tiny_time:.4f}s, Large: {large_time:.4f}s, Ratio: {large_time/tiny_time:.1f}x")
    
    def test_performance_recommendations(self):
        """Test that performance recommendations are provided appropriately."""
        # Test with different file sizes to trigger different recommendations
        test_cases = [
            (50, False),    # Should not trigger recommendations
            (150, True),    # Should trigger recommendations
            (1000, True)    # Should definitely trigger recommendations
        ]
        
        for num_rows, should_have_recommendations in test_cases:
            csv_path = self.create_large_csv(f"recommendations_test_{num_rows}.csv", num_rows)
            
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                validation_results = self.cli.validate_csv_file(csv_path)
            
            output = mock_stdout.getvalue()
            
            if should_have_recommendations:
                # Should contain performance-related warnings or tips
                has_performance_info = any(keyword in output.lower() for keyword in 
                                         ['processing time', 'large dataset', 'batches', 'warning'])
                self.assertTrue(has_performance_info, 
                              f"File with {num_rows} rows should include performance recommendations")
            
            # Verify performance estimate is reasonable
            expected_estimates = {
                50: 'moderate',
                150: 'slow', 
                1000: 'very_slow'
            }
            
            self.assertEqual(validation_results['performance_estimate'], 
                           expected_estimates[num_rows])


class TestLargeDatasetIntegration(unittest.TestCase):
    """Integration tests with large datasets to verify end-to-end performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_large_csv(self, filename: str, num_rows: int) -> str:
        """Create a realistic large CSV file with varied content."""
        csv_path = os.path.join(self.temp_dir, filename)
        
        # More realistic deal templates
        deal_templates = [
            "Exclusive {brand} {product} deal! Save {discount}% on premium {category}. Was €{old_price}, now €{new_price}. Limited time offer at {url}",
            "Flash sale: {product} from {brand} at unbeatable prices! {discount}% off everything. Shop now at {url}",
            "Student discount available: {product} perfect for studies. {discount}% off with valid student ID. Visit {url}",
            "Weekend special: Free shipping on all {category} orders. {brand} {product} starting from €{new_price}. Order at {url}",
            "Clearance sale: {brand} {product} must go! Up to {discount}% off selected items. While stocks last at {url}",
            "New arrival: Latest {product} from {brand}. Special launch price €{new_price}. Get yours at {url}",
            "Bundle deal: {product} + accessories for €{new_price}. Save {discount}% compared to individual purchase. Available at {url}",
            "Seasonal offer: {category} collection from {brand}. {discount}% off everything. Limited time at {url}",
            "Premium {product} at affordable prices. {brand} quality for €{new_price}. {discount}% savings at {url}",
            "Last chance: {product} clearance from {brand}. Final {discount}% discount. Don't miss out - {url}"
        ]
        
        brands = ["TechCorp", "StyleBrand", "HomeMax", "SportsPro", "EcoLife", "LuxuryPlus", "ValueMart", "InnovateTech"]
        products = ["headphones", "laptop", "smartphone", "shoes", "jacket", "watch", "tablet", "camera"]
        categories = ["electronics", "fashion", "home", "sports", "lifestyle", "tech", "accessories", "gadgets"]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'category', 'brand', 'expected_price'])
            
            for i in range(num_rows):
                template = deal_templates[i % len(deal_templates)]
                brand = brands[i % len(brands)]
                product = products[i % len(products)]
                category = categories[i % len(categories)]
                
                discount = [10, 15, 20, 25, 30, 40, 50, 60, 70][i % 9]
                old_price = [99, 149, 199, 299, 399, 499, 599, 799, 999][i % 9]
                new_price = round(old_price * (100 - discount) / 100, 2)
                url = f"https://{brand.lower()}.com/deal-{i+1}"
                
                text = template.format(
                    brand=brand,
                    product=product,
                    category=category,
                    discount=discount,
                    old_price=old_price,
                    new_price=new_price,
                    url=url
                )
                
                writer.writerow([text, category, brand, new_price])
        
        return csv_path
    
    @patch('src.cli.DealAnalysisPipeline')
    def test_large_dataset_end_to_end_performance(self, mock_pipeline_class):
        """Test complete workflow performance with large realistic dataset."""
        # Create large realistic dataset
        csv_path = self.create_realistic_large_csv("large_realistic.csv", 300)
        output_path = os.path.join(self.temp_dir, "large_performance_report.md")
        
        # Mock pipeline with realistic processing times
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        def realistic_mock_analyze(text, keywords, existing_deals=None):
            # Simulate slight processing delay
            time.sleep(0.002)  # 2ms per deal
            
            # Extract realistic QA results based on text content
            has_price = "€" in text
            price_match = None
            if has_price:
                import re
                price_matches = re.findall(r'€(\d+(?:\.\d{2})?)', text)
                price_match = float(price_matches[-1]) if price_matches else None
            
            has_url = "http" in text
            url_match = None
            if has_url:
                import re
                url_matches = re.findall(r'https?://[^\s]+', text)
                url_match = url_matches[0] if url_matches else None
            
            # Check for keywords
            missing_keywords = []
            for keyword in keywords:
                if keyword.lower() not in text.lower():
                    missing_keywords.append(keyword)
            
            return {
                "summary": f"Great deal alert! {text[:100]}..." if len(text) > 100 else text,
                "qa": QAResult(
                    has_price=has_price,
                    price_value=price_match,
                    has_source_url=has_url,
                    source_url=url_match,
                    within_length_limit=len(text.split()) <= 200,
                    missing_keywords=missing_keywords,
                    duplicate_suspect=False,
                    notes="Realistic test data"
                )
            }
        
        mock_pipeline.analyze_deal.side_effect = realistic_mock_analyze
        
        # Run complete CLI workflow
        cli = DealSummaryQACLI()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            start_time = time.time()
            
            with self.assertRaises(SystemExit) as context:
                cli.run([
                    "--input_csv", csv_path,
                    "--keywords", "deal", "%", "euro",
                    "--out", output_path
                ])
            
            total_time = time.time() - start_time
            
            # Should exit successfully
            self.assertEqual(context.exception.code, 0)
        
        output = mock_stdout.getvalue()
        
        # Verify performance characteristics
        self.assertLess(total_time, 30.0, "Large dataset processing should complete in reasonable time")
        
        # Verify progress reporting
        self.assertIn("CSV Validation Summary", output)
        self.assertIn("Expected processing time", output)
        self.assertIn("Processing 300 deals", output)
        self.assertIn("Batch processing complete", output)
        self.assertIn("Average rate:", output)
        
        # Verify report was generated
        self.assertTrue(os.path.exists(output_path))
        
        # Verify report content
        with open(output_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        self.assertIn("Total items processed: 300", report_content)
        self.assertIn("✅ **Successful:** 300", report_content)
        
        # Verify processing rate is reasonable
        rate_match = None
        import re
        rate_matches = re.findall(r'Average rate: ([\d.]+) deals/second', output)
        if rate_matches:
            rate = float(rate_matches[0])
            self.assertGreater(rate, 5.0, "Processing rate should be at least 5 deals/second with mocking")
            self.assertLess(rate, 1000.0, "Processing rate should be realistic")


if __name__ == '__main__':
    unittest.main()