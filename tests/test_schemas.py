"""
Unit tests for data models and validation schemas.

Tests cover valid and invalid data scenarios for the QAResult model,
including field validation, type checking, and business logic validation.
"""

import pytest
from pydantic import ValidationError
from src.schemas import QAResult


class TestQAResult:
    """Test cases for QAResult Pydantic model."""
    
    def test_valid_complete_qa_result(self):
        """Test creation of QAResult with all valid fields."""
        data = {
            "has_price": True,
            "price_value": 29.99,
            "has_source_url": True,
            "source_url": "https://example.com/deal",
            "within_length_limit": True,
            "missing_keywords": [],
            "duplicate_suspect": False,
            "notes": "Valid deal content"
        }
        
        result = QAResult(**data)
        
        assert result.has_price is True
        assert result.price_value == 29.99
        assert result.has_source_url is True
        assert str(result.source_url) == "https://example.com/deal"
        assert result.within_length_limit is True
        assert result.missing_keywords == []
        assert result.duplicate_suspect is False
        assert result.notes == "Valid deal content"
    
    def test_valid_minimal_qa_result(self):
        """Test creation of QAResult with minimal required fields."""
        data = {
            "has_price": False,
            "has_source_url": False,
            "within_length_limit": True,
            "duplicate_suspect": False
        }
        
        result = QAResult(**data)
        
        assert result.has_price is False
        assert result.price_value is None
        assert result.has_source_url is False
        assert result.source_url is None
        assert result.within_length_limit is True
        assert result.missing_keywords == []
        assert result.duplicate_suspect is False
        assert result.notes is None
    
    def test_valid_with_missing_keywords(self):
        """Test QAResult with missing keywords list."""
        data = {
            "has_price": True,
            "price_value": 15.50,
            "has_source_url": True,
            "source_url": "https://shop.example.com",
            "within_length_limit": False,
            "missing_keywords": ["deal", "discount"],
            "duplicate_suspect": True
        }
        
        result = QAResult(**data)
        
        assert result.missing_keywords == ["deal", "discount"]
        assert result.duplicate_suspect is True
    
    def test_price_validation_consistency_valid(self):
        """Test valid price consistency scenarios."""
        # Case 1: has_price=True with price_value
        data1 = {
            "has_price": True,
            "price_value": 99.99,
            "has_source_url": False,
            "within_length_limit": True,
            "duplicate_suspect": False
        }
        result1 = QAResult(**data1)
        assert result1.has_price is True
        assert result1.price_value == 99.99
        
        # Case 2: has_price=False without price_value
        data2 = {
            "has_price": False,
            "has_source_url": False,
            "within_length_limit": True,
            "duplicate_suspect": False
        }
        result2 = QAResult(**data2)
        assert result2.has_price is False
        assert result2.price_value is None
    
    def test_url_validation_consistency_valid(self):
        """Test valid URL consistency scenarios."""
        # Case 1: has_source_url=True with source_url
        data1 = {
            "has_price": False,
            "has_source_url": True,
            "source_url": "https://deals.example.com/item/123",
            "within_length_limit": True,
            "duplicate_suspect": False
        }
        result1 = QAResult(**data1)
        assert result1.has_source_url is True
        assert str(result1.source_url) == "https://deals.example.com/item/123"
        
        # Case 2: has_source_url=False without source_url
        data2 = {
            "has_price": False,
            "has_source_url": False,
            "within_length_limit": True,
            "duplicate_suspect": False
        }
        result2 = QAResult(**data2)
        assert result2.has_source_url is False
        assert result2.source_url is None
    
    def test_invalid_price_consistency(self):
        """Test invalid price consistency scenarios."""
        # Case 1: has_price=True but no price_value
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=True,
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "price_value must be provided when has_price is True" in str(exc_info.value)
        
        # Case 2: has_price=False but price_value provided
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                price_value=25.00,
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "price_value should be None when has_price is False" in str(exc_info.value)
    
    def test_invalid_url_consistency(self):
        """Test invalid URL consistency scenarios."""
        # Case 1: has_source_url=True but no source_url
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=True,
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "source_url must be provided when has_source_url is True" in str(exc_info.value)
        
        # Case 2: has_source_url=False but source_url provided
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=False,
                source_url="https://example.com",
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "source_url should be None when has_source_url is False" in str(exc_info.value)
    
    def test_invalid_price_negative(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=True,
                price_value=-10.50,
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_url_format(self):
        """Test that invalid URL formats are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=True,
                source_url="not-a-valid-url",
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "source_url must be a valid HTTP/HTTPS URL" in str(exc_info.value)
    
    def test_invalid_missing_keywords_type(self):
        """Test that invalid missing_keywords types are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=False,
                within_length_limit=True,
                missing_keywords="not-a-list",
                duplicate_suspect=False
            )
        assert "Input should be a valid list" in str(exc_info.value)
    
    def test_invalid_missing_keywords_empty_strings(self):
        """Test that empty strings in missing_keywords are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=False,
                within_length_limit=True,
                missing_keywords=["deal", "", "discount"],
                duplicate_suspect=False
            )
        assert "All missing keywords must be non-empty strings" in str(exc_info.value)
    
    def test_missing_keywords_normalization(self):
        """Test that missing_keywords are normalized to lowercase and trimmed."""
        data = {
            "has_price": False,
            "has_source_url": False,
            "within_length_limit": True,
            "missing_keywords": ["  DEAL  ", "Discount", "EURO "],
            "duplicate_suspect": False
        }
        
        result = QAResult(**data)
        assert result.missing_keywords == ["deal", "discount", "euro"]
    
    def test_notes_max_length(self):
        """Test that notes field respects maximum length constraint."""
        long_notes = "x" * 501  # Exceeds 500 character limit
        
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False,
                notes=long_notes
            )
        assert "String should have at most 500 characters" in str(exc_info.value)
    
    def test_required_fields_missing(self):
        """Test that missing required fields raise validation errors."""
        # Missing has_price
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False
            )
        assert "Field required" in str(exc_info.value)
    
    def test_json_serialization(self):
        """Test that QAResult can be serialized to JSON."""
        data = {
            "has_price": True,
            "price_value": 42.99,
            "has_source_url": True,
            "source_url": "https://example.com/deal",
            "within_length_limit": False,
            "missing_keywords": ["deal"],
            "duplicate_suspect": True,
            "notes": "Test serialization"
        }
        
        result = QAResult(**data)
        json_data = result.model_dump()
        
        assert json_data["has_price"] is True
        assert json_data["price_value"] == 42.99
        assert str(json_data["source_url"]) == "https://example.com/deal"
        assert json_data["missing_keywords"] == ["deal"]
    
    def test_schema_generation(self):
        """Test that JSON schema can be generated for OpenAI structured outputs."""
        schema = QAResult.model_json_schema()
        
        assert "properties" in schema
        assert "has_price" in schema["properties"]
        assert "price_value" in schema["properties"]
        assert "source_url" in schema["properties"]
        assert "missing_keywords" in schema["properties"]
        assert schema["properties"]["has_price"]["type"] == "boolean"
        # In Pydantic v2, the schema structure is slightly different
        assert "anyOf" in schema["properties"]["price_value"] or schema["properties"]["price_value"]["type"] == "number"


class TestQAResultEdgeCases:
    """Additional edge case tests for QAResult model."""
    
    def test_extreme_price_values(self):
        """Test QAResult with extreme price values."""
        # Very small price
        result1 = QAResult(
            has_price=True,
            price_value=0.01,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result1.price_value == 0.01
        
        # Very large price
        result2 = QAResult(
            has_price=True,
            price_value=999999.99,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result2.price_value == 999999.99
        
        # Zero price (free item)
        result3 = QAResult(
            has_price=True,
            price_value=0.0,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result3.price_value == 0.0
    
    def test_complex_url_formats(self):
        """Test QAResult with various complex URL formats."""
        test_urls = [
            "https://example.com/path/to/deal?param1=value1&param2=value2",
            "http://subdomain.example.com:8080/deal",
            "https://example.com/deal#section",
            "https://example.com/deal?utm_source=test&utm_medium=email&utm_campaign=deal",
            "https://example-store.com/category/subcategory/product-name-123"
        ]
        
        for url in test_urls:
            result = QAResult(
                has_price=False,
                has_source_url=True,
                source_url=url,
                within_length_limit=True,
                duplicate_suspect=False
            )
            assert str(result.source_url) == url
    
    def test_missing_keywords_edge_cases(self):
        """Test missing_keywords field with various edge cases."""
        # Empty list (default)
        result1 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result1.missing_keywords == []
        
        # Single keyword
        result2 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            missing_keywords=["deal"],
            duplicate_suspect=False
        )
        assert result2.missing_keywords == ["deal"]
        
        # Many keywords
        many_keywords = [f"keyword{i}" for i in range(20)]
        result3 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            missing_keywords=many_keywords,
            duplicate_suspect=False
        )
        assert len(result3.missing_keywords) == 20
        
        # Keywords with special characters (should be normalized)
        result4 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            missing_keywords=["  DEAL!  ", "50%", "€uro"],
            duplicate_suspect=False
        )
        assert result4.missing_keywords == ["deal!", "50%", "€uro"]
    
    def test_notes_field_edge_cases(self):
        """Test notes field with various content types."""
        # Unicode characters
        result1 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False,
            notes="Deal with émojis and spëcial chäractërs: 🎉💰"
        )
        assert "émojis" in result1.notes
        
        # JSON-like content in notes
        result2 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False,
            notes='{"analysis": "complex", "score": 0.85}'
        )
        assert '"analysis"' in result2.notes
        
        # Multiline notes
        multiline_notes = """Line 1: Analysis results
Line 2: Additional observations
Line 3: Recommendations"""
        result3 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False,
            notes=multiline_notes
        )
        assert "Line 1" in result3.notes
        assert "Line 3" in result3.notes
        
        # Maximum length notes (exactly 500 chars)
        max_notes = "x" * 500
        result4 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False,
            notes=max_notes
        )
        assert len(result4.notes) == 500
    
    def test_model_serialization_edge_cases(self):
        """Test model serialization with edge cases."""
        # Model with all optional fields as None
        result1 = QAResult(
            has_price=False,
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        
        serialized = result1.model_dump()
        assert serialized["price_value"] is None
        assert serialized["source_url"] is None
        assert serialized["notes"] is None
        assert serialized["missing_keywords"] == []
        
        # Model with all fields populated
        result2 = QAResult(
            has_price=True,
            price_value=42.99,
            has_source_url=True,
            source_url="https://example.com/deal",
            within_length_limit=False,
            missing_keywords=["deal", "discount"],
            duplicate_suspect=True,
            notes="Complete analysis with all fields"
        )
        
        serialized = result2.model_dump()
        assert all(key in serialized for key in [
            "has_price", "price_value", "has_source_url", "source_url",
            "within_length_limit", "missing_keywords", "duplicate_suspect", "notes"
        ])
    
    def test_model_validation_with_type_coercion(self):
        """Test that model handles type coercion appropriately."""
        # String price that can be converted to float
        result1 = QAResult(
            has_price=True,
            price_value="29.99",  # String instead of float
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result1.price_value == 29.99
        assert isinstance(result1.price_value, float)
        
        # Integer price
        result2 = QAResult(
            has_price=True,
            price_value=30,  # Integer instead of float
            has_source_url=False,
            within_length_limit=True,
            duplicate_suspect=False
        )
        assert result2.price_value == 30.0
        assert isinstance(result2.price_value, float)
    
    def test_model_copy_and_modification(self):
        """Test model copying and field modification."""
        original = QAResult(
            has_price=True,
            price_value=29.99,
            has_source_url=True,
            source_url="https://example.com",
            within_length_limit=True,
            missing_keywords=["deal"],
            duplicate_suspect=False,
            notes="Original notes"
        )
        
        # Create a copy with modifications
        modified = original.model_copy(update={
            "price_value": 39.99,
            "missing_keywords": [],
            "notes": "Modified notes"
        })
        
        # Original should be unchanged
        assert original.price_value == 29.99
        assert original.missing_keywords == ["deal"]
        assert original.notes == "Original notes"
        
        # Modified should have new values
        assert modified.price_value == 39.99
        assert modified.missing_keywords == []
        assert modified.notes == "Modified notes"
        
        # Other fields should be the same
        assert modified.has_price == original.has_price
        assert modified.source_url == original.source_url
    
    def test_model_validation_error_messages(self):
        """Test that validation error messages are helpful."""
        # Test price consistency error message
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=True,  # True but no price_value
                has_source_url=False,
                within_length_limit=True,
                duplicate_suspect=False
            )
        
        error_msg = str(exc_info.value)
        assert "price_value must be provided when has_price is True" in error_msg
        
        # Test URL consistency error message
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=True,  # True but no source_url
                within_length_limit=True,
                duplicate_suspect=False
            )
        
        error_msg = str(exc_info.value)
        assert "source_url must be provided when has_source_url is True" in error_msg
        
        # Test invalid URL format error
        with pytest.raises(ValidationError) as exc_info:
            QAResult(
                has_price=False,
                has_source_url=True,
                source_url="not-a-valid-url",
                within_length_limit=True,
                duplicate_suspect=False
            )
        
        error_msg = str(exc_info.value)
        assert "source_url must be a valid HTTP/HTTPS URL" in error_msg