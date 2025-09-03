"""
Data models and validation schemas for the Deal Summary & QA Bot.

This module defines Pydantic models for structured data validation,
particularly for QA results that need to conform to strict JSON schemas
for OpenAI structured outputs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class QAResult(BaseModel):
    """
    Quality assurance validation result model.
    
    This model represents the structured output of deal content validation,
    checking for price presence, source URLs, length limits, required keywords,
    and potential duplicates.
    """
    
    model_config = ConfigDict(
        # Enable JSON schema generation for OpenAI structured outputs
        json_schema_extra={
            "example": {
                "has_price": True,
                "price_value": 29.99,
                "has_source_url": True,
                "source_url": "https://example.com/deal",
                "within_length_limit": True,
                "missing_keywords": [],
                "duplicate_suspect": False,
                "notes": "Valid deal content with all required elements"
            }
        },
        # Allow validation of assignment
        validate_assignment=True,
        # Use enum values for better JSON schema generation
        use_enum_values=True
    )
    
    has_price: bool = Field(
        description="Whether the content contains a price (EUR or identifiable number)"
    )
    
    price_value: Optional[float] = Field(
        default=None,
        description="Extracted numeric price value if found",
        ge=0  # Price must be non-negative
    )
    
    has_source_url: bool = Field(
        description="Whether the content contains a source URL (http/https)"
    )
    
    source_url: Optional[str] = Field(
        default=None,
        description="Extracted source URL if found"
    )
    
    within_length_limit: bool = Field(
        description="Whether the content is within 200 words limit"
    )
    
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="List of required keywords that are missing from the content"
    )
    
    duplicate_suspect: bool = Field(
        description="Whether the content is suspected to be a duplicate based on price+product pattern"
    )
    
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional notes or observations about the validation"
    )
    
    @field_validator('missing_keywords')
    @classmethod
    def validate_missing_keywords(cls, v):
        """Ensure missing_keywords contains only non-empty strings."""
        if not isinstance(v, list):
            raise ValueError("missing_keywords must be a list")
        
        for keyword in v:
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError("All missing keywords must be non-empty strings")
        
        return [keyword.strip().lower() for keyword in v]
    
    @field_validator('source_url')
    @classmethod
    def validate_source_url(cls, v):
        """Validate source URL format."""
        if v is not None and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("source_url must be a valid HTTP/HTTPS URL")
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Ensure consistency between boolean flags and their corresponding values."""
        # Validate price consistency
        if self.has_price and self.price_value is None:
            raise ValueError("price_value must be provided when has_price is True")
        if not self.has_price and self.price_value is not None:
            raise ValueError("price_value should be None when has_price is False")
        
        # Validate URL consistency
        if self.has_source_url and self.source_url is None:
            raise ValueError("source_url must be provided when has_source_url is True")
        if not self.has_source_url and self.source_url is not None:
            raise ValueError("source_url should be None when has_source_url is False")
        
        return self