"""
LangChain pipeline integration for the Deal Summary & QA Bot.

This module provides the core processing logic that orchestrates summary generation
and quality assurance validation using LangChain chains and OpenAI's structured outputs.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional, Set
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import ValidationError

from .prompts import create_summary_prompt, create_qa_prompt
from .schemas import QAResult


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class LLMAPIError(PipelineError):
    """Exception for LLM API-related failures."""
    pass


class StructuredOutputError(PipelineError):
    """Exception for structured output parsing failures."""
    pass


def validate_keywords_case_insensitive(text: str, required_keywords: List[str]) -> List[str]:
    """
    Validate presence of required keywords in text using case-insensitive matching.
    
    Args:
        text: Text content to search for keywords
        required_keywords: List of keywords that must be present
        
    Returns:
        List of missing keywords (case-insensitive)
    """
    if not text or not required_keywords:
        return []
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    missing_keywords = []
    
    for keyword in required_keywords:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            continue
            
        # Check if keyword appears in text (case-insensitive)
        # Use word boundaries to avoid partial matches where appropriate
        if keyword_lower in text_lower:
            continue
        else:
            missing_keywords.append(keyword_lower)
    
    return missing_keywords


def detect_duplicate_by_price_product(text: str, price_value: Optional[float], 
                                    existing_deals: List[Dict[str, Any]] = None) -> bool:
    """
    Detect potential duplicates using price + product pattern matching.
    
    This function analyzes the combination of price and product keywords to identify
    potential duplicate deals. It looks for similar price points combined with
    similar product descriptions or brand names.
    
    Args:
        text: Deal text content to analyze
        price_value: Extracted price value from the deal
        existing_deals: List of previously processed deals for comparison
                       Each deal should have 'text' and 'price_value' keys
        
    Returns:
        True if the deal is suspected to be a duplicate, False otherwise
    """
    if not text or price_value is None:
        return False
    
    # If no existing deals provided, cannot detect duplicates
    if not existing_deals:
        return False
    
    # Extract key product indicators from current deal
    current_products = _extract_product_keywords(text)
    
    # Compare with existing deals
    for existing_deal in existing_deals:
        existing_price = existing_deal.get('price_value')
        existing_text = existing_deal.get('text', '')
        
        if existing_price is None or not existing_text:
            continue
        
        # Check for similar price (within 5% tolerance)
        price_similarity = _calculate_price_similarity(price_value, existing_price)
        if price_similarity < 0.95:  # Less than 95% similar
            continue
        
        # Check for product similarity
        existing_products = _extract_product_keywords(existing_text)
        product_similarity = _calculate_product_similarity(current_products, existing_products)
        
        # If both price and product are similar, flag as potential duplicate
        if price_similarity >= 0.95 and product_similarity >= 0.6:
            return True
    
    return False


def _extract_product_keywords(text: str) -> Set[str]:
    """
    Extract product-related keywords from deal text.
    
    Args:
        text: Deal text to analyze
        
    Returns:
        Set of normalized product keywords
    """
    if not text:
        return set()
    
    # Common product categories and brand patterns
    product_patterns = [
        r'\b(?:iphone|samsung|apple|sony|lg|huawei|xiaomi)\b',  # Electronics brands
        r'\b(?:headphones|earbuds|speaker|phone|tablet|laptop|tv)\b',  # Electronics
        r'\b(?:nike|adidas|puma|reebok)\b',  # Sports brands
        r'\b(?:shirt|shoes|jacket|pants|dress|watch)\b',  # Clothing/accessories
        r'\b(?:book|game|movie|software|app)\b',  # Digital products
        r'\b(?:coffee|tea|food|snack|drink)\b',  # Food/beverage
        r'\b(?:car|bike|vehicle|auto)\b',  # Automotive
    ]
    
    text_lower = text.lower()
    keywords = set()
    
    for pattern in product_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        keywords.update(matches)
    
    # Also extract potential brand names (capitalized words)
    brand_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    brand_matches = re.findall(brand_pattern, text)
    keywords.update([brand.lower() for brand in brand_matches])
    
    return keywords


def _calculate_price_similarity(price1: float, price2: float) -> float:
    """
    Calculate similarity between two prices.
    
    Args:
        price1: First price value
        price2: Second price value
        
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if price1 == price2:
        return 1.0
    
    # Calculate percentage difference
    max_price = max(price1, price2)
    min_price = min(price1, price2)
    
    if max_price == 0:
        return 1.0 if min_price == 0 else 0.0
    
    similarity = min_price / max_price
    return similarity


def _calculate_product_similarity(products1: Set[str], products2: Set[str]) -> float:
    """
    Calculate similarity between two sets of product keywords.
    
    Args:
        products1: First set of product keywords
        products2: Second set of product keywords
        
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if not products1 and not products2:
        return 1.0
    
    if not products1 or not products2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(products1.intersection(products2))
    union = len(products1.union(products2))
    
    if union == 0:
        return 1.0
    
    return intersection / union


class DealAnalysisPipeline:
    """
    Main pipeline class for processing deal content through summary and QA chains.
    
    This class encapsulates the LangChain integration and provides a clean interface
    for analyzing deal content with both summary generation and quality assurance.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "openai/gpt-4o-mini", base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize the pipeline with OpenRouter API configuration by default.
        
        Args:
            api_key: API key. If None, will use OPENROUTER_API_KEY or OPENAI_API_KEY environment variable
            model_name: Model to use for processing (default: openai/gpt-4o-mini for OpenRouter)
            base_url: Base URL for API (default: https://openrouter.ai/api/v1 for OpenRouter)
            
        Raises:
            PipelineError: If initialization fails due to invalid configuration
        """
        try:
            # Validate and set up API key - support both OpenRouter and OpenAI
            if api_key:
                if not isinstance(api_key, str) or not api_key.strip():
                    raise PipelineError("API key must be a non-empty string")
                os.environ["OPENAI_API_KEY"] = api_key.strip()
            elif not os.getenv("OPENAI_API_KEY"):
                # Try OpenRouter API key as fallback
                openrouter_key = os.getenv("OPENROUTER_API_KEY")
                if openrouter_key:
                    os.environ["OPENAI_API_KEY"] = openrouter_key
                else:
                    raise PipelineError(
                        "API key must be provided either as parameter or OPENROUTER_API_KEY/OPENAI_API_KEY environment variable. "
                        "Get your API key from https://openrouter.ai/keys or https://platform.openai.com/api-keys"
                    )
            
            # Validate model name
            if not isinstance(model_name, str) or not model_name.strip():
                raise PipelineError("Model name must be a non-empty string")
            
            # Initialize LLM with appropriate settings
            llm_kwargs = {
                "model": model_name.strip(),
                "timeout": 30,  # 30 second timeout for API calls
                "max_retries": 3,  # Retry failed requests up to 3 times
            }
            
            # Add base_url if provided (for OpenRouter or other providers)
            if base_url:
                if not isinstance(base_url, str) or not base_url.strip():
                    raise PipelineError("Base URL must be a non-empty string")
                llm_kwargs["base_url"] = base_url.strip()
            
            # Summary generation uses higher temperature for creativity
            self.summary_llm = ChatOpenAI(
                temperature=0.7,  # More creative for engaging summaries
                max_tokens=100,   # Limit tokens for concise summaries
                **llm_kwargs
            )
            
            # QA validation uses lower temperature for consistency (requirement 6.5)
            self.qa_llm = ChatOpenAI(
                temperature=0.1,  # More deterministic for validation
                max_tokens=500,   # Allow more tokens for detailed analysis
                **llm_kwargs
            )
            
            # Create prompt templates
            try:
                self.summary_prompt = create_summary_prompt()
                self.qa_prompt = create_qa_prompt()
            except Exception as e:
                raise PipelineError(f"Failed to create prompt templates: {str(e)}")
            
            # Build chains
            self._build_chains()
            
        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Failed to initialize pipeline: {str(e)}")
    
    def _build_chains(self):
        """
        Build the LangChain processing chains.
        
        Raises:
            PipelineError: If chain building fails
        """
        try:
            # Summary chain: prompt -> LLM -> string output
            self.summary_chain = (
                self.summary_prompt 
                | self.summary_llm 
                | StrOutputParser()
            )
            
            # QA chain: prompt -> LLM with structured output -> Pydantic validation
            self.qa_chain = (
                self.qa_prompt 
                | self.qa_llm.with_structured_output(QAResult)
            )
        except Exception as e:
            raise PipelineError(f"Failed to build processing chains: {str(e)}")
    
    def generate_summary(self, deal_text: str) -> str:
        """
        Generate a social-media ready summary for deal content.
        
        Args:
            deal_text: Raw deal content to summarize
            
        Returns:
            Concise summary between 180-220 characters
            
        Raises:
            LLMAPIError: If LLM API call fails
            PipelineError: If summary generation fails for other reasons
        """
        # Input validation
        if not isinstance(deal_text, str):
            raise PipelineError("Deal text must be a string")
        if not deal_text.strip():
            raise PipelineError("Deal text cannot be empty")
        
        # Retry logic for API failures
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                summary = self.summary_chain.invoke({
                    "deal_text": deal_text.strip()
                })
                
                # Validate summary is a string
                if not isinstance(summary, str):
                    raise PipelineError(f"Expected string summary, got {type(summary)}")
                
                summary = summary.strip()
                
                # Validate summary is not empty
                if not summary:
                    raise PipelineError("Generated summary is empty")
                
                # Validate summary length (180-220 characters)
                if len(summary) < 180 or len(summary) > 220:
                    # Log warning but don't fail - let the user decide
                    print(f"Warning: Summary length ({len(summary)} chars) outside target range (180-220)")
                
                return summary
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    print(f"API error on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise LLMAPIError(f"LLM API failed after {max_retries} attempts: {str(e)}")
            
            except OpenAIError as e:
                raise LLMAPIError(f"OpenAI API error during summary generation: {str(e)}")
            
            except Exception as e:
                if isinstance(e, (LLMAPIError, PipelineError)):
                    raise
                raise PipelineError(f"Unexpected error during summary generation: {str(e)}")
    
    def validate_content(self, deal_text: str, required_keywords: List[str], 
                        existing_deals: List[Dict[str, Any]] = None) -> QAResult:
        """
        Perform quality assurance validation on deal content.
        
        Args:
            deal_text: Raw deal content to validate
            required_keywords: List of keywords that must be present
            existing_deals: List of previously processed deals for duplicate detection
            
        Returns:
            QAResult object with validation results
            
        Raises:
            LLMAPIError: If LLM API call fails
            StructuredOutputError: If structured output parsing fails
            PipelineError: If validation fails for other reasons
        """
        # Input validation
        if not isinstance(deal_text, str):
            raise PipelineError("Deal text must be a string")
        if not deal_text.strip():
            raise PipelineError("Deal text cannot be empty")
        if not isinstance(required_keywords, list):
            raise PipelineError("Required keywords must be a list")
        
        # Retry logic for API failures
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                # Format keywords for prompt - use defaults if empty list provided
                keywords_str = ", ".join(required_keywords) if required_keywords else "deal, %, euro"
                
                # Get initial QA result from LLM
                qa_result = self.qa_chain.invoke({
                    "deal_text": deal_text.strip(),
                    "required_keywords": keywords_str
                })
                
                # Validate that we got a QAResult object
                if not isinstance(qa_result, QAResult):
                    raise StructuredOutputError(f"Expected QAResult object, got {type(qa_result)}")
                
                # Enhance with programmatic keyword validation
                try:
                    actual_missing_keywords = validate_keywords_case_insensitive(
                        deal_text, required_keywords or ["deal", "%", "euro"]
                    )
                    
                    # Update missing keywords with our programmatic validation
                    # Combine LLM results with our validation for better accuracy
                    all_missing = set(qa_result.missing_keywords + actual_missing_keywords)
                    qa_result.missing_keywords = sorted(list(all_missing))
                except Exception as e:
                    print(f"Warning: Keyword validation enhancement failed: {e}")
                    # Continue with LLM results only
                
                # Enhance with programmatic duplicate detection
                try:
                    if existing_deals and qa_result.price_value is not None:
                        is_duplicate = detect_duplicate_by_price_product(
                            deal_text, qa_result.price_value, existing_deals
                        )
                        # Use OR logic - if either LLM or our algorithm detects duplicate
                        qa_result.duplicate_suspect = qa_result.duplicate_suspect or is_duplicate
                except Exception as e:
                    print(f"Warning: Duplicate detection enhancement failed: {e}")
                    # Continue with LLM results only
                
                return qa_result
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    print(f"API error on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise LLMAPIError(f"LLM API failed after {max_retries} attempts: {str(e)}")
            
            except OpenAIError as e:
                raise LLMAPIError(f"OpenAI API error during content validation: {str(e)}")
            
            except ValidationError as e:
                # Graceful degradation for structured output parsing failures (requirement 6.4)
                print(f"Warning: Structured output validation failed: {e}")
                return self._create_fallback_qa_result(deal_text, required_keywords, existing_deals)
            
            except Exception as e:
                if isinstance(e, (LLMAPIError, StructuredOutputError, PipelineError)):
                    raise
                # Try fallback for unexpected errors
                print(f"Warning: Unexpected error during validation, using fallback: {e}")
                return self._create_fallback_qa_result(deal_text, required_keywords, existing_deals)
    
    def _create_fallback_qa_result(self, deal_text: str, required_keywords: List[str], 
                                  existing_deals: List[Dict[str, Any]] = None) -> QAResult:
        """
        Create a fallback QA result using basic text parsing when structured output fails.
        
        This provides graceful degradation when the LLM structured output fails,
        ensuring the system can still provide basic validation results.
        
        Args:
            deal_text: Raw deal content to validate
            required_keywords: List of keywords that must be present
            existing_deals: List of previously processed deals for duplicate detection
            
        Returns:
            QAResult object with basic validation results
        """
        try:
            # Basic price detection
            price_patterns = [
                r'€\s*(\d+(?:[.,]\d{2})?)',  # €29.99 or €29,99
                r'(\d+(?:[.,]\d{2})?)\s*€',  # 29.99€ or 29,99€
                r'(\d+(?:[.,]\d{2})?)\s*euro',  # 29.99 euro
                r'\$\s*(\d+(?:[.,]\d{2})?)',  # $29.99
                r'(\d+(?:[.,]\d{2})?)\s*\$',  # 29.99$
            ]
            
            price_value = None
            has_price = False
            
            for pattern in price_patterns:
                matches = re.findall(pattern, deal_text, re.IGNORECASE)
                if matches:
                    try:
                        # Take the first match and convert to float
                        price_str = matches[0].replace(',', '.')
                        price_value = float(price_str)
                        has_price = True
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Basic URL detection
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, deal_text)
            has_source_url = len(urls) > 0
            source_url = urls[0] if urls else None
            
            # Word count check (200 words limit)
            word_count = len(deal_text.split())
            within_length_limit = word_count <= 200
            
            # Keyword validation
            missing_keywords = validate_keywords_case_insensitive(
                deal_text, required_keywords or ["deal", "%", "euro"]
            )
            
            # Duplicate detection
            duplicate_suspect = False
            if existing_deals and price_value is not None:
                duplicate_suspect = detect_duplicate_by_price_product(
                    deal_text, price_value, existing_deals
                )
            
            return QAResult(
                has_price=has_price,
                price_value=price_value,
                has_source_url=has_source_url,
                source_url=source_url,
                within_length_limit=within_length_limit,
                missing_keywords=missing_keywords,
                duplicate_suspect=duplicate_suspect,
                notes="Generated using fallback validation due to structured output failure"
            )
            
        except Exception as e:
            # Last resort fallback with minimal validation
            return QAResult(
                has_price=False,
                price_value=None,
                has_source_url=False,
                source_url=None,
                within_length_limit=True,  # Assume within limit if we can't check
                missing_keywords=required_keywords or ["deal", "%", "euro"],
                duplicate_suspect=False,
                notes=f"Minimal validation due to fallback error: {str(e)}"
            )
    
    def analyze_deal(self, deal_text: str, required_keywords: List[str] = None, 
                    existing_deals: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate both summary generation and QA validation for deal content.
        
        This is the main entry point for processing deal content through both
        the summary and QA chains.
        
        Args:
            deal_text: Raw deal content to process
            required_keywords: List of required keywords for validation.
                             Defaults to ["deal", "%", "euro"] if None
            existing_deals: List of previously processed deals for duplicate detection
            
        Returns:
            Dictionary containing:
                - summary: Generated social media summary
                - qa: QAResult object with validation results
                
        Raises:
            PipelineError: If input validation fails
            LLMAPIError: If LLM API calls fail
            StructuredOutputError: If structured output parsing fails
        """
        # Input validation
        if not isinstance(deal_text, str):
            raise PipelineError("Deal text must be a string")
        if not deal_text.strip():
            raise PipelineError("Deal text cannot be empty")
        
        # Set default keywords if none provided
        if required_keywords is None:
            required_keywords = ["deal", "%", "euro"]
        elif not isinstance(required_keywords, list):
            raise PipelineError("Required keywords must be a list")
        
        # Validate existing_deals if provided
        if existing_deals is not None and not isinstance(existing_deals, list):
            raise PipelineError("Existing deals must be a list")
        
        summary_error = None
        qa_error = None
        
        # Try to generate summary
        try:
            summary = self.generate_summary(deal_text)
        except Exception as e:
            summary_error = e
            summary = f"[Summary generation failed: {str(e)}]"
        
        # Try to validate content
        try:
            qa_result = self.validate_content(deal_text, required_keywords, existing_deals)
        except Exception as e:
            qa_error = e
            # Use fallback QA result
            qa_result = self._create_fallback_qa_result(deal_text, required_keywords, existing_deals)
        
        # If both failed with critical errors, raise the more severe one
        if summary_error and qa_error:
            if isinstance(summary_error, LLMAPIError) or isinstance(qa_error, LLMAPIError):
                # Prioritize API errors as they indicate connectivity issues
                api_error = summary_error if isinstance(summary_error, LLMAPIError) else qa_error
                raise api_error
            else:
                # Raise the first error encountered
                raise summary_error
        
        return {
            "summary": summary,
            "qa": qa_result
        }


# Convenience function for simple usage
def analyze_deal(deal_text: str, required_keywords: List[str] = None, api_key: str = None, 
                model_name: str = "openai/gpt-4o-mini", base_url: str = "https://openrouter.ai/api/v1",
                existing_deals: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze deal content without creating a pipeline instance.
    
    Args:
        deal_text: Raw deal content to process
        required_keywords: List of required keywords for validation
        api_key: API key (optional if set in environment)
        model_name: Model to use for processing (default: openai/gpt-4o-mini for OpenRouter)
        base_url: Base URL for API (default: https://openrouter.ai/api/v1 for OpenRouter)
        existing_deals: List of previously processed deals for duplicate detection
        
    Returns:
        Dictionary containing summary and QA results
        
    Raises:
        PipelineError: If pipeline initialization or processing fails
        LLMAPIError: If LLM API calls fail
        StructuredOutputError: If structured output parsing fails
    """
    try:
        pipeline = DealAnalysisPipeline(api_key=api_key, model_name=model_name, base_url=base_url)
        return pipeline.analyze_deal(deal_text, required_keywords, existing_deals)
    except Exception as e:
        if isinstance(e, (PipelineError, LLMAPIError, StructuredOutputError)):
            raise
        raise PipelineError(f"Failed to analyze deal: {str(e)}")