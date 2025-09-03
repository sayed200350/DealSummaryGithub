"""
Tests for prompt templates and configuration.

This module tests the prompt template creation, rendering, and validation
to ensure they work correctly with sample data and meet the requirements.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from src.prompts import (
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    QA_SYSTEM,
    QA_USER,
    create_summary_prompt,
    create_qa_prompt,
    SUMMARY_TEMPLATE_VARS,
    QA_TEMPLATE_VARS
)


class TestPromptConstants:
    """Test the prompt constant definitions."""
    
    def test_summary_system_prompt_content(self):
        """Test that SUMMARY_SYSTEM contains required rules."""
        assert "180-220 characters" in SUMMARY_SYSTEM
        assert "brand or product name" in SUMMARY_SYSTEM
        assert "exactly one" in SUMMARY_SYSTEM
        assert "call-to-action" in SUMMARY_SYSTEM
        assert "no emojis" in SUMMARY_SYSTEM or "NOT include any emojis" in SUMMARY_SYSTEM
        assert "engaging" in SUMMARY_SYSTEM
        assert "concise" in SUMMARY_SYSTEM
    
    def test_summary_user_prompt_template(self):
        """Test that SUMMARY_USER contains template variable."""
        assert "{deal_text}" in SUMMARY_USER
        assert "180-220 characters" in SUMMARY_USER
        assert "no emojis" in SUMMARY_USER
    
    def test_qa_system_prompt_content(self):
        """Test that QA_SYSTEM contains all validation rules."""
        # Price validation
        assert "price" in QA_SYSTEM.lower()
        assert "eur" in QA_SYSTEM.lower()
        
        # URL validation
        assert "http" in QA_SYSTEM.lower()
        assert "url" in QA_SYSTEM.lower()
        
        # Length validation
        assert "200 words" in QA_SYSTEM
        
        # Keyword validation
        assert "keyword" in QA_SYSTEM.lower()
        assert "case-insensitive" in QA_SYSTEM.lower()
        
        # Duplicate detection
        assert "duplicate" in QA_SYSTEM.lower()
        assert "pattern" in QA_SYSTEM.lower()
        
        # JSON format requirement
        assert "JSON" in QA_SYSTEM
    
    def test_qa_user_prompt_template(self):
        """Test that QA_USER contains required template variables."""
        assert "{deal_text}" in QA_USER
        assert "{required_keywords}" in QA_USER
        assert "JSON format" in QA_USER


class TestPromptTemplateCreation:
    """Test the prompt template creation functions."""
    
    def test_create_summary_prompt_returns_chat_template(self):
        """Test that create_summary_prompt returns a ChatPromptTemplate."""
        prompt = create_summary_prompt()
        assert isinstance(prompt, ChatPromptTemplate)
    
    def test_create_qa_prompt_returns_chat_template(self):
        """Test that create_qa_prompt returns a ChatPromptTemplate."""
        prompt = create_qa_prompt()
        assert isinstance(prompt, ChatPromptTemplate)
    
    def test_summary_prompt_has_correct_variables(self):
        """Test that summary prompt has the expected input variables."""
        prompt = create_summary_prompt()
        assert set(prompt.input_variables) == set(SUMMARY_TEMPLATE_VARS)
    
    def test_qa_prompt_has_correct_variables(self):
        """Test that QA prompt has the expected input variables."""
        prompt = create_qa_prompt()
        assert set(prompt.input_variables) == set(QA_TEMPLATE_VARS)


class TestPromptRendering:
    """Test prompt template rendering with sample data."""
    
    @pytest.fixture
    def sample_deal_text(self):
        """Sample deal text for testing."""
        return """
        Amazing deal on Nike Air Max sneakers! Get 50% off the regular price of €99.99.
        These premium running shoes feature advanced cushioning technology and stylish design.
        Perfect for athletes and casual wear. Limited time offer - only while supplies last!
        Check out the full details at https://example-store.com/nike-air-max-deal
        Don't miss this incredible opportunity to upgrade your footwear collection.
        """
    
    @pytest.fixture
    def sample_keywords(self):
        """Sample required keywords for testing."""
        return ["deal", "%", "euro"]
    
    def test_summary_prompt_rendering(self, sample_deal_text):
        """Test that summary prompt renders correctly with sample data."""
        prompt = create_summary_prompt()
        
        # Test rendering
        messages = prompt.format_messages(deal_text=sample_deal_text)
        
        # Should have system and human messages
        assert len(messages) == 2
        
        # Check system message content
        system_message = messages[0]
        assert "180-220 characters" in system_message.content
        assert "copywriter" in system_message.content
        
        # Check human message content
        human_message = messages[1]
        assert sample_deal_text in human_message.content
        assert "180-220 characters" in human_message.content
    
    def test_qa_prompt_rendering(self, sample_deal_text, sample_keywords):
        """Test that QA prompt renders correctly with sample data."""
        prompt = create_qa_prompt()
        
        # Test rendering
        messages = prompt.format_messages(
            deal_text=sample_deal_text,
            required_keywords=sample_keywords
        )
        
        # Should have system and human messages
        assert len(messages) == 2
        
        # Check system message content
        system_message = messages[0]
        assert "quality assurance" in system_message.content
        assert "JSON format" in system_message.content
        
        # Check human message content
        human_message = messages[1]
        assert sample_deal_text in human_message.content
        assert str(sample_keywords) in human_message.content
    
    def test_summary_prompt_with_empty_text(self):
        """Test summary prompt rendering with empty text."""
        prompt = create_summary_prompt()
        
        messages = prompt.format_messages(deal_text="")
        assert len(messages) == 2
        assert "" in messages[1].content
    
    def test_qa_prompt_with_empty_keywords(self, sample_deal_text):
        """Test QA prompt rendering with empty keywords list."""
        prompt = create_qa_prompt()
        
        messages = prompt.format_messages(
            deal_text=sample_deal_text,
            required_keywords=[]
        )
        assert len(messages) == 2
        assert "[]" in messages[1].content
    
    def test_qa_prompt_with_special_characters(self):
        """Test QA prompt rendering with special characters in content."""
        prompt = create_qa_prompt()
        special_text = "Deal with €29.99 & 50% off! Visit https://test.com?ref=deal&utm=special"
        
        messages = prompt.format_messages(
            deal_text=special_text,
            required_keywords=["deal", "€"]
        )
        
        assert len(messages) == 2
        assert special_text in messages[1].content


class TestPromptTemplateVariables:
    """Test the template variable constants."""
    
    def test_summary_template_vars_constant(self):
        """Test that SUMMARY_TEMPLATE_VARS contains expected variables."""
        assert SUMMARY_TEMPLATE_VARS == ["deal_text"]
    
    def test_qa_template_vars_constant(self):
        """Test that QA_TEMPLATE_VARS contains expected variables."""
        assert set(QA_TEMPLATE_VARS) == {"deal_text", "required_keywords"}
    
    def test_template_vars_match_actual_prompts(self):
        """Test that template variable constants match actual prompt variables."""
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        assert set(summary_prompt.input_variables) == set(SUMMARY_TEMPLATE_VARS)
        assert set(qa_prompt.input_variables) == set(QA_TEMPLATE_VARS)


class TestPromptRequirements:
    """Test that prompts meet specific requirements from the spec."""
    
    def test_summary_character_limit_mentioned(self):
        """Test that summary prompts mention the 180-220 character limit."""
        # Requirement 1.1: 180-220 characters
        assert "180-220" in SUMMARY_SYSTEM
        assert "180-220" in SUMMARY_USER
    
    def test_summary_brand_requirement_mentioned(self):
        """Test that summary prompts mention brand/product name requirement."""
        # Requirement 1.2: include brand or product name
        assert "brand" in SUMMARY_SYSTEM.lower() or "product" in SUMMARY_SYSTEM.lower()
    
    def test_summary_cta_requirement_mentioned(self):
        """Test that summary prompts mention call-to-action requirement."""
        # Requirement 1.3: exactly one call-to-action
        assert "call-to-action" in SUMMARY_SYSTEM.lower()
        assert "exactly one" in SUMMARY_SYSTEM.lower()
    
    def test_summary_emoji_restriction_mentioned(self):
        """Test that summary prompts mention emoji restriction."""
        # Requirement 1.4: exclude emojis
        assert "emoji" in SUMMARY_SYSTEM.lower()
        assert ("no emoji" in SUMMARY_SYSTEM.lower() or 
                "not include" in SUMMARY_SYSTEM.lower() or
                "exclude emoji" in SUMMARY_SYSTEM.lower())
    
    def test_qa_price_validation_mentioned(self):
        """Test that QA prompts mention price validation."""
        # Requirement 2.1: check for price presence
        assert "price" in QA_SYSTEM.lower()
        assert "eur" in QA_SYSTEM.lower()
    
    def test_qa_keyword_validation_mentioned(self):
        """Test that QA prompts mention keyword validation."""
        # Requirement 2.6: verify presence of required keywords
        assert "keyword" in QA_SYSTEM.lower()
        assert "required" in QA_SYSTEM.lower()


class TestPromptEdgeCases:
    """Test prompt handling with edge cases and complex scenarios."""
    
    def test_prompt_with_very_long_deal_text(self):
        """Test prompt rendering with very long deal text."""
        # Create very long text (over 2000 characters)
        long_text = "Amazing deal on premium products! " * 100
        
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Should handle long text without errors
        summary_messages = summary_prompt.format_messages(deal_text=long_text)
        qa_messages = qa_prompt.format_messages(
            deal_text=long_text,
            required_keywords=["deal", "premium"]
        )
        
        assert len(summary_messages) == 2
        assert len(qa_messages) == 2
        assert long_text in summary_messages[1].content
        assert long_text in qa_messages[1].content
    
    def test_prompt_with_special_characters_and_unicode(self):
        """Test prompt rendering with special characters and Unicode."""
        special_text = "Spëcial déal with émojis 🎉💰! Price: €29.99 & 50% off 中文字符 русский العربية"
        special_keywords = ["déal", "€", "50%", "中文"]
        
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Should handle special characters without errors
        summary_messages = summary_prompt.format_messages(deal_text=special_text)
        qa_messages = qa_prompt.format_messages(
            deal_text=special_text,
            required_keywords=special_keywords
        )
        
        assert special_text in summary_messages[1].content
        assert special_text in qa_messages[1].content
        assert str(special_keywords) in qa_messages[1].content
    
    def test_prompt_with_json_like_content(self):
        """Test prompt rendering with JSON-like content in deal text."""
        json_like_text = '''Deal: {"product": "laptop", "price": 999.99, "discount": "50%"}'''
        
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Should handle JSON-like content without breaking template
        summary_messages = summary_prompt.format_messages(deal_text=json_like_text)
        qa_messages = qa_prompt.format_messages(
            deal_text=json_like_text,
            required_keywords=["deal", "laptop"]
        )
        
        assert json_like_text in summary_messages[1].content
        assert json_like_text in qa_messages[1].content
    
    def test_prompt_with_multiline_deal_text(self):
        """Test prompt rendering with multiline deal text."""
        multiline_text = """Line 1: Amazing deal on laptops!
Line 2: Get 50% off premium models
Line 3: Visit our store today
Line 4: Limited time offer"""
        
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Should preserve multiline structure
        summary_messages = summary_prompt.format_messages(deal_text=multiline_text)
        qa_messages = qa_prompt.format_messages(
            deal_text=multiline_text,
            required_keywords=["deal", "laptops"]
        )
        
        assert "Line 1:" in summary_messages[1].content
        assert "Line 4:" in summary_messages[1].content
        assert multiline_text in qa_messages[1].content
    
    def test_prompt_with_empty_and_none_values(self):
        """Test prompt rendering with empty and None values."""
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Test with empty string
        summary_messages = summary_prompt.format_messages(deal_text="")
        qa_messages = qa_prompt.format_messages(deal_text="", required_keywords=[])
        
        assert len(summary_messages) == 2
        assert len(qa_messages) == 2
        
        # Empty values should be handled gracefully
        assert summary_messages[1].content is not None
        assert qa_messages[1].content is not None
    
    def test_prompt_with_large_keyword_lists(self):
        """Test prompt rendering with very large keyword lists."""
        large_keywords = [f"keyword{i}" for i in range(100)]
        
        qa_prompt = create_qa_prompt()
        
        # Should handle large keyword lists
        messages = qa_prompt.format_messages(
            deal_text="Test deal content",
            required_keywords=large_keywords
        )
        
        assert len(messages) == 2
        # Keywords should be converted to string representation
        assert "keyword0" in messages[1].content
        assert "keyword99" in messages[1].content
    
    def test_prompt_template_consistency(self):
        """Test that prompt templates are consistent and well-formed."""
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Test that templates have expected structure
        assert len(summary_prompt.messages) == 2
        assert len(qa_prompt.messages) == 2
        
        # First message should be system, second should be human
        # Check the class type instead of type attribute
        assert "SystemMessage" in str(type(summary_prompt.messages[0]))
        assert "HumanMessage" in str(type(summary_prompt.messages[1]))
        assert "SystemMessage" in str(type(qa_prompt.messages[0]))
        assert "HumanMessage" in str(type(qa_prompt.messages[1]))
    
    def test_prompt_variable_validation(self):
        """Test that prompt templates validate required variables."""
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        # Should raise error if required variables are missing
        with pytest.raises(KeyError):
            summary_prompt.format_messages()  # Missing deal_text
        
        with pytest.raises(KeyError):
            qa_prompt.format_messages(deal_text="test")  # Missing required_keywords
        
        with pytest.raises(KeyError):
            qa_prompt.format_messages(required_keywords=["test"])  # Missing deal_text
    
    def test_prompt_output_format_consistency(self):
        """Test that prompt outputs are consistently formatted."""
        summary_prompt = create_summary_prompt()
        qa_prompt = create_qa_prompt()
        
        test_cases = [
            {"deal_text": "Short deal", "keywords": ["deal"]},
            {"deal_text": "Medium length deal with more details", "keywords": ["deal", "details"]},
            {"deal_text": "Very long deal text " * 50, "keywords": ["deal", "text", "long"]}
        ]
        
        for case in test_cases:
            # Summary prompt should always produce 2 messages
            summary_messages = summary_prompt.format_messages(deal_text=case["deal_text"])
            assert len(summary_messages) == 2
            
            # QA prompt should always produce 2 messages
            qa_messages = qa_prompt.format_messages(
                deal_text=case["deal_text"],
                required_keywords=case["keywords"]
            )
            assert len(qa_messages) == 2
            
            # Messages should have content
            assert all(msg.content.strip() for msg in summary_messages)
            assert all(msg.content.strip() for msg in qa_messages)


class TestPromptContentValidation:
    """Test that prompt content meets quality and completeness standards."""
    
    def test_summary_prompt_completeness(self):
        """Test that summary prompt contains all necessary instructions."""
        system_content = SUMMARY_SYSTEM.lower()
        user_content = SUMMARY_USER.lower()
        
        # Check for key instruction elements
        required_elements = [
            "180-220",  # Character limit
            "brand",    # Brand requirement
            "product",  # Product requirement
            "call-to-action",  # CTA requirement
            "emoji",    # Emoji restriction
            "engaging", # Tone requirement
            "concise"   # Style requirement
        ]
        
        for element in required_elements:
            assert element in system_content or element in user_content, f"Missing element: {element}"
    
    def test_qa_prompt_completeness(self):
        """Test that QA prompt contains all validation criteria."""
        system_content = QA_SYSTEM.lower()
        user_content = QA_USER.lower()
        
        # Check for all validation criteria
        required_criteria = [
            "price",      # Price validation
            "eur",        # Currency validation
            "url",        # URL validation
            "http",       # URL format
            "200 words",  # Length validation
            "keyword",    # Keyword validation
            "duplicate",  # Duplicate detection
            "json"        # Output format
        ]
        
        for criterion in required_criteria:
            assert criterion in system_content or criterion in user_content, f"Missing criterion: {criterion}"
    
    def test_prompt_clarity_and_specificity(self):
        """Test that prompts are clear and specific in their instructions."""
        # Summary prompt should be specific about requirements
        assert "must" in SUMMARY_SYSTEM.lower() or "shall" in SUMMARY_SYSTEM.lower()
        assert "exactly one" in SUMMARY_SYSTEM.lower()  # Specific about CTA count
        
        # QA prompt should be specific about validation
        assert "strict" in QA_SYSTEM.lower() or "exact" in QA_SYSTEM.lower()
        assert "schema" in QA_SYSTEM.lower()  # Specific about output format
    
    def test_prompt_consistency_across_templates(self):
        """Test that prompts are consistent in style and terminology."""
        # Both prompts should use similar professional tone
        summary_words = set(SUMMARY_SYSTEM.lower().split())
        qa_words = set(QA_SYSTEM.lower().split())
        
        # Should have some common professional terms
        common_professional_terms = ["must", "requirements", "content", "analysis"]
        found_terms = []
        
        for term in common_professional_terms:
            if term in summary_words or term in qa_words:
                found_terms.append(term)
        
        # Should have at least some professional terminology overlap
        assert len(found_terms) > 0, "Prompts should share some professional terminology"