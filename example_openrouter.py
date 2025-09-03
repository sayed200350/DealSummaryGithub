#!/usr/bin/env python3
"""
Example script showing how to use the Deal Summary & QA Bot with OpenRouter.

OpenRouter provides access to various AI models, including free tiers.
Sign up at https://openrouter.ai/ to get your API key.
"""

import os
from src.pipeline import DealAnalysisPipeline

def main():
    # Example deal text
    deal_text = """
    🔥 AMAZING LAPTOP DEAL! 🔥
    
    Get 50% OFF on premium gaming laptops at TechStore!
    Originally €1,200, now only €600!
    
    ✅ Free shipping included
    ✅ 2-year warranty
    ✅ Limited time offer - ends tonight!
    
    Shop now: https://techstore.com/gaming-laptops
    Use code: SAVE50
    """
    
    # OpenRouter configuration (now default)
    # You can get a free API key at https://openrouter.ai/
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   Get your free key at: https://openrouter.ai/")
        print("   Or set OPENAI_API_KEY (also supported)")
        return
    
    try:
        print("🤖 Initializing with OpenRouter (default configuration)...")
        
        # Initialize pipeline - OpenRouter is now default, so minimal config needed
        pipeline = DealAnalysisPipeline(
            api_key=api_key,
            model_name="meta-llama/llama-3.2-3b-instruct:free"  # Free model
            # base_url is automatically set to OpenRouter by default
        )
        
        print("📝 Analyzing deal content...")
        
        # Analyze the deal
        result = pipeline.analyze_deal(
            deal_text=deal_text,
            required_keywords=["deal", "%", "euro", "laptop"]
        )
        
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📝 Generated Summary ({len(result['summary'])} chars):")
        print(f"   {result['summary']}")
        
        print(f"\n🔍 Quality Assurance:")
        qa = result['qa']
        print(f"   ✅ Has Price: {qa.has_price} ({qa.price_value}€ if found)")
        print(f"   🔗 Has Source URL: {qa.has_source_url}")
        print(f"   📏 Within Length Limit: {qa.within_length_limit}")
        print(f"   🏷️  Missing Keywords: {qa.missing_keywords or 'None'}")
        print(f"   🔄 Duplicate Suspect: {qa.duplicate_suspect}")
        
        if qa.notes:
            print(f"   📋 Notes: {qa.notes}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()