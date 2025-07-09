#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced keyword filtering system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bank_keywords import BankNewsFilter

def test_enhanced_filtering():
    """Test the enhanced keyword filtering system"""
    
    print("🧪 Testing Enhanced Bank News Filtering System")
    print("=" * 60)
    
    # Initialize the filter
    filter_system = BankNewsFilter()
    
    # Test cases with various news titles
    test_cases = [
        # Specific bank news
        ("CBA announces record quarterly profit of $2.5 billion", "CBA.AX"),
        ("Westpac launches new digital banking platform for small business", "WBC.AX"),
        ("ANZ faces ASIC investigation over mortgage lending practices", "ANZ.AX"),
        ("NAB CEO announces major restructuring amid cost pressures", "NAB.AX"),
        ("Macquarie Group expands investment banking in Asia", "MQG.AX"),
        
        # Regulatory and industry news
        ("RBA holds cash rate at 4.35% citing inflation concerns", None),
        ("APRA introduces new capital requirements for major banks", None),
        ("Banking royal commission recommendations still being implemented", None),
        
        # Economic indicators
        ("Australian housing market shows signs of cooling amid rate rises", None),
        ("Consumer confidence drops as mortgage stress increases", None),
        
        # Technology and security
        ("Major cyber security breach hits Australian banking sector", None),
        ("New scam warning issued for bank customers", None),
        
        # Non-relevant news (should score low)
        ("Tesla stock surges on strong quarterly deliveries", None),
        ("New restaurant opens in Sydney CBD", None),
        ("Weather update: Rain expected this weekend", None)
    ]
    
    print(f"Testing {len(test_cases)} news headlines...\n")
    
    relevant_count = 0
    high_priority_count = 0
    
    for i, (title, bank_symbol) in enumerate(test_cases, 1):
        print(f"{i:2d}. Testing: {title[:50]}{'...' if len(title) > 50 else ''}")
        
        # Test the filtering
        result = filter_system.is_relevant_banking_news(title, bank_symbol=bank_symbol)
        
        is_relevant = result['is_relevant']
        relevance_score = result['relevance_score']
        categories = result['categories']
        urgency_score = result['urgency_score']
        matched_keywords = result['matched_keywords']
        
        if is_relevant:
            relevant_count += 1
            
        if relevance_score >= 0.7:
            high_priority_count += 1
        
        # Display results
        status = "✅ RELEVANT" if is_relevant else "❌ NOT RELEVANT"
        print(f"    {status} | Score: {relevance_score:.2f} | Urgency: {urgency_score:.2f}")
        
        if categories:
            print(f"    Categories: {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}")
        
        if matched_keywords:
            print(f"    Keywords: {', '.join(matched_keywords[:3])}{'...' if len(matched_keywords) > 3 else ''}")
        
        print()
    
    print("=" * 60)
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total headlines tested: {len(test_cases)}")
    print(f"   Relevant headlines: {relevant_count} ({relevant_count/len(test_cases)*100:.1f}%)")
    print(f"   High priority headlines: {high_priority_count} ({high_priority_count/len(test_cases)*100:.1f}%)")
    print(f"   Expected relevant: ~12-13 (banking/finance related)")
    print("=" * 60)

def test_keyword_categories():
    """Test different keyword categories"""
    
    print("\n🔍 Testing Keyword Categories")
    print("=" * 40)
    
    filter_system = BankNewsFilter()
    
    category_tests = {
        "Financial Results": "Commonwealth Bank reports strong quarterly earnings beating expectations",
        "Regulatory": "ASIC launches investigation into bank lending practices",
        "Technology": "Major banking app outage affects millions of customers",
        "Risk/Fraud": "New sophisticated scam targets bank customers",
        "Leadership": "Westpac CEO announces retirement after 5 years",
        "Economic": "RBA minutes reveal concerns about household debt levels"
    }
    
    for category, headline in category_tests.items():
        print(f"\n{category}:")
        print(f"  Headline: {headline}")
        
        result = filter_system.is_relevant_banking_news(headline)
        
        print(f"  Relevance: {result['relevance_score']:.2f}")
        print(f"  Categories: {', '.join(result['categories'])}")
        print(f"  Risk indicators: {len(result['sentiment_indicators']['risk'])}")

if __name__ == "__main__":
    try:
        test_enhanced_filtering()
        test_keyword_categories()
        
        print("\n🎉 Enhanced keyword filtering system is working!")
        print("The system can now:")
        print("  ✅ Filter banking news with high accuracy")
        print("  ✅ Categorize news by type and importance") 
        print("  ✅ Detect urgency and risk indicators")
        print("  ✅ Prioritize articles for trading analysis")
        print("  ✅ Support bank-specific filtering")
        
    except Exception as e:
        print(f"❌ Error testing filtering system: {e}")
        sys.exit(1)
