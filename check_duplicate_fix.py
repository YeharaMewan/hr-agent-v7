#!/usr/bin/env python3
"""
Simple test to check if duplicate message issue is fixed by examining the code
"""

import re

def check_for_duplicates_in_code():
    """Check the main.py file for potential duplicate message patterns"""
    
    with open('backend/main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔍 Checking for potential duplicate message patterns in code...")
    
    # Find all message strings that contain confirmation patterns
    confirmation_patterns = [
        r'"message":\s*f?"[^"]*Do you want the employee details[^"]*"',
        r'"message":\s*f?"[^"]*I know this employee[^"]*"',
        r'"message":\s*f?"[^"]*who is working in our company[^"]*"'
    ]
    
    for i, pattern in enumerate(confirmation_patterns):
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        print(f"\nPattern {i+1}: Looking for duplicate confirmation phrases")
        print(f"Found {len(matches)} instances:")
        
        for j, match in enumerate(matches):
            # Clean up the match for display
            clean_match = match.replace('\n', ' ').strip()
            if len(clean_match) > 100:
                clean_match = clean_match[:100] + "..."
            print(f"  {j+1}. {clean_match}")
    
    # Check for specific problematic phrases
    problematic_phrases = [
        "Do you want the employee details of that employee?",
        "I know this employee who is working in our company. They work",
        "Yes, I know this employee"
    ]
    
    print("\n" + "="*60)
    print("🚨 Checking for specific problematic duplicate phrases:")
    print("="*60)
    
    for phrase in problematic_phrases:
        count = content.count(phrase)
        if count > 0:
            print(f"❌ Found '{phrase}' - {count} times")
        else:
            print(f"✅ '{phrase}' - not found (good!)")
    
    # Check for cleaner messages
    clean_patterns = [
        "Yes, I found **{employee['name']}** who is working in our company. They work",
        "Yes, I found **{best_match['name']}** in our database who closely matches",
        "I couldn't find '{employee_name}' exactly, but found similar names"
    ]
    
    print("\n" + "="*60)
    print("✅ Checking for clean message patterns:")
    print("="*60)
    
    for pattern in clean_patterns:
        # Convert f-string pattern to regex
        regex_pattern = re.escape(pattern).replace(r'\{[^}]+\}', r'[^"]*')
        if re.search(regex_pattern, content):
            print(f"✅ Found clean pattern: {pattern}")
        else:
            print(f"❓ Clean pattern not found: {pattern}")

if __name__ == "__main__":
    print("🧪 Testing duplicate message fix by code analysis...")
    check_for_duplicates_in_code()
    print("\n✅ Code analysis completed!")
