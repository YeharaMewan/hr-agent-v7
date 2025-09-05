#!/usr/bin/env python3
"""
Test script to verify the duplicate message fix works correctly
"""

import sys
import os
import json

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

try:
    from main import query_employee_details_intelligent
    print("✓ Successfully imported the function")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_single_employee_query():
    """Test that a simple employee name query shows clean confirmation message without duplication"""
    print("\n" + "="*50)
    print("Testing: Single employee query message cleanliness")
    print("="*50)
    
    test_cases = [
        "Yehara",
        "John",
        "Alice"
    ]
    
    for employee_name in test_cases:
        print(f"\n--- Testing query: '{employee_name}' ---")
        
        try:
            result = query_employee_details_intelligent(employee_name)
            
            if result.get("success"):
                message = result.get("message", "")
                response_type = result.get("response_type", "")
                
                print(f"Response type: {response_type}")
                print(f"Message: {message}")
                
                # Check for duplicate phrases
                duplicate_checks = [
                    ("Do you want the employee details", message.count("Do you want the employee details")),
                    ("who is working in our company", message.count("who is working in our company")),
                    ("I know this employee", message.count("I know this employee"))
                ]
                
                duplicates_found = False
                for phrase, count in duplicate_checks:
                    if count > 1:
                        print(f"⚠️  DUPLICATE FOUND: '{phrase}' appears {count} times")
                        duplicates_found = True
                    elif count == 1:
                        print(f"✓ '{phrase}' appears once (good)")
                    else:
                        print(f"ℹ️  '{phrase}' not found in message")
                
                if not duplicates_found:
                    print("✅ No duplicate phrases detected!")
                else:
                    print("❌ Duplicate phrases found - needs fixing")
                    
                # Check follow-up questions
                follow_ups = result.get("follow_up_questions", [])
                print(f"Follow-up questions: {follow_ups}")
                
            else:
                print(f"Query failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error testing '{employee_name}': {e}")

def test_similar_employee_query():
    """Test that similar name suggestions don't have duplicate messages"""
    print("\n" + "="*50)
    print("Testing: Similar employee query message cleanliness")
    print("="*50)
    
    # Try a name that should trigger similarity search
    test_name = "Yehara123NotFound"
    
    print(f"\n--- Testing similar name query: '{test_name}' ---")
    
    try:
        result = query_employee_details_intelligent(test_name)
        
        if result.get("success"):
            message = result.get("message", "")
            response_type = result.get("response_type", "")
            
            print(f"Response type: {response_type}")
            print(f"Message: {message}")
            
            # Check for duplicate phrases in similarity search
            duplicate_checks = [
                ("Do you want the employee details", message.count("Do you want the employee details")),
                ("I know this employee", message.count("I know this employee")),
                ("Did you mean", message.count("Did you mean"))
            ]
            
            duplicates_found = False
            for phrase, count in duplicate_checks:
                if count > 1:
                    print(f"⚠️  DUPLICATE FOUND: '{phrase}' appears {count} times")
                    duplicates_found = True
                elif count == 1:
                    print(f"✓ '{phrase}' appears once (good)")
                else:
                    print(f"ℹ️  '{phrase}' not found in message")
            
            if not duplicates_found:
                print("✅ No duplicate phrases detected in similarity search!")
            else:
                print("❌ Duplicate phrases found in similarity search - needs fixing")
                
        else:
            print(f"Query failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error testing similar name query: {e}")

if __name__ == "__main__":
    print("🧪 Testing duplicate message fix...")
    
    test_single_employee_query()
    test_similar_employee_query()
    
    print("\n" + "="*50)
    print("✅ Duplicate message fix testing completed!")
    print("="*50)
