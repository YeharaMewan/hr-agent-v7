#!/usr/bin/env python3
"""
Direct test of update employee functionality without LangChain tool wrapper
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function directly
from main import update_employee_intelligent

def test_update_direct():
    """Test update employee function directly"""
    print("=== Testing Employee Update Direct ===")
    
    test_cases = [
        {
            "query": "Update Simion details", 
            "identifier": "Simion", 
            "expected": "Should suggest Simon"
        },
        {
            "query": "Update Johnny info", 
            "identifier": "Johnny", 
            "expected": "Should suggest John"
        },
        {
            "query": "Change Mike's role", 
            "identifier": "Mike", 
            "expected": "Should not find similar, offer create"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: '{test_case['query']}' (identifier: {test_case['identifier']})")
        print(f"Expected: {test_case['expected']}")
        try:
            # Call the tool using the invoke method for LangChain tools
            result = update_employee_intelligent.invoke({
                "query": test_case["query"],
                "employee_identifier": test_case["identifier"]
            })
            print(f"Response type: {result.get('response_type')}")
            print(f"Message: {result.get('message')}")
            if result.get('suggested_employee'):
                suggested_name = result.get('suggested_employee', {}).get('name')
                print(f"Suggested: {suggested_name}")
            if result.get('all_similar'):
                print(f"All similar employees: {len(result.get('all_similar', []))}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_update_direct()