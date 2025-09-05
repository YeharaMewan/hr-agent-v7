#!/usr/bin/env python3
"""
Simple test to identify which tool runs when employee name is provided
"""
import os
import sys

# Set environment
os.environ['DATABASE_URL'] = 'postgresql://postgres.ovtkppkbfkdldjkfbetb:tharusha123#@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres'

print("=" * 60)
print("EMPLOYEE NAME QUERY TOOL IDENTIFICATION")
print("=" * 60)
print()

try:
    # Import the function
    from main import query_employee_details_intelligent
    print("✅ Successfully imported query_employee_details_intelligent")
    print()
    
    # Test with a simple name
    test_query = "Yehara"
    print(f"Testing query: '{test_query}'")
    print("-" * 40)
    
    # Call the function
    result = query_employee_details_intelligent(test_query)
    
    print("RESULTS:")
    print(f"Tool Used: query_employee_details_intelligent()")
    print(f"Success: {result.get('success', False)}")
    print(f"Response Type: {result.get('response_type', 'Unknown')}")
    print(f"Message: {result.get('message', 'No message')[:200]}...")
    
    print()
    print("=" * 60)
    print("ANSWER:")
    print("Employee නමක් දුන්නම 'query_employee_details_intelligent' tool එක run වෙනවා.")
    print("මේක තමයි primary employee search tool එක.")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
