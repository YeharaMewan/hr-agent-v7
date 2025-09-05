#!/usr/bin/env python3
"""
Quick test for specific employee name scenarios
"""
import os
os.environ['DATABASE_URL'] = 'postgresql://postgres.ovtkppkbfkdldjkfbetb:tharusha123#@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres'

def quick_test():
    try:
        from main import query_employee_details_intelligent
        
        # Test with a simple name that should trigger individual query
        result = query_employee_details_intelligent("Yehara")
        
        print("QUICK TEST RESULT:")
        print(f"Query: 'Yehara'")
        print(f"Response Type: {result.get('response_type')}")
        print(f"Success: {result.get('success')}")
        
        if result.get('response_type') == 'human_loop_question':
            print("✅ SUCCESS: Correct human-loop flow triggered")
        elif result.get('response_type') == 'large_group_summary':
            print("❌ ISSUE: Still getting large_group_summary")
        else:
            print(f"ℹ️  OTHER: {result.get('response_type')}")
            
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    quick_test()
