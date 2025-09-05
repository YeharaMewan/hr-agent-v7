#!/usr/bin/env python3
"""
Test to verify employee name query flow works correctly 
(නම් දුන්නම හරිම ආකාරයට human-in-the-loop confirmation එක වෙනවද කියලා test කරන්න)
"""
import os
import sys

# Set environment
os.environ['DATABASE_URL'] = 'postgresql://postgres.ovtkppkbfkdldjkfbetb:tharusha123#@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres'

print("=" * 80)
print("EMPLOYEE NAME QUERY FLOW TEST")
print("=" * 80)
print("Testing to ensure individual employee names get proper human-in-the-loop confirmation")
print("(නම් දුන්නම large_group_summary නොව human-loop-question වෙනවද බලන්න)")
print()

try:
    from main import query_employee_details_intelligent
    
    # Test cases for individual employee names
    test_cases = [
        "Yehara",           # Simple name
        "John Smith",       # Full name  
        "do you know Sarah", # Question format
        "tell me about Mike", # Info request
        "who is David"      # Who is format
    ]
    
    for test_query in test_cases:
        print(f"🔍 Testing: '{test_query}'")
        print("-" * 40)
        
        result = query_employee_details_intelligent(test_query)
        
        response_type = result.get('response_type', 'unknown')
        success = result.get('success', False)
        message = result.get('message', '')
        
        print(f"Success: {success}")
        print(f"Response Type: {response_type}")
        
        # Check if it's the correct flow
        if response_type == 'human_loop_question':
            print("✅ CORRECT: Human-in-the-loop confirmation triggered")
            
            # Check what action is pending
            conv_state = result.get('conversation_state', {})
            pending_action = conv_state.get('pending_action', '')
            
            if pending_action == 'show_employee_details':
                print("✅ CORRECT: Will show employee details after confirmation")
                employee = conv_state.get('suggested_data', {}).get('employee', {})
                if employee:
                    print(f"   Found Employee: {employee.get('name', 'Unknown')}")
                    print(f"   Department: {employee.get('department_name', 'Unknown')}")
                else:
                    print("   Similar employee suggested for confirmation")
                    
            elif pending_action == 'create_employee':
                print("✅ CORRECT: Will create new employee after confirmation")
                suggested_name = conv_state.get('suggested_data', {}).get('suggested_name', '')
                print(f"   Suggested Name: {suggested_name}")
                
        elif response_type == 'large_group_summary':
            print("❌ WRONG: Got large_group_summary instead of human-loop confirmation")
            print("   This should not happen for individual employee names!")
            
        elif response_type == 'individual_profile':
            print("❌ WRONG: Got direct profile without confirmation")
            print("   Should ask for confirmation first!")
            
        elif response_type == 'no_results':
            print("ℹ️  INFO: No employees found")
            
        else:
            print(f"⚠️  UNKNOWN: Response type '{response_type}'")
        
        # Show message preview
        if message:
            print(f"Message: {message[:120]}...")
        
        print()
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ If most tests show 'human_loop_question', the flow is working correctly")
    print("❌ If any show 'large_group_summary', that needs to be fixed")
    print()
    print("EXPECTED BEHAVIOR:")
    print("1. User provides employee name")
    print("2. System searches for employee")
    print("3. If found: Ask 'Do you want employee details?' (human_loop_question)")
    print("4. If not found: Ask 'Do you want to create employee?' (human_loop_question)")
    print("5. User confirms, then system shows details or opens form")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
