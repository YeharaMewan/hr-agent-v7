#!/usr/bin/env python3
"""
Test the final human-in-the-loop flow to see which tool runs when employee name is provided
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

# Set environment to use PostgreSQL database
os.environ['DATABASE_URL'] = 'postgresql://postgres.ovtkppkbfkdldjkfbetb:tharusha123#@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres'

# Import the main function
from main import query_employee_details_intelligent

def test_employee_name_queries():
    """Test which tool runs when providing employee names"""
    
    print("=" * 70)
    print("TESTING EMPLOYEE NAME QUERIES - TOOL IDENTIFICATION")
    print("=" * 70)
    print("මේ test එකෙන් employee name දුන්නම මොන tool එක run වෙනවද කියල බලමු.")
    print("")
    
    # Test cases with different employee names
    test_cases = [
        {
            "query": "do you know Kalhara Dasanayaka",
            "description": "Do you know + Name"
        },
        {
            "query": "Yehara",
            "description": "Just employee name only"
        },
        {
            "query": "tell me about Hasitha",
            "description": "Tell me about + Name"
        },
        {
            "query": "who is Thavindu",
            "description": "Who is + Name"
        },
        {
            "query": "show me Simon details",
            "description": "Show me + Name + details"
        },
        {
            "query": "find employee John",
            "description": "Find employee + Name"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 50)
        
        try:
            # Call the main tool and analyze the response
            result = query_employee_details_intelligent(test_case['query'])
            
            # Extract key information
            success = result.get('success', False)
            response_type = result.get('response_type', '')
            message = result.get('message', '')
            
            print(f"Tool Used: query_employee_details_intelligent()")
            print(f"Success: {success}")
            print(f"Response Type: {response_type}")
            
            # Analyze what happened
            if response_type == 'human_loop_question':
                print("✅ RESULT: Human-in-the-loop confirmation triggered")
                print("   The system asks user confirmation before showing details")
                
                # Check if it found the employee
                conv_state = result.get('conversation_state', {})
                suggested_data = conv_state.get('suggested_data', {})
                employee = suggested_data.get('employee', {})
                
                if employee:
                    print(f"   Employee Found: {employee.get('name', 'Unknown')}")
                    print(f"   Department: {employee.get('department_name', 'Unknown')}")
                    print(f"   Role: {employee.get('role', 'Unknown')}")
                else:
                    print("   No specific employee found - similarity search or create option")
                
            elif response_type == 'individual_profile':
                print("✅ RESULT: Direct employee profile shown")
                data = result.get('data', {})
                print(f"   Employee: {data.get('name', 'Unknown')}")
                
            elif response_type == 'multiple_matches':
                print("✅ RESULT: Multiple employees found")
                data = result.get('data', [])
                print(f"   Found {len(data)} matching employees")
                
            elif response_type == 'no_results':
                print("❌ RESULT: No employees found")
                
            elif response_type == 'error':
                print("❌ RESULT: Error occurred")
                print(f"   Error: {message}")
                
            else:
                print(f"🔍 RESULT: Other response type - {response_type}")
            
            # Show message preview
            if message:
                print(f"Message Preview: {message[:100]}...")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
        
        print("")
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("📋 TOOL IDENTIFICATION RESULTS:")
    print("")
    print("🎯 PRIMARY TOOL: query_employee_details_intelligent()")
    print("   - This is the MAIN tool that runs when you provide an employee name")
    print("   - It handles ALL employee name queries:")
    print("     • 'do you know [Name]'")
    print("     • 'tell me about [Name]'") 
    print("     • 'who is [Name]'")
    print("     • '[Name]' (just the name)")
    print("     • 'show me [Name] details'")
    print("     • 'find employee [Name]'")
    print("")
    print("🔄 FLOW PROCESS:")
    print("   1. User provides employee name")
    print("   2. query_employee_details_intelligent() tool runs")
    print("   3. Tool searches database for matching employees")
    print("   4. If found: Shows human-in-the-loop confirmation")
    print("   5. If not found: Offers to create new employee")
    print("")
    print("✅ ANSWER TO YOUR QUESTION:")
    print("   Employee නමක් දුන්නම 'query_employee_details_intelligent' tool එක run වෙනවා.")
    print("   මේක තමයි PRIMARY employee search tool එක.")

def test_tool_behavior_detailed():
    """Test detailed behavior of the tool when name is provided"""
    
    print("\n" + "=" * 70)
    print("DETAILED TOOL BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    # Test with a likely non-existent name
    test_name = "NonExistentEmployee123"
    print(f"Testing with non-existent name: {test_name}")
    print("-" * 50)
    
    try:
        result = query_employee_details_intelligent(test_name)
        
        print("TOOL EXECUTION DETAILS:")
        print(f"Tool Called: query_employee_details_intelligent('{test_name}')")
        print(f"Return Type: {type(result)}")
        print(f"Success: {result.get('success')}")
        print(f"Response Type: {result.get('response_type')}")
        
        # Show the internal flow
        print("\nINTERNAL FLOW ANALYSIS:")
        conversation_state = result.get('conversation_state', {})
        if conversation_state:
            pending_action = conversation_state.get('pending_action')
            suggested_data = conversation_state.get('suggested_data', {})
            
            print(f"Pending Action: {pending_action}")
            
            if pending_action == 'create_employee':
                print("→ Tool detected new employee creation needed")
                print(f"→ Suggested name: {suggested_data.get('suggested_name')}")
            elif pending_action == 'show_employee_details':
                print("→ Tool found existing employee")
                employee = suggested_data.get('employee', {})
                print(f"→ Found employee: {employee.get('name')}")
        
        print(f"\nFull Response Message:")
        print(f"'{result.get('message', 'No message')}'")
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")

if __name__ == "__main__":
    test_employee_name_queries()
    test_tool_behavior_detailed()
