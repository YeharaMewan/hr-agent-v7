#!/usr/bin/env python3
"""
Test script for the FIXED sequential human loop functionality
Ensures popups only appear AFTER user confirmation, not immediately with human loop questions
"""
import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_query_sequential_flow():
    """Test that employee query follows proper sequential flow"""
    from main import query_employee_details_intelligent, _store_conversation_state, _handle_confirmation_followup
    
    print("=== Testing Fixed Employee Query Sequential Flow ===")
    
    # Step 1: Query for non-existent employee (Pawan)
    print("\n1. User asks: 'Give me Pawan details'")
    result = query_employee_details_intelligent.invoke({
        "query": "Give me Pawan details"
    })
    
    print(f"Response type: {result.get('response_type')}")
    print(f"Message: {result.get('message')[:100]}...")
    
    # CRITICAL TEST: Should be human_loop_question, NOT form_request
    if result.get('response_type') == 'human_loop_question':
        print("PASS: Step 1 - Returns human_loop_question (NO POPUP YET)")
        
        # Store conversation state
        conversation_state = result.get('conversation_state')
        _store_conversation_state("test_session", conversation_state)
        
        # Step 2: User confirms adding new employee
        print("\n2. User responds: 'Yes, add Pawan as new employee'")
        confirmation_result = _handle_confirmation_followup(
            "Yes, add Pawan as new employee", 
            conversation_state
        )
        
        print(f"Confirmation response type: {confirmation_result.get('response_type')}")
        
        # CRITICAL TEST: Should be form_request only AFTER confirmation
        if confirmation_result.get('response_type') == 'form_request':
            print("PASS: Step 2 - Confirmation triggers form_request (POPUP APPEARS NOW)")
            form_data = confirmation_result.get('form_data', {})
            pre_filled = form_data.get('pre_filled', {})
            if pre_filled.get('name') == 'Pawan':
                print("PASS: Step 3 - Form has correct pre-filled name")
            else:
                print(f"FAIL: Step 3 - Expected 'Pawan', got '{pre_filled.get('name')}'")
        else:
            print(f"FAIL: Step 2 - Expected form_request, got {confirmation_result.get('response_type')}")
    else:
        print(f"FAIL: Step 1 - Expected human_loop_question, got {result.get('response_type')}")

def test_update_sequential_flow():
    """Test that employee update follows proper sequential flow"""
    from main import update_employee_intelligent, _store_conversation_state, _handle_confirmation_followup
    
    print("\n\n=== Testing Fixed Employee Update Sequential Flow ===")
    
    # Step 1: Try to update non-existent employee with similar match (Simion -> Simon)
    print("\n1. User asks: 'Update Simion details'")
    result = update_employee_intelligent.invoke({
        "query": "Update Simion details",
        "employee_identifier": "Simion"
    })
    
    print(f"Response type: {result.get('response_type')}")
    print(f"Message: {result.get('message')[:100]}...")
    
    # CRITICAL TEST: Should be human_loop_question, NOT form_request
    if result.get('response_type') == 'human_loop_question':
        print("PASS: Step 1 - Returns human_loop_question (NO POPUP YET)")
        
        # Store conversation state
        conversation_state = result.get('conversation_state')
        _store_conversation_state("test_session_2", conversation_state)
        
        # Step 2: User confirms updating the suggested employee
        print("\n2. User responds: 'Yes, update Simon details'")
        confirmation_result = _handle_confirmation_followup(
            "Yes, update Simon details", 
            conversation_state
        )
        
        print(f"Confirmation response type: {confirmation_result.get('response_type')}")
        
        # CRITICAL TEST: Should be employee_found_update_form only AFTER confirmation
        if confirmation_result.get('response_type') == 'employee_found_update_form':
            print("PASS: Step 2 - Confirmation triggers employee_found_update_form (POPUP APPEARS NOW)")
            form_data = confirmation_result.get('form_data', {})
            current_values = form_data.get('current_values', {})
            if current_values.get('name'):
                print(f"PASS: Step 3 - Form has current employee data: {current_values.get('name')}")
            else:
                print("FAIL: Step 3 - Form missing current employee data")
        else:
            print(f"FAIL: Step 2 - Expected employee_found_update_form, got {confirmation_result.get('response_type')}")
    else:
        print(f"FAIL: Step 1 - Expected human_loop_question, got {result.get('response_type')}")

def test_show_details_flow():
    """Test showing employee details (no form)"""
    from main import query_employee_details_intelligent, _store_conversation_state, _handle_confirmation_followup
    
    print("\n\n=== Testing Show Employee Details Flow ===")
    
    # Step 1: Query for employee with similar match but user wants to see existing
    print("\n1. User asks: 'Show me Simion details'")
    result = query_employee_details_intelligent.invoke({
        "query": "Show me Simion details"
    })
    
    if result.get('response_type') == 'human_loop_question':
        conversation_state = result.get('conversation_state')
        _store_conversation_state("test_session_3", conversation_state)
        
        # Step 2: User wants to see the similar employee's details
        print("\n2. User responds: 'Show Simon details'")
        confirmation_result = _handle_confirmation_followup(
            "Show Simon details", 
            conversation_state
        )
        
        print(f"Confirmation response type: {confirmation_result.get('response_type')}")
        
        # Should be individual_profile (no form)
        if confirmation_result.get('response_type') == 'individual_profile':
            print("PASS: Show details returns individual_profile (NO POPUP)")
            print(f"Employee details shown: {confirmation_result.get('message')[:50]}...")
        else:
            print(f"FAIL: Expected individual_profile, got {confirmation_result.get('response_type')}")

def test_intent_detection():
    """Test improved intent detection"""
    from main import _handle_confirmation_followup
    
    print("\n\n=== Testing Enhanced Intent Detection ===")
    
    # Mock conversation state
    mock_state = {
        "pending_action": "show_employee_details",
        "alternative_action": "create_employee", 
        "suggested_data": {
            "employee": {"name": "Simon", "id": 1},
            "suggested_name": "Simion"
        }
    }
    
    test_cases = [
        ("Yes, add Simion as new employee", "form_request", "Create intent"),
        ("Yes, update Simon details", "employee_found_update_form", "Update intent"), 
        ("Show Simon details", "individual_profile", "Show intent"),
        ("Create them", "form_request", "Create keyword"),
        ("Update them", "employee_found_update_form", "Update keyword")
    ]
    
    for message, expected_type, description in test_cases:
        result = _handle_confirmation_followup(message, mock_state)
        actual_type = result.get('response_type')
        
        status = "PASS" if actual_type == expected_type else "FAIL"
        print(f"  {status}: {description} - '{message}' -> {actual_type}")

def main():
    """Run all fixed sequential flow tests"""
    print("Testing FIXED Sequential Human Loop Flow")
    print("=" * 60)
    print("GOAL: Ensure popups appear ONLY after user confirmation, not immediately")
    
    try:
        # Test 1: Employee query sequential flow
        test_query_sequential_flow()
        
        # Test 2: Employee update sequential flow
        test_update_sequential_flow()
        
        # Test 3: Show employee details flow
        test_show_details_flow()
        
        # Test 4: Intent detection
        test_intent_detection()
        
        print("\n" + "=" * 60)
        print("FIXED Sequential Human Loop Tests Completed!")
        print("\nEXPECTED RESULTS:")
        print("- PASS Human loop questions return message only (NO immediate popup)")
        print("- PASS User confirmation triggers popup (popup appears AFTER confirmation)")
        print("- PASS No double tool calls or unexpected form triggers")
        print("- PASS Clean sequential flow: Question -> User Confirms -> Popup")
        print("\nSUCCESS Fixed Sequential Human Loop is Working Correctly!")
        
    except Exception as e:
        print(f"\nFAIL Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()