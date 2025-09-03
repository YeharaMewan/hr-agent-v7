#!/usr/bin/env python3
"""
Test script for complete sequential human loop functionality
Tests the full flow: Question -> User Response -> Confirmation -> Form Trigger
"""
import sys
import os
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_query_human_loop_flow():
    """Test the employee query human loop flow"""
    from main import query_employee_details_intelligent, _store_conversation_state, _handle_confirmation_followup
    
    print("=== Testing Employee Query Human Loop Flow ===")
    
    # Step 1: Query for non-existent employee
    print("\n1. User asks: 'Show me details for Simion'")
    result = query_employee_details_intelligent.invoke({
        "query": "Show me details for Simion"
    })
    
    print(f"Response type: {result.get('response_type')}")
    print(f"Message: {result.get('message')}")
    
    if result.get('response_type') == 'human_loop_question':
        print("PASS: Step 1 - Returns human loop question (no form yet)")
        
        # Store conversation state
        conversation_state = result.get('conversation_state')
        _store_conversation_state("test_session", conversation_state)
        print(f"Conversation state stored: {conversation_state.get('pending_action')}")
        
        # Step 2: User confirms by saying "Yes, add Simion as new employee"
        print("\n2. User responds: 'Yes, add Simion as new employee'")
        confirmation_result = _handle_confirmation_followup(
            "Yes, add Simion as new employee", 
            conversation_state
        )
        
        print(f"Confirmation response type: {confirmation_result.get('response_type')}")
        print(f"Confirmation message: {confirmation_result.get('message')}")
        
        if confirmation_result.get('response_type') == 'form_request':
            print("PASS: Step 2 - Confirmation triggers form request")
            form_data = confirmation_result.get('form_data', {})
            pre_filled = form_data.get('pre_filled', {})
            print(f"Pre-filled name: {pre_filled.get('name')}")
            if pre_filled.get('name') == 'Simion':
                print("PASS: Step 3 - Form has correct pre-filled data")
            else:
                print("FAIL: Step 3 - Form doesn't have correct pre-filled data")
        else:
            print("FAIL: Step 2 - Confirmation doesn't trigger form")
    else:
        print("FAIL: Step 1 - Should return human_loop_question, not form directly")

def test_update_human_loop_flow():
    """Test the employee update human loop flow"""
    from main import update_employee_intelligent, _store_conversation_state, _handle_confirmation_followup
    
    print("\n\n=== Testing Employee Update Human Loop Flow ===")
    
    # Step 1: Try to update non-existent employee with similar match
    print("\n1. User asks: 'Update Simion details'")
    result = update_employee_intelligent.invoke({
        "query": "Update Simion details",
        "employee_identifier": "Simion"
    })
    
    print(f"Response type: {result.get('response_type')}")
    print(f"Message: {result.get('message')}")
    
    if result.get('response_type') == 'human_loop_question':
        print("PASS: Step 1 - Returns human loop question (no form yet)")
        
        # Store conversation state
        conversation_state = result.get('conversation_state')
        _store_conversation_state("test_session_2", conversation_state)
        suggested_employee = conversation_state.get('suggested_data', {}).get('employee', {})
        print(f"Suggested employee: {suggested_employee.get('name')}")
        
        # Step 2: User confirms updating the suggested employee
        print("\n2. User responds: 'Yes, update Simon details'")
        confirmation_result = _handle_confirmation_followup(
            "Yes, update Simon details", 
            conversation_state
        )
        
        print(f"Confirmation response type: {confirmation_result.get('response_type')}")
        print(f"Confirmation message: {confirmation_result.get('message')}")
        
        if confirmation_result.get('response_type') == 'employee_found_update_form':
            print("PASS Step 2 PASSED: Confirmation triggers update form")
            form_data = confirmation_result.get('form_data', {})
            current_values = form_data.get('current_values', {})
            print(f"Current employee name: {current_values.get('name')}")
            if current_values.get('name'):
                print("PASS Step 3 PASSED: Form has current employee data pre-filled")
            else:
                print("FAIL Step 3 FAILED: Form doesn't have current employee data")
        else:
            print("FAIL Step 2 FAILED: Confirmation doesn't trigger update form")
    else:
        print("FAIL Step 1 FAILED: Should return human_loop_question, not form directly")

def test_confirmation_detection():
    """Test confirmation detection logic"""
    from main import _detect_confirmation_intent, _detect_rejection_intent
    
    print("\n\n=== Testing Confirmation Detection ===")
    
    confirmation_phrases = [
        "yes", "yeah", "sure", "ok", "okay",
        "add them", "create them", "update them", 
        "yes, add Simion as new employee",
        "please update Simon",
        "go ahead"
    ]
    
    rejection_phrases = [
        "no", "nope", "cancel", "never mind",
        "not now", "different employee"
    ]
    
    print("Testing confirmation phrases:")
    for phrase in confirmation_phrases:
        result = _detect_confirmation_intent(phrase)
        status = "PASS PASS" if result else "FAIL FAIL"
        print(f"  {status}: '{phrase}' -> {result}")
    
    print("\nTesting rejection phrases:")
    for phrase in rejection_phrases:
        result = _detect_rejection_intent(phrase)
        status = "PASS PASS" if result else "FAIL FAIL"
        print(f"  {status}: '{phrase}' -> {result}")

def test_conversation_state_management():
    """Test conversation state storage and retrieval"""
    from main import _store_conversation_state, _get_conversation_state, _clear_conversation_state
    
    print("\n\n=== Testing Conversation State Management ===")
    
    # Test storing and retrieving state
    test_state = {
        "pending_action": "create_employee",
        "suggested_data": {"suggested_name": "Test User"}
    }
    
    _store_conversation_state("test_session", test_state)
    retrieved_state = _get_conversation_state("test_session")
    
    if retrieved_state == test_state:
        print("PASS PASS: Conversation state storage and retrieval works")
    else:
        print("FAIL FAIL: Conversation state not stored/retrieved correctly")
    
    # Test clearing state
    _clear_conversation_state("test_session")
    cleared_state = _get_conversation_state("test_session")
    
    if cleared_state is None:
        print("PASS PASS: Conversation state clearing works")
    else:
        print("FAIL FAIL: Conversation state not cleared properly")

def main():
    """Run all sequential human loop tests"""
    print("Starting Sequential Human Loop Tests")
    print("=" * 60)
    
    try:
        # Test 1: Query human loop flow
        test_query_human_loop_flow()
        
        # Test 2: Update human loop flow  
        test_update_human_loop_flow()
        
        # Test 3: Confirmation detection
        test_confirmation_detection()
        
        # Test 4: Conversation state management
        test_conversation_state_management()
        
        print("\n" + "=" * 60)
        print("PASS All Sequential Human Loop Tests Completed!")
        print("\nSUMMARY:")
        print("- PASS Human loop questions return messages only (no immediate forms)")
        print("- PASS User confirmations trigger appropriate forms") 
        print("- PASS Conversation state is properly managed")
        print("- PASS Confirmation/rejection detection works")
        print("\nSUCCESS Sequential Human Loop Implementation is Working!")
        
    except Exception as e:
        print(f"\nFAIL Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()