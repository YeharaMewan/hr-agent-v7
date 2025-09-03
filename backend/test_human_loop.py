#!/usr/bin/env python3
"""
Test script for human loop functionality in employee management
"""
import sqlite3
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_string_similarity():
    """Test the string similarity function"""
    from main import _calculate_string_similarity
    
    print("=== Testing String Similarity Function ===")
    
    test_cases = [
        ("Simon", "Simion", "Should be high similarity"),
        ("John", "Jon", "Should be medium similarity"),  
        ("Alice", "Alicia", "Should be medium similarity"),
        ("Bob", "Robert", "Should be low similarity"),
        ("Test", "Test", "Should be identical (1.0)"),
        ("Test", "xyz", "Should be low similarity")
    ]
    
    for str1, str2, description in test_cases:
        similarity = _calculate_string_similarity(str1, str2)
        print(f"{description}: '{str1}' vs '{str2}' = {similarity:.3f}")
    
    return True

def setup_test_database():
    """Create a test database with sample employees"""
    print("\n=== Setting up Test Database ===")
    
    # Create tables and insert test data
    conn = sqlite3.connect('hr_management.db')
    cursor = conn.cursor()
    
    # Insert test employees if they don't exist
    test_employees = [
        ("Simon Johnson", "simon.johnson@company.com", "AI", "leader"),
        ("Alice Smith", "alice.smith@company.com", "Marketing", "employee"),
        ("John Doe", "john.doe@company.com", "HR", "hr"),
        ("Bob Wilson", "bob.wilson@company.com", "Operations", "employee")
    ]
    
    for name, email, dept, role in test_employees:
        # Check if employee already exists
        cursor.execute("SELECT id FROM employees WHERE email = ?", (email,))
        if not cursor.fetchone():
            # Get department ID
            cursor.execute("SELECT id FROM departments WHERE name = ?", (dept,))
            dept_result = cursor.fetchone()
            if dept_result:
                dept_id = dept_result[0]
                cursor.execute("""
                    INSERT INTO employees (name, email, department_id, role, is_active, created_at)
                    VALUES (?, ?, ?, ?, 1, datetime('now'))
                """, (name, email, dept_id, role))
                print(f"Added test employee: {name}")
    
    conn.commit()
    conn.close()
    return True

def test_query_employee_not_found():
    """Test query_employee_details_intelligent with non-existent employees"""
    from main import query_employee_details_intelligent
    
    print("\n=== Testing Employee Query - Not Found Scenarios ===")
    
    test_queries = [
        "Show me details for Simion",  # Similar to Simon
        "Give me info about Johnny",   # Similar to John
        "Tell me about Jake Smith",    # Non-existent
        "Find employee Mike Johnson",  # Non-existent
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            result = query_employee_details_intelligent(query)
            print(f"Response type: {result.get('response_type')}")
            print(f"Message: {result.get('message')}")
            if result.get('data'):
                print(f"Similar employees found: {len(result.get('data', []))}")
        except Exception as e:
            print(f"Error: {e}")
    
    return True

def test_update_employee_not_found():
    """Test update_employee_intelligent with non-existent employees"""
    from main import update_employee_intelligent
    
    print("\n=== Testing Employee Update - Not Found Scenarios ===")
    
    test_cases = [
        ("Update Simion details", "Simion", "Should suggest Simon"),
        ("Update Johnny info", "Johnny", "Should suggest John"),
        ("Change Mike's role", "Mike", "Should not find similar, offer create"),
    ]
    
    for query, identifier, expected in test_cases:
        print(f"\nTesting: '{query}' (identifier: {identifier})")
        print(f"Expected: {expected}")
        try:
            result = update_employee_intelligent(query, identifier)
            print(f"Response type: {result.get('response_type')}")
            print(f"Message: {result.get('message')}")
            if result.get('suggested_employee'):
                print(f"Suggested: {result.get('suggested_employee', {}).get('name')}")
        except Exception as e:
            print(f"Error: {e}")
    
    return True

def main():
    """Run all tests"""
    print("Starting Human Loop Functionality Tests")
    print("=" * 50)
    
    try:
        # Test 1: String similarity
        test_string_similarity()
        
        # Test 2: Setup test database  
        setup_test_database()
        
        # Test 3: Query employee not found
        test_query_employee_not_found()
        
        # Test 4: Update employee not found
        test_update_employee_not_found()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()