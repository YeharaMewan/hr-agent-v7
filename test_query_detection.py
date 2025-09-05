#!/usr/bin/env python3
"""
Test to verify if AI text suppression works for employee queries
"""

def test_employee_query_detection():
    """Test the employee query detection logic"""
    
    def is_employee_query(query_text: str) -> bool:
        """Check if the query is likely an employee lookup that will trigger tools"""
        employee_keywords = [
            "employee", "user", "details", "information", "show me", "give me", 
            "find", "tell me about", "who is", "add", "create", "update", 
            "change", "edit", "modify", "department", "role"
        ]
        query_lower = query_text.lower().strip()
        
        # Check for explicit employee keywords
        if any(keyword in query_lower for keyword in employee_keywords):
            return True
        
        # Check if it's likely a single name query (common employee lookup pattern)
        # Single word that's likely a name (capitalized, no special chars, reasonable length)
        if ' ' not in query_lower and len(query_lower) >= 3 and len(query_lower) <= 20:
            # Check if it starts with capital letter (likely name)
            if query_text[0].isupper() and query_text.isalpha():
                return True
        
        return False
    
    test_cases = [
        ("Yehara", True),  # This should be detected as employee query
        ("give me Yehara details", True),
        ("show me Simon information", True),
        ("update John Smith", True),
        ("add new employee", True),
        ("what is the weather?", False),
        ("how are you?", False),
        ("hello", False),
        ("employee", True),
        ("user information", True)
    ]
    
    print("🧪 Testing employee query detection...")
    for query, expected in test_cases:
        result = is_employee_query(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' -> {result} (expected: {expected})")
    
    print("\n🔍 Single name queries that should suppress AI text:")
    single_names = ["Yehara", "John", "Alice", "Simon", "Mike"]
    for name in single_names:
        result = is_employee_query(name)
        status = "❌ PROBLEM" if not result else "✅ Good"
        print(f"{status} '{name}' -> {result}")

if __name__ == "__main__":
    test_employee_query_detection()
