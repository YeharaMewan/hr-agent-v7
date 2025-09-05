#!/usr/bin/env python3
"""
Test the employee query detection logic
"""

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
    # Single word that's likely a name (no special chars, reasonable length)
    if ' ' not in query_lower and len(query_lower) >= 4 and len(query_lower) <= 20:
        # Check if it's alphabetic only (likely name) - accept both cases
        # But exclude common non-name words
        common_words = ["hello", "hi", "hey", "yes", "no", "ok", "okay", "thanks", "help"]
        if query_text.isalpha() and query_lower not in common_words:
            return True
    
    return False

# Test cases
test_queries = [
    "kalhara",
    "Kalhara", 
    "deshan",
    "Deshan",
    "yehara",
    "Yehara",
    "show me John details",
    "find employee Sarah",
    "hello",
    "what is the weather",
    "department info"
]

print("Testing employee query detection:")
print("="*50)

for query in test_queries:
    result = is_employee_query(query)
    print(f"'{query}' -> {result}")
