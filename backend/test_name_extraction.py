#!/usr/bin/env python3
"""
Test name extraction logic without database
"""
import re

def test_name_extraction():
    """Test the name extraction logic from the main function"""
    
    # Copy the patterns from main.py
    name_patterns = [
        # Knowledge-based query patterns - prioritize these first
        r'do\s+you\s+know\s+(?:the\s+name\s+of\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        r'(?:do\s+you\s+)?know\s+(?:about\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        r'familiar\s+with\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        r'heard\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        r'who\s+is\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        r'does\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)\s+work\s+here',
        r'is\s+there\s+(?:a|an|someone\s+named\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$|\?)',
        
        # Direct information query patterns
        r'give\s+me\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s+details|\s+information|\s*$)',
        r'show\s+me\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s+details|\s+information|\s*$)', 
        r'tell\s+me\s+about\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s+details|\s+information|\s*$)',
        r'find\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s+details|\s+information|\s*$)',
        
        # General patterns
        r'(?:details|information|info)\s+(?:for|about|of)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$)',
        r'([A-Za-z]+(?:\s+[A-Za-z]+)*?)\s+(?:details|information|info)(?:\s|$)',
        
        # Quoted names
        r'"([^"]+)"',
        r"'([^']+)'",
        
        # NEW: Single name patterns - For simple queries like "kalhara", "simon", etc.
        r'^([A-Za-z]{2,20})$',  # Single word that looks like a name
        r'^([A-Za-z]{2,20}\s+[A-Za-z]{2,20})$',  # Two words (first + last name)
        r'^([A-Za-z]{2,20}\s+[A-Za-z]{2,20}\s+[A-Za-z]{2,20})$',  # Three words (full name)
        
        # Fallback - single word or simple names
        r'\b([A-Za-z]{3,})\b(?=\s+(?:details|information|info)|$)'
    ]
    
    def normalize_name(name: str) -> str:
        """Normalize name for consistent matching"""
        if not name:
            return ""
        # Clean up the name: strip whitespace, remove extra spaces
        cleaned = re.sub(r'\s+', ' ', name.strip())
        # Title case for consistency (first letter of each word capitalized)
        normalized = cleaned.title()
        return normalized
    
    # Test cases
    test_queries = [
        "Yehara",
        "John Smith", 
        "do you know Sarah",
        "tell me about Mike",
        "who is David",
        "show me all employees",  # Should NOT extract name
        "AI department employees",          # Should NOT extract name
        "marketing team",        # Should NOT extract name
    ]
    
    print("NAME EXTRACTION TEST")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Determine query type 
        query_lower = query.lower().strip()
        
        # First check if it's a simple name pattern (1-3 words that look like names)
        simple_name_pattern = r'^([A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20}){0,2})$'
        simple_name_match = re.match(simple_name_pattern, query.strip())
        
        if simple_name_match:
            # This looks like a simple name query
            query_type = "individual"
        elif any(phrase in query_lower for phrase in ['all employee', 'everyone', 'show me all', 'entire company']):
            query_type = "all_employees"
        elif any(phrase in query_lower for phrase in [
            'details', 'information', 'info about', 'give me',
            'do you know', 'know', 'familiar with', 'heard of',
            'who is', 'is there', 'does', 'work here',
            'tell me about', 'show me', 'find'
        ]) and not any(dept in query_lower for dept in ['department', 'team']):
            query_type = "individual"
        elif any(phrase in query_lower for phrase in ['department', 'team', 'dept']):
            query_type = "department" 
        elif any(phrase in query_lower for phrase in ['ai', 'marketing', 'hr', 'finance']):
            query_type = "department"  # Department names 
        else:
            query_type = "general"
        
        print(f"Query Type: {query_type}")
        
        # Extract name if individual
        employee_name = None
        if query_type == "individual":
            # For simple name queries, try direct extraction first
            if simple_name_match:
                # This is a simple name query like "Yehara" or "John Smith"
                potential_name = simple_name_match.group(1).strip()
                exclude_words = ['details', 'information', 'info', 'employee', 'staff', 'person', 'data', 'profile', 'all', 'everyone']
                if not any(word in potential_name.lower() for word in exclude_words):
                    employee_name = normalize_name(potential_name)
            
            # If no name extracted yet, try pattern matching
            if not employee_name:
                for pattern in name_patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        potential_name = match.group(1).strip()
                        # Filter out common words that aren't names
                        exclude_words = ['details', 'information', 'info', 'employee', 'staff', 'person', 'data', 'profile']
                        if not any(word in potential_name.lower() for word in exclude_words) and len(potential_name.split()) <= 3:
                            employee_name = normalize_name(potential_name)
                            break
        
        print(f"Extracted Name: {employee_name}")
        
        # Check if it should go to individual flow
        if query_type == "individual" and employee_name:
            print("✅ WILL GO TO: Individual employee flow (human_loop_question)")
        elif query_type == "all_employees":
            print("ℹ️  WILL GO TO: All employees summary")
        elif query_type == "department":
            print("ℹ️  WILL GO TO: Department query")
        else:
            print("ℹ️  WILL GO TO: General query (might be large_group_summary)")

if __name__ == "__main__":
    test_name_extraction()
