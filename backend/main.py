from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import logging
import sqlite3
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Sequence, TypedDict, Annotated
import re
import os
from dotenv import load_dotenv

# Tenacity imports for retry logic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Setup logging first
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

class LangChainConfig:
    """Centralized configuration for LangChain settings"""
    
    # OpenAI Model Configuration
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    OPENAI_REQUEST_TIMEOUT: int = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    EMBEDDINGS_CHUNK_SIZE: int = int(os.getenv("EMBEDDINGS_CHUNK_SIZE", "1000"))
    
    # Agent Configuration
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "120"))
    MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
    
    # Performance Configuration
    STREAMING_ENABLED: bool = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
    ASYNC_OPERATIONS: bool = os.getenv("ASYNC_OPERATIONS", "true").lower() == "true"
    
    # Monitoring Configuration
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "hr-management-system")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    # Token Tracking Configuration
    ENABLE_TOKEN_TRACKING: bool = os.getenv("ENABLE_TOKEN_TRACKING", "true").lower() == "true"
    COST_TRACKING_ENABLED: bool = os.getenv("COST_TRACKING_ENABLED", "true").lower() == "true"
    MAX_MONTHLY_TOKENS: int = int(os.getenv("MAX_MONTHLY_TOKENS", "1000000"))  # 1M tokens default
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.8"))  # 80% threshold
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Validate numeric ranges
        if not 0 <= cls.OPENAI_TEMPERATURE <= 2:
            raise ValueError("OPENAI_TEMPERATURE must be between 0 and 2")
        
        if cls.OPENAI_MAX_TOKENS < 1:
            raise ValueError("OPENAI_MAX_TOKENS must be positive")
        
        if cls.AGENT_TIMEOUT < 10:
            raise ValueError("AGENT_TIMEOUT must be at least 10 seconds")
        
        logger.info("LangChain configuration validated successfully")
    
    @classmethod
    def get_model_kwargs(cls) -> dict:
        """Get model initialization kwargs"""
        return {
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
            "streaming": cls.STREAMING_ENABLED,
            "max_retries": cls.OPENAI_MAX_RETRIES,
            "request_timeout": cls.OPENAI_REQUEST_TIMEOUT
        }
    
    @classmethod
    def get_embeddings_kwargs(cls) -> dict:
        """Get embeddings initialization kwargs"""
        return {
            "model": cls.EMBEDDINGS_MODEL,
            "chunk_size": cls.EMBEDDINGS_CHUNK_SIZE
        }

# Validate configuration on startup
LangChainConfig.validate_config()

# Initialize LangSmith monitoring if enabled
if LangChainConfig.LANGSMITH_TRACING and LangChainConfig.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LangChainConfig.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LangChainConfig.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_ENDPOINT"] = LangChainConfig.LANGSMITH_ENDPOINT
    
    # Validate LangSmith connection
    try:
        logger.info(f"LangSmith monitoring enabled for project: {LangChainConfig.LANGSMITH_PROJECT}")
        logger.info(f"LangSmith endpoint: {LangChainConfig.LANGSMITH_ENDPOINT}")
        logger.info(f"LangSmith API key configured: {LangChainConfig.LANGSMITH_API_KEY[:8]}...")
    except Exception as e:
        logger.warning(f"LangSmith configuration issue: {e}")
else:
    logger.info("LangSmith monitoring disabled - configure LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable")

# LangChain imports - Required dependencies
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig

from database import execute_query, get_db_connection, search_similar_documents
from models import (
    Employee, EmployeeWithDepartment, EmployeeCreate, EmployeeQuery,
    Attendance, AttendanceCreate, AttendanceQuery,
    Task, TaskCreate, TaskWithDetails, TaskQuery, TaskTimeFilter, TaskWithFullDetails,
    TaskQueryResponse, TaskIntelligentQuery, MultipleTasksResponse, TaskStatisticsResponse,
    TaskNotFoundResponse, TaskHumanLoopQuestionResponse, IndividualTaskResponse,
    Labour, LabourWithFullDetails, LabourWorkloadStatus, LabourQueryResponse, LabourIntelligentQuery,
    IndividualLabourResponse, LabourSkillGroupResponse, LabourWorkloadAnalysisResponse, 
    MultipleLabourersResponse, LabourStatisticsResponse, LabourNotFoundResponse, LabourHumanLoopQuestionResponse,
    LabourTaskAssignmentSummary,
    Department, ChatMessage, APIResponse, AttendanceStatistics,
    EmployeeCreateRequest, EmployeeUpdateRequest
)

# Logging already setup above

app = FastAPI(title="HR Management System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings with configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def initialize_embeddings():
    """Initialize OpenAI embeddings with retry logic"""
    return OpenAIEmbeddings(**LangChainConfig.get_embeddings_kwargs())

try:
    embeddings_model = initialize_embeddings()
    logger.info("OpenAI embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI embeddings after retries: {e}")
    raise RuntimeError("LangChain embeddings are required for this application")

# ==============================================================================
# LANGCHAIN TOOLS FOR HR AGENT
# ==============================================================================

# ==============================================================================
# ENHANCED EMPLOYEE QUERY TOOLS
# ==============================================================================

@tool
def query_employee_details_intelligent(query: str) -> Dict[str, Any]:
    """
    PRIMARY EMPLOYEE QUERY TOOL: Use for ALL conversational employee queries with human-in-the-loop confirmation.
    
    🎯 USE THIS TOOL FOR ALL THESE QUERIES:
    - "do you know [NAME]" / "know [NAME]" 
    - "tell me about [NAME]" / "[NAME] information"
    - "give me [NAME] details" / "show me [NAME]"
    - "find employee [NAME]" / "who is [NAME]"
    - "[NAME] details" / "[NAME] profile"
    - "AI department employees" / "show me marketing team"
    - "employees who are leaders" / "all HR staff"
    - "all employees" / "show me everyone"
    
    ✅ This tool provides human-in-the-loop confirmation before showing details.
    ⚠️ This tool ONLY shows employee information. It NEVER opens forms or triggers updates.
    
    Do NOT use for:
    - "update [NAME] details" (use update_employee_intelligent instead)
    - "change [NAME]" (use update_employee_intelligent instead) 
    - "modify [NAME]" (use update_employee_intelligent instead)
    
    Returns employee details for viewing only - no forms will be triggered.
    """
    try:
        import re
        
        # Initialize query parameters
        department = None
        role = None
        employee_name = None
        query_type = "general"  # individual, department, role, general, all_employees
        
        query_lower = query.lower().strip()
        
        # Detect query intent - PRIORITIZE individual employee queries
        # First check if it's a simple name pattern (1-3 words that look like names)
        simple_name_pattern = r'^([A-Za-z]{2,20}(?:\s+[A-Za-z]{2,20}){0,2})$'
        simple_name_match = re.match(simple_name_pattern, query.strip())
        
        if simple_name_match:
            # This looks like a simple name query
            query_type = "individual"
        elif any(phrase in query_lower for phrase in ['all employee', 'everyone', 'show me all', 'entire company']):
            query_type = "all_employees"
        elif any(phrase in query_lower for phrase in [
            # Information/details queries
            'details', 'information', 'info about', 'give me',
            # Knowledge-based queries  
            'do you know', 'know', 'familiar with', 'heard of',
            # Existence queries
            'who is', 'is there', 'does', 'work here',
            # Question words
            'tell me about', 'show me', 'find'
        ]) and not any(dept in query_lower for dept in ['department', 'team']):
            query_type = "individual"
        elif any(phrase in query_lower for phrase in ['department', 'team', 'dept']):
            query_type = "department" 
        elif any(phrase in query_lower for phrase in ['ai', 'marketing', 'hr', 'finance']):
            query_type = "department"  # Department names
        elif any(phrase in query_lower for phrase in ['leader', 'hr', 'employee', 'staff']):
            query_type = "role"
        
        # Extract department information - BUT NOT for individual employee queries
        if query_type != "individual":  # Only extract department info for non-individual queries
            dept_patterns = {
                'ai': ['ai', 'artificial intelligence', 'rise ai'],
                'marketing': ['marketing', 'marketing team'],
                'hr': ['hr', 'human resources', 'hr team'],
                'finance': ['finance', 'accounting', 'finance team'],
                'operations': ['operations', 'ops', 'operations team'],
                'tech': ['tech', 'technology', 'tech team'],
                'ramstudios': ['ramstudios', 'ram studios']
            }
            
            for dept_key, patterns in dept_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    department = dept_key
                    break
        
        # Extract role information - BUT NOT for individual employee queries
        if query_type != "individual":  # Only extract role info for non-individual queries
            role_patterns = {
                'leader': ['leader', 'leaders', 'lead', 'manager', 'head'],
                'hr': ['hr staff', 'hr employee', 'hr person'],
                'employee': ['employee', 'staff member', 'worker'],
                'labourer': ['labourer', 'labor', 'worker']
            }
            
            for role_key, patterns in role_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    role = role_key
                    break
        
        # Extract individual employee name with improved patterns and normalization
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
        
        # CRITICAL FIX: If we extracted a name but query_type is not individual, force it to individual
        # This handles cases like simple name queries: "kalhara", "simon", etc.
        if employee_name and query_type != "individual":
            # Double-check this looks like a person's name and not a common word
            potential_name_lower = employee_name.lower()
            common_non_names = ['all', 'everyone', 'staff', 'team', 'people', 'employees', 'department', 'summary', 'list', 'details', 'information']
            
            if not any(word in potential_name_lower for word in common_non_names):
                query_type = "individual"
                logger.info(f"DEBUG: Forced query_type to 'individual' because we extracted name: '{employee_name}'")
        
        
        # Build SQL query based on detected intent - Use exact matching first for individual employee queries
        results = []
        
        if query_type == "individual" and employee_name:
            # Enhanced multi-approach search for exact matching
            normalized_search_name = employee_name.strip()
            results = []
            
            logger.info(f"DEBUG: Searching for employee: '{normalized_search_name}'")
            
            # Approach 1: Comprehensive exact matching with multiple case variations
            search_variations = [
                normalized_search_name,           # As normalized (Title Case)
                normalized_search_name.lower(),   # lowercase
                normalized_search_name.upper(),   # UPPERCASE
                employee_name.strip()             # Original case
            ]
            
            # Remove duplicates while preserving order
            search_variations = list(dict.fromkeys(search_variations))
            
            for search_name in search_variations:
                if results:  # Stop if we found a match
                    break
                    
                exact_where_conditions = ["e.is_active = true", "LOWER(TRIM(e.name)) = LOWER(TRIM(%s))"]
                exact_params = [search_name]
                
                if department:
                    exact_where_conditions.append("LOWER(d.name) LIKE %s")
                    exact_params.append(f"%{department}%")
                
                if role:
                    exact_where_conditions.append("LOWER(e.role) = %s")
                    exact_params.append(role.lower())
                
                exact_query = """
                SELECT 
                    e.id,
                    e.name,
                    e.email,
                    e.role,
                    e.phone_number,
                    e.address,
                    e.is_active,
                    d.name as department_name,
                    d.id as department_id
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE {where_clause}
                ORDER BY d.name, e.role, e.name
                """.format(where_clause=" AND ".join(exact_where_conditions))
                
                logger.info(f"DEBUG: Trying exact match for: '{search_name}'")
                results = execute_query(exact_query, tuple(exact_params))
            
            # If no exact match found, try partial matching
            if not results:
                partial_where_conditions = ["e.is_active = true", "LOWER(TRIM(e.name)) LIKE LOWER(TRIM(%s))"]
                partial_params = [f"%{normalized_search_name}%"]
                
                # Add department/role filters if specified for partial match
                if department:
                    partial_where_conditions.append("LOWER(d.name) LIKE %s")
                    partial_params.append(f"%{department}%")
                
                if role:
                    partial_where_conditions.append("LOWER(e.role) = %s")
                    partial_params.append(role.lower())
                
                partial_query = """
                SELECT 
                    e.id,
                    e.name,
                    e.email,
                    e.role,
                    e.phone_number,
                    e.address,
                    e.is_active,
                    d.name as department_name,
                    d.id as department_id
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE {where_clause}
                ORDER BY 
                    CASE WHEN LOWER(TRIM(e.name)) = LOWER(TRIM(%s)) THEN 1 ELSE 2 END,
                    LENGTH(e.name) ASC
                """.format(where_clause=" AND ".join(partial_where_conditions))
                
                partial_params.append(normalized_search_name)  # For ORDER BY exact match priority
                results = execute_query(partial_query, tuple(partial_params))
        else:
            # For non-individual queries, use the original logic
            where_conditions = ["e.is_active = true"]
            params = []
            
            if department:
                where_conditions.append("LOWER(d.name) LIKE %s")
                params.append(f"%{department}%")
            
            if role:
                where_conditions.append("LOWER(e.role) = %s")
                params.append(role.lower())
            
            if employee_name:
                where_conditions.append("LOWER(e.name) LIKE %s")
                params.append(f"%{employee_name.lower()}%")
            
            # Construct base query
            base_query = """
            SELECT 
                e.id,
                e.name,
                e.email,
                e.role,
                e.phone_number,
                e.address,
                e.is_active,
                d.name as department_name,
                d.id as department_id
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE {where_clause}
            ORDER BY d.name, e.role, e.name
            """.format(where_clause=" AND ".join(where_conditions))
            
            # Execute query
            results = execute_query(base_query, tuple(params))
        
        if not results:
            # If this was an individual name search and no results found, show similar names but don't offer to create
            if query_type == "individual" and employee_name:
                logger.info(f"DEBUG: No exact matches found for '{employee_name}', checking similar names...")
                # Use lower threshold for lookup queries to find similar names
                similar_employees = _find_similar_employees(employee_name, threshold=0.4)
                
                if similar_employees:
                    # Check if the best match is very close (possible exact match missed)
                    best_match = similar_employees[0]
                    similarity_score = best_match.get('similarity_score', 0)
                    
                    print(f"DEBUG: Best similar match: '{best_match['name']}' (score: {similarity_score})")
                    
                    # If similarity is very high (>0.8), treat as likely exact match but still confirm
                    if similarity_score > 0.8:
                        print(f"DEBUG: High similarity score, asking for confirmation before showing details")
                        return {
                            "success": True,
                            "message": f"✅ Found **{best_match['name']}** who closely matches '{employee_name}'! Department: {best_match['department_name']}, Role: {best_match['role']}. Would you like to see their full employee details?",
                            "response_type": "human_loop_question",
                            "conversation_state": {
                                "pending_action": "show_employee_details",
                                "suggested_data": {
                                    "employee": best_match,
                                    "searched_name": employee_name,
                                    "original_query": query
                                }
                            },
                            "follow_up_questions": [
                                f"Yes, show me {best_match['name']}'s full employee details",
                                "No, search for someone else"
                            ]
                        }
                    else:
                        # Regular similar names suggestion
                        return {
                            "success": True,
                            "message": f"I couldn't find '{employee_name}' exactly, but found similar names:\n\n" + 
                                     "\n".join([f"• {emp['name']} ({emp['department_name']} - {emp['role']})" for emp in similar_employees]) + 
                                     f"\n\nDid you mean **{best_match['name']}**? They work in our company. Would you like to see their full employee details?",
                            "response_type": "human_loop_question",
                            "conversation_state": {
                                "pending_action": "show_employee_details",
                                "suggested_data": {
                                    "employee": best_match,
                                    "searched_name": employee_name,
                                    "all_similar": similar_employees,
                                    "original_query": query
                                }
                            },
                            "follow_up_questions": [
                                f"Yes, show me {best_match['name']}'s full employee details",
                                "Show me a different employee from the list",
                                "Try a different search term"
                            ]
                        }
                else:
                    # No similar employees found - offer to create new employee
                    return {
                        "success": True,
                        "message": f"I don't find this employee {employee_name} in our company. Do you want to add {employee_name} as a new employee?",
                        "data": [],
                        "response_type": "human_loop_question",
                        "conversation_state": {
                            "pending_action": "create_employee",
                            "suggested_data": {
                                "suggested_name": employee_name,
                                "original_query": query
                            }
                        },
                        "follow_up_questions": [
                            f"Yes, add {employee_name} as a new employee",
                            "No, try a different search"
                        ]
                    }
            else:
                # General search with no results
                return {
                    "success": True,
                    "message": f"No employees found matching '{query}'. Please check the spelling or try a different search.",
                    "data": [],
                    "response_type": "no_results",
                    "total_found": 0,
                    "follow_up_questions": [
                        "Try searching for a specific department like 'AI team' or 'Marketing'",
                        "Search by role like 'leaders' or 'HR staff'",
                        "Check if you spelled the employee name correctly"
                    ]
                }
        
        total_found = len(results)
        
        # Determine response strategy based on results count and query type
        # PRIORITIZE individual employee queries when names are detected
        if query_type == "individual" and employee_name:
            # Individual employee query - handle all cases first before checking large groups
            if total_found == 1:
                # Single result - Always confirm first
                employee = results[0]
                logger.info(f"DEBUG: Individual query with single result - asking for confirmation before showing details")
                return {
                    "success": True,
                    "message": f"**{employee['name']}** works in our company! Department: {employee['department_name']}, Role: {employee['role']}. Would you like to see their full employee details?",
                    "response_type": "human_loop_question",
                    "conversation_state": {
                        "pending_action": "show_employee_details",
                        "suggested_data": {
                            "employee": employee,
                            "searched_name": employee_name,
                            "original_query": query
                        }
                    },
                    "follow_up_questions": [
                        f"Yes, show me {employee['name']}'s full employee details",
                        "No, I don't need more details right now"
                    ]
                }
            elif total_found > 1:
                # Multiple results for individual name
                logger.info(f"DEBUG: Individual query with multiple results - routing to multiple matches handler")
                return _handle_multiple_matches_query(results, employee_name, query)
            else:
                # No results for individual name - handle similarity search
                logger.info(f"DEBUG: Individual query with no exact results - will trigger similarity search")
                pass  # Let it continue to the no-results handling below
        elif query_type == "all_employees":
            return _handle_all_employees_query(results)
        elif query_type == "department" and total_found <= 15:
            return _handle_department_query(results, department, query)
        elif total_found > 15:
            return _handle_large_group_query(results, query)
        else:
            return _handle_small_group_query(results, query)
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing employee query: {str(e)}",
            "data": [],
            "response_type": "error",
            "query_understood": query
        }

def _handle_all_employees_query(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Handle 'all employees' queries with department summary approach"""
    dept_summary = {}
    total_employees = len(results)
    
    # Group by department
    for emp in results:
        dept_name = emp['department_name']
        if dept_name not in dept_summary:
            dept_summary[dept_name] = {'total': 0, 'roles': {}, 'employees': []}
        
        dept_summary[dept_name]['total'] += 1
        role = emp['role']
        if role not in dept_summary[dept_name]['roles']:
            dept_summary[dept_name]['roles'][role] = 0
        dept_summary[dept_name]['roles'][role] += 1
        dept_summary[dept_name]['employees'].append(emp['name'])
    
    # Create summary message
    summary_lines = [f"**Company Overview: {total_employees} total employees across {len(dept_summary)} departments**\n"]
    
    for dept_name, data in sorted(dept_summary.items()):
        role_breakdown = ", ".join([f"{count} {role}{'s' if count > 1 else ''}" for role, count in data['roles'].items()])
        summary_lines.append(f"• **{dept_name}**: {data['total']} employees ({role_breakdown})")
    
    return {
        "success": True,
        "message": "\n".join(summary_lines),
        "data": dept_summary,
        "response_type": "all_employees_summary",
        "total_found": total_employees,
        "follow_up_questions": [
            "Which department would you like to explore in detail?",
            "Need information about employees in a specific role?",
            "Want to see individual employee details?"
        ]
    }

def _handle_individual_employee_query(employee: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Handle individual employee queries with full details"""
    profile_lines = [
        f"**{employee['name']}** - {employee['role'].title()}",
        f"📧 **Email**: {employee['email']}",
        f"🏢 **Department**: {employee['department_name']}",
    ]
    
    if employee.get('phone_number'):
        profile_lines.append(f"📞 **Phone**: {employee['phone_number']}")
    
    if employee.get('address'):
        profile_lines.append(f"🏠 **Address**: {employee['address']}")
    
    profile_lines.append(f"✅ **Status**: {'Active' if employee['is_active'] else 'Inactive'}")
    
    return {
        "success": True,
        "message": "\n".join(profile_lines),
        "data": employee,
        "response_type": "individual_profile",
        "total_found": 1,
        "follow_up_questions": [
            f"Need {employee['name']}'s attendance records?",
            f"Want to see other employees in {employee['department_name']}?",
            f"Looking for other {employee['role']}s in the company?"
        ]
    }

def _handle_multiple_matches_query(results: List[Dict[str, Any]], name: str, query: str) -> Dict[str, Any]:
    """Handle queries where multiple employees match the name"""
    matches_list = []
    for emp in results:
        matches_list.append(f"• **{emp['name']}** ({emp['role']}) - {emp['department_name']} Department")
    
    message = f"Found {len(results)} employees matching '{name}':\n\n" + "\n".join(matches_list)
    
    return {
        "success": True,
        "message": message + "\n\nPlease specify which person you're looking for.",
        "data": results,
        "response_type": "multiple_matches",
        "total_found": len(results),
        "follow_up_questions": [
            "Which specific person did you mean?",
            "Try adding the department name to your search",
            "Need details for all of these employees?"
        ]
    }

def _handle_department_query(results: List[Dict[str, Any]], department: str, query: str) -> Dict[str, Any]:
    """Handle department-based queries with name list approach"""
    dept_name = results[0]['department_name'] if results else department
    
    # Group by role
    role_groups = {}
    for emp in results:
        role = emp['role']
        if role not in role_groups:
            role_groups[role] = []
        role_groups[role].append(emp['name'])
    
    # Create organized list
    summary_lines = [f"**{dept_name} Department - {len(results)} employees:**\n"]
    
    for role, names in sorted(role_groups.items()):
        role_title = f"{role.title()}s" if len(names) > 1 else role.title()
        summary_lines.append(f"**{role_title}**: {', '.join(names)}")
    
    return {
        "success": True,
        "message": "\n".join(summary_lines),
        "data": {"department": dept_name, "employees": results, "role_groups": role_groups},
        "response_type": "department_list",
        "total_found": len(results),
        "follow_up_questions": [
            "Need detailed information for any specific person?",
            f"Want to see attendance records for {dept_name} team?",
            "Looking for employees with a specific role?"
        ]
    }

def _handle_large_group_query(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Handle queries with large result sets (>15 employees)"""
    # Create department and role summaries
    dept_counts = {}
    role_counts = {}
    
    for emp in results:
        dept = emp['department_name']
        role = emp['role']
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
    
    summary_lines = [f"**Found {len(results)} employees matching your query**\n"]
    
    if len(dept_counts) > 1:
        summary_lines.append("**By Department:**")
        for dept, count in sorted(dept_counts.items()):
            summary_lines.append(f"• {dept}: {count} employees")
        summary_lines.append("")
    
    if len(role_counts) > 1:
        summary_lines.append("**By Role:**")
        for role, count in sorted(role_counts.items()):
            summary_lines.append(f"• {role.title()}s: {count}")
    
    return {
        "success": True,
        "message": "\n".join(summary_lines),
        "data": {"summary": {"departments": dept_counts, "roles": role_counts}, "total": len(results)},
        "response_type": "large_group_summary",
        "total_found": len(results),
        "follow_up_questions": [
            "Which department would you like to see in detail?",
            "Filter by a specific role?",
            "Looking for someone specific in this group?"
        ]
    }

def _handle_small_group_query(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Handle queries with small result sets (2-15 employees)"""
    summary_lines = [f"**Found {len(results)} employees:**\n"]
    
    for emp in results:
        summary_lines.append(f"• **{emp['name']}** ({emp['role']}) - {emp['department_name']}")
    
    return {
        "success": True,
        "message": "\n".join(summary_lines),
        "data": results,
        "response_type": "small_group",
        "total_found": len(results),
        "follow_up_questions": [
            "Need detailed information for any of these employees?",
            "Want to see their contact information?",
            "Looking for attendance records?"
        ]
    }

@tool
def get_department_employee_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of all employees organized by department.
    Perfect for answering "How many employees do we have?" or "Show me our team structure"
    
    Returns:
    - Department-wise employee counts
    - Role distribution within each department  
    - Total company statistics
    - Organized summary for decision-making
    """
    try:
        query = """
        SELECT 
            d.name as department_name,
            e.role,
            COUNT(*) as employee_count,
            STRING_AGG(e.name, ', ' ORDER BY e.name) as employee_names
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE e.is_active = true
        GROUP BY d.name, e.role
        ORDER BY d.name, e.role
        """
        
        results = execute_query(query)
        
        if not results:
            return {
                "success": True,
                "message": "No active employees found in the system.",
                "data": {},
                "total_employees": 0
            }
        
        # Organize data by department
        dept_data = {}
        total_employees = 0
        
        for row in results:
            dept_name = row['department_name']
            role = row['role']
            count = row['employee_count']
            names = row['employee_names']
            
            if dept_name not in dept_data:
                dept_data[dept_name] = {
                    'total_employees': 0,
                    'roles': {},
                    'all_employees': []
                }
            
            dept_data[dept_name]['total_employees'] += count
            dept_data[dept_name]['roles'][role] = {
                'count': count,
                'names': names.split(', ') if names else []
            }
            dept_data[dept_name]['all_employees'].extend(names.split(', ') if names else [])
            
            total_employees += count
        
        # Create summary message
        summary_lines = [
            f"🏢 **Company Structure Overview**",
            f"**Total Employees**: {total_employees} across {len(dept_data)} departments\n"
        ]
        
        for dept_name, data in sorted(dept_data.items()):
            role_summary = ", ".join([f"{info['count']} {role}{'s' if info['count'] > 1 else ''}" 
                                    for role, info in data['roles'].items()])
            summary_lines.append(f"**{dept_name} ({data['total_employees']} employees)**")
            summary_lines.append(f"  └── {role_summary}")
            summary_lines.append("")
        
        return {
            "success": True,
            "message": "\n".join(summary_lines),
            "data": dept_data,
            "total_employees": total_employees,
            "total_departments": len(dept_data),
            "response_type": "department_summary",
            "follow_up_questions": [
                "Which department would you like to explore in detail?",
                "Need specific employee information from any team?",
                "Want to see attendance patterns by department?"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error generating department summary: {str(e)}",
            "data": {}
        }

@tool  
def get_individual_employee_profile(employee_identifier: str) -> Dict[str, Any]:
    """
    DIRECT PROFILE LOOKUP: Get complete profile for a specific employee (INTERNAL USE ONLY).
    
    ⚠️ DO NOT USE FOR CONVERSATIONAL QUERIES LIKE:
    - "do you know [name]" 
    - "tell me about [name]"
    - "who is [name]"
    
    USE query_employee_details_intelligent FOR CONVERSATIONAL QUERIES INSTEAD.
    
    This tool is for direct profile lookup when you already have confirmed employee identifier.
    
    Parameters:
    - employee_identifier: Employee ID (number), email, or full/partial name
    
    Returns:
    - Complete employee profile with all available information
    - Department details and role information
    - Contact information and status
    - Recent activity summary if available
    """
    try:
        # Determine search type
        employee = None
        search_type = "unknown"
        
        if employee_identifier.strip().isdigit():
            # Search by ID
            search_type = "id"
            employee = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE e.id = %s AND e.is_active = true
                """,
                (int(employee_identifier.strip()),)
            )
        elif '@' in employee_identifier:
            # Search by email
            search_type = "email"
            employee = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE LOWER(e.email) = LOWER(%s) AND e.is_active = true
                """,
                (employee_identifier.strip(),)
            )
        else:
            # Search by name (fuzzy matching)
            search_type = "name"
            employee = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE LOWER(e.name) LIKE LOWER(%s) AND e.is_active = true
                ORDER BY 
                    CASE WHEN LOWER(e.name) = LOWER(%s) THEN 1 ELSE 2 END,
                    LENGTH(e.name) ASC
                """,
                (f"%{employee_identifier.strip()}%", employee_identifier.strip())
            )
        
        if not employee:
            return {
                "success": False,
                "message": f"No active employee found matching '{employee_identifier}'. Please check the spelling or try a different identifier.",
                "suggestions": [
                    "Try using the employee's full name",
                    "Use their email address",
                    "Check if the employee is still active"
                ]
            }
        
        if len(employee) > 1:
            # Multiple matches - show options
            matches_list = []
            for emp in employee[:5]:  # Show top 5 matches
                matches_list.append(f"• **{emp['name']}** ({emp['role']}) - {emp['department_name']} | {emp['email']}")
            
            return {
                "success": True,
                "message": f"Found {len(employee)} employees matching '{employee_identifier}':\n\n" + "\n".join(matches_list),
                "data": employee,
                "response_type": "multiple_matches",
                "total_found": len(employee),
                "follow_up_questions": [
                    "Which specific employee did you mean?",
                    "Use the employee's email for exact match",
                    "Try being more specific with the name"
                ]
            }
        
        # Single match found
        emp = employee[0]
        
        # Build comprehensive profile
        profile_sections = [
            f"👤 **{emp['name']}**",
            f"📋 **Role**: {emp['role'].title()}",
            f"🏢 **Department**: {emp['department_name']}",
            f"📧 **Email**: {emp['email']}",
        ]
        
        if emp.get('phone_number'):
            profile_sections.append(f"📞 **Phone**: {emp['phone_number']}")
        
        if emp.get('address'):
            profile_sections.append(f"🏠 **Address**: {emp['address']}")
        
        profile_sections.extend([
            f"🆔 **Employee ID**: {emp['id']}",
            f"✅ **Status**: {'Active' if emp['is_active'] else 'Inactive'}"
        ])
        
        if emp.get('created_at'):
            from datetime import datetime
            created_date = emp['created_at']
            if isinstance(created_date, str):
                created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            profile_sections.append(f"📅 **Joined**: {created_date.strftime('%B %d, %Y')}")
        
        # Get additional context (recent attendance if available)
        try:
            recent_attendance = execute_query(
                """
                SELECT attendance_date, status
                FROM attendances
                WHERE employee_id = %s
                ORDER BY attendance_date DESC
                LIMIT 5
                """,
                (emp['id'],)
            )
            
            if recent_attendance:
                profile_sections.append(f"\n📊 **Recent Attendance** (Last 5 days):")
                for record in recent_attendance:
                    date_str = record['attendance_date'].strftime('%Y-%m-%d') if hasattr(record['attendance_date'], 'strftime') else str(record['attendance_date'])
                    profile_sections.append(f"  • {date_str}: {record['status']}")
        except:
            # If attendance query fails, continue without it
            pass
        
        return {
            "success": True,
            "message": "\n".join(profile_sections),
            "data": emp,
            "response_type": "individual_profile",
            "search_type": search_type,
            "follow_up_questions": [
                f"Need {emp['name']}'s full attendance history?",
                f"Want to see other employees in {emp['department_name']}?",
                f"Looking for other {emp['role']}s in the company?",
                "Need to update any of this information?"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving employee profile: {str(e)}",
            "data": {}
        }

# ==============================================================================



@tool
def get_all_employees_overview(employee_name: str = None, department: str = None) -> Dict[str, Any]:
    """
    Get overview of all employees with optional filtering by name or department.
    """
    try:
        where_conditions = []
        params = []
        
        if employee_name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{employee_name}%")
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT e.id, e.name, e.email, e.role, e.phone_number, e.address, 
                   e.is_active, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY e.name
        """
        
        employees = execute_query(query, params)
        
        if not employees:
            return {"success": False, "message": "No employees found matching the criteria."}
        
        return {
            "success": True, 
            "data": [dict(emp) for emp in employees], 
            "message": f"Successfully retrieved {len(employees)} employee records."
        }
        
    except Exception as e:
        return {"success": False, "message": f"An error occurred: {e}"}

@tool
def get_tasks_report(leader_name: str = None, department: str = None, 
                    priority: str = None, location: str = None, days: int = 30) -> str:
    """
    Fetches a detailed report of tasks with optional filtering.
    """
    try:
        where_conditions = ["t.created_at >= %s"]
        params = [datetime.now() - timedelta(days=days)]
        
        if leader_name:
            where_conditions.append("e.name ILIKE %s AND e.role = 'leader'")
            params.append(f"%{leader_name}%")
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        if priority:
            where_conditions.append("t.priority = %s")
            params.append(priority)
        
        if location:
            where_conditions.append("t.location ILIKE %s")
            params.append(f"%{location}%")
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
            SELECT DISTINCT t.id, t.task_title, t.location, t.expected_days, 
                   t.priority, t.notes, t.created_at, tg.group_name,
                   e.name as leader_name, d.name as department_name
            FROM tasks t
            LEFT JOIN task_groups tg ON t.task_group_id = tg.id
            LEFT JOIN task_group_leaders tgl ON tg.id = tgl.task_group_id
            LEFT JOIN employees e ON tgl.employee_id = e.id
            LEFT JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY t.created_at DESC
        """
        
        tasks = execute_query(query, params)
        
        if not tasks:
            return f"No tasks found for the specified criteria in the last {days} days."
        
        report_lines = [
            f"**Tasks Report - Last {days} Days**",
            "---"
        ]
        
        for task in tasks:
            report_lines.extend([
                f"* **Task:** {task['task_title']} (Priority: {task['priority']})",
                f"  - **Group:** {task['group_name'] or 'N/A'}",
                f"  - **Leader:** {task['leader_name'] or 'N/A'}",
                f"  - **Department:** {task['department_name'] or 'N/A'}",
                f"  - **Location:** {task['location'] or 'N/A'}",
                f"  - **Expected Days:** {task['expected_days'] or 'N/A'}",
                f"  - **Created:** {task['created_at'].strftime('%Y-%m-%d')}",
                ""
            ])
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"Error generating tasks report: {e}"

# ==============================================================================
# INTELLIGENT TASK QUERY TOOLS
# ==============================================================================

@tool
def query_tasks_intelligent(query: str) -> Dict[str, Any]:
    """
    Intelligent task query tool that understands natural language requests about tasks.
    
    Handles queries like:
    - "today's tasks" / "tasks for today"
    - "urgent tasks" / "high priority tasks"
    - "tasks led by John" / "Sarah's tasks"
    - "carpentry tasks" / "electrical work"
    - "tasks in downtown office" / "remote tasks"
    - "last week's completed tasks"
    - "show me all tasks" / "monthly task summary"
    
    Uses smart response patterns:
    - Individual queries (1 task): Full details immediately
    - Small groups (2-10): Show basic details, offer more info
    - Large groups (>10): Summary first, then ask for specifics
    - Time-based queries: Relevant timeframe analysis
    
    Parameters:
    - query: Natural language query about tasks
    
    Returns:
    - success: bool
    - message: Human-readable summary
    - data: Task data (varies based on query type)
    - response_type: "individual_task", "multiple_tasks", "task_statistics", etc.
    - follow_up_questions: Suggested next actions
    - total_found: Number of tasks found
    """
    try:
        import re
        from datetime import datetime, timedelta
        
        # Initialize query parameters
        time_filter = None
        priority_filter = []
        leader_names = []
        labour_skills = []
        departments = []
        locations = []
        task_titles = []
        query_type = "general"
        
        query_lower = query.lower().strip()
        
        # Detect time-based filters
        current_date = datetime.now().date()
        time_patterns = {
            "today": (current_date, current_date),
            "yesterday": (current_date - timedelta(days=1), current_date - timedelta(days=1)),
            "this week": (current_date - timedelta(days=current_date.weekday()), current_date),
            "last week": (current_date - timedelta(days=current_date.weekday() + 7), current_date - timedelta(days=current_date.weekday() + 1)),
            "this month": (current_date.replace(day=1), current_date),
            "last month": ((current_date.replace(day=1) - timedelta(days=1)).replace(day=1), current_date.replace(day=1) - timedelta(days=1))
        }
        
        for time_key, (start_date, end_date) in time_patterns.items():
            if time_key in query_lower or time_key.replace(" ", "") in query_lower:
                time_filter = (start_date, end_date)
                break
        
        # Detect priority filters
        priority_patterns = {
            "urgent": ["Urgent"],
            "high priority": ["High"],
            "high": ["High"],
            "medium priority": ["Medium"],
            "medium": ["Medium"],
            "low priority": ["Low"],
            "low": ["Low"]
        }
        
        for priority_key, priority_values in priority_patterns.items():
            if priority_key in query_lower:
                priority_filter.extend(priority_values)
        
        # Detect specific people (leaders or employees)
        name_patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        potential_names = [name for name in name_patterns if len(name.split()) <= 3 and name not in ['Priority', 'High', 'Low', 'Medium', 'Urgent']]
        
        # Detect locations
        location_keywords = ['office', 'downtown', 'remote', 'site', 'warehouse', 'factory', 'field']
        for keyword in location_keywords:
            if keyword in query_lower:
                locations.append(keyword)
        
        # Detect skills/labour types
        skill_keywords = ['carpentry', 'electrical', 'plumbing', 'painting', 'construction', 'maintenance', 'repair', 'installation']
        for skill in skill_keywords:
            if skill in query_lower:
                labour_skills.append(skill)
        
        # Build dynamic SQL query
        base_query = """
            SELECT DISTINCT 
                t.id, t.task_title, t.location, t.expected_days, 
                t.priority, t.notes, t.created_at,
                tg.group_name,
                STRING_AGG(DISTINCT e.name, ', ') as leader_names,
                STRING_AGG(DISTINCT d.name, ', ') as department_names,
                STRING_AGG(DISTINCT l.name, ', ') as labour_names,
                STRING_AGG(DISTINCT l.skill, ', ') as labour_skills
            FROM tasks t
            LEFT JOIN task_groups tg ON t.task_group_id = tg.id
            LEFT JOIN task_group_leaders tgl ON tg.id = tgl.task_group_id
            LEFT JOIN employees e ON tgl.employee_id = e.id
            LEFT JOIN departments d ON e.department_id = d.id
            LEFT JOIN task_labours tl ON t.id = tl.task_id
            LEFT JOIN labours l ON tl.labour_id = l.id
        """
        
        where_conditions = []
        params = []
        param_count = 0
        
        # Apply time filter
        if time_filter:
            start_date, end_date = time_filter
            where_conditions.append(f"DATE(t.created_at) >= %s AND DATE(t.created_at) <= %s")
            params.extend([start_date, end_date])
            param_count += 2
        
        # Apply priority filter
        if priority_filter:
            placeholders = ', '.join(['%s'] * len(priority_filter))
            where_conditions.append(f"t.priority IN ({placeholders})")
            params.extend(priority_filter)
            param_count += len(priority_filter)
        
        # Apply name filter (search in leaders)
        if potential_names:
            name_conditions = []
            for name in potential_names:
                name_conditions.append("e.name ILIKE %s")
                params.append(f"%{name}%")
                param_count += 1
            if name_conditions:
                where_conditions.append(f"({' OR '.join(name_conditions)})")
        
        # Apply location filter
        if locations:
            location_conditions = []
            for location in locations:
                location_conditions.append("t.location ILIKE %s")
                params.append(f"%{location}%")
                param_count += 1
            if location_conditions:
                where_conditions.append(f"({' OR '.join(location_conditions)})")
        
        # Apply skill filter
        if labour_skills:
            skill_conditions = []
            for skill in labour_skills:
                skill_conditions.append("l.skill ILIKE %s")
                params.append(f"%{skill}%")
                param_count += 1
            if skill_conditions:
                where_conditions.append(f"({' OR '.join(skill_conditions)})")
        
        # Build final query
        if where_conditions:
            final_query = base_query + " WHERE " + " AND ".join(where_conditions)
        else:
            final_query = base_query
            
        final_query += " GROUP BY t.id, t.task_title, t.location, t.expected_days, t.priority, t.notes, t.created_at, tg.group_name ORDER BY t.created_at DESC"
        
        # Execute query
        tasks = execute_query(final_query, tuple(params))
        
        if not tasks:
            return {
                "success": False,
                "message": "No tasks found matching your criteria. Try adjusting your search terms or time period.",
                "data": [],
                "response_type": "task_not_found",
                "total_found": 0,
                "follow_up_questions": [
                    "Would you like to see all tasks?",
                    "Try searching for a different time period?",
                    "Search by task priority instead?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "priority": priority_filter,
                    "leaders": potential_names,
                    "locations": locations,
                    "skills": labour_skills,
                    "has_time_filter": time_filter is not None
                }
            }
        
        # Determine response type based on result count
        task_count = len(tasks)
        
        if task_count == 1:
            # Individual task - show full details
            task = tasks[0]
            return {
                "success": True,
                "message": f"Found 1 task: **{task['task_title']}**",
                "data": {
                    "task_id": task['id'],
                    "title": task['task_title'],
                    "priority": task['priority'],
                    "location": task['location'],
                    "expected_days": task['expected_days'],
                    "notes": task['notes'],
                    "group_name": task['group_name'],
                    "leaders": task['leader_names'] or "No leaders assigned",
                    "departments": task['department_names'] or "No departments",
                    "labourers": task['labour_names'] or "No labourers assigned",
                    "labour_skills": task['labour_skills'] or "No skills specified",
                    "created_at": task['created_at'].strftime('%Y-%m-%d %H:%M') if task['created_at'] else None
                },
                "response_type": "individual_task",
                "total_found": 1,
                "follow_up_questions": [
                    "Would you like to see related tasks?",
                    "Show me other tasks with the same priority?",
                    "Find tasks from the same group?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "priority": priority_filter,
                    "leaders": potential_names,
                    "locations": locations,
                    "skills": labour_skills
                }
            }
        
        elif task_count <= 10:
            # Small group - show basic details with expansion option
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task['id'],
                    "title": task['task_title'],
                    "priority": task['priority'],
                    "group": task['group_name'],
                    "leaders": task['leader_names'],
                    "location": task['location'],
                    "created_at": task['created_at'].strftime('%Y-%m-%d') if task['created_at'] else None
                })
            
            return {
                "success": True,
                "message": f"Found {task_count} tasks matching your criteria:",
                "data": task_list,
                "response_type": "multiple_tasks",
                "total_found": task_count,
                "follow_up_questions": [
                    "Show me full details for any specific task?",
                    "Filter by priority level?",
                    "Group by task leaders?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "priority": priority_filter,
                    "leaders": potential_names,
                    "locations": locations,
                    "skills": labour_skills
                }
            }
        
        else:
            # Large group - show statistics and summary
            priority_breakdown = {}
            group_breakdown = {}
            location_breakdown = {}
            
            for task in tasks:
                # Priority breakdown
                priority = task['priority'] or 'Unspecified'
                priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                
                # Group breakdown
                group = task['group_name'] or 'No Group'
                group_breakdown[group] = group_breakdown.get(group, 0) + 1
                
                # Location breakdown
                location = task['location'] or 'No Location'
                location_breakdown[location] = location_breakdown.get(location, 0) + 1
            
            return {
                "success": True,
                "message": f"Found {task_count} tasks. Here's a summary breakdown:",
                "data": {
                    "total_tasks": task_count,
                    "priority_breakdown": priority_breakdown,
                    "group_breakdown": group_breakdown,
                    "location_breakdown": location_breakdown,
                    "sample_tasks": [
                        {
                            "title": task['task_title'],
                            "priority": task['priority'],
                            "group": task['group_name'],
                            "created_at": task['created_at'].strftime('%Y-%m-%d') if task['created_at'] else None
                        } for task in tasks[:5]  # Show first 5 as samples
                    ]
                },
                "response_type": "task_statistics",
                "total_found": task_count,
                "priority_breakdown": priority_breakdown,
                "completion_trends": {},  # Could be enhanced later
                "follow_up_questions": [
                    "Show me only urgent/high priority tasks?",
                    "Filter by specific task group?",
                    "Show tasks from last week only?",
                    "Group tasks by location?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "priority": priority_filter,
                    "leaders": potential_names,
                    "locations": locations,
                    "skills": labour_skills
                }
            }
    
    except Exception as e:
        logger.error(f"Error in intelligent task query: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"An error occurred while searching for tasks: {str(e)}",
            "data": None,
            "response_type": "error",
            "total_found": 0,
            "follow_up_questions": [
                "Try rephrasing your query?",
                "Search for tasks in a different way?"
            ]
        }

# ==============================================================================
# INTELLIGENT LABOUR QUERY TOOLS
# ==============================================================================

@tool
def query_labourers_intelligent(query: str) -> Dict[str, Any]:
    """
    Intelligent labour query tool that understands natural language requests about labourers.
    
    Handles queries like:
    - "show me all carpenters" / "electricians available"
    - "who is John the carpenter" / "labour details for Sarah"
    - "busy labourers" / "available workers" / "overloaded staff"
    - "who worked on urgent tasks" / "labourers on downtown projects"
    - "workload analysis" / "skill distribution"
    - "carpenters working with John (leader)" / "electrical workers this week"
    
    Uses smart response patterns:
    - Individual queries (1 person): Full profile with task assignments
    - Skill groups: Labour teams organized by specialization
    - Workload analysis: Busy vs available labourers with task counts
    - Large datasets: Statistical summaries with skill breakdowns
    
    Parameters:
    - query: Natural language query about labourers
    
    Returns:
    - success: bool
    - message: Human-readable summary
    - data: Labour data (varies based on query type)
    - response_type: "individual_labour", "skill_group", "workload_analysis", etc.
    - follow_up_questions: Suggested next actions
    - total_found: Number of labourers found
    """
    try:
        import re
        from datetime import datetime, timedelta
        
        # Initialize query parameters
        skill_filter = []
        labour_names = []
        workload_filter = []
        task_types = []
        leader_names = []
        departments = []
        time_filter = None
        query_type = "general"
        
        query_lower = query.lower().strip()
        
        # Detect labour names (people search)
        name_patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        potential_names = [name for name in name_patterns if len(name.split()) <= 3 and name not in ['Priority', 'High', 'Low', 'Medium', 'Urgent']]
        
        # Detect skill-based queries
        skill_patterns = {
            'carpenter': ['carpenter', 'carpentry', 'wood', 'woodwork'],
            'electrician': ['electrician', 'electrical', 'electric', 'wiring'],
            'plumber': ['plumber', 'plumbing', 'pipes', 'water'],
            'painter': ['painter', 'painting', 'paint'],
            'construction': ['construction', 'builder', 'building'],
            'maintenance': ['maintenance', 'repair', 'fix'],
            'installation': ['installation', 'installer', 'install'],
            'welder': ['welder', 'welding', 'metal'],
            'mason': ['mason', 'masonry', 'brick', 'stone']
        }
        
        for skill_key, skill_aliases in skill_patterns.items():
            if any(alias in query_lower for alias in skill_aliases):
                skill_filter.append(skill_key)
        
        # Detect workload-related queries
        workload_patterns = {
            'available': ['available', 'free', 'idle', 'no tasks', 'not busy'],
            'busy': ['busy', 'working', 'occupied', 'assigned'],
            'overloaded': ['overloaded', 'too much work', 'overwhelmed', 'heavy load'],
            'light_load': ['light load', 'few tasks', 'not much work']
        }
        
        for workload_key, workload_aliases in workload_patterns.items():
            if any(alias in query_lower for alias in workload_aliases):
                workload_filter.append(workload_key)
        
        # Detect task priority and urgency filters
        if any(word in query_lower for word in ['urgent', 'high priority', 'critical']):
            task_types.append('urgent')
        
        # Detect time-based filters (reusing from task system)
        current_date = datetime.now().date()
        time_patterns = {
            "today": (current_date, current_date),
            "yesterday": (current_date - timedelta(days=1), current_date - timedelta(days=1)),
            "this week": (current_date - timedelta(days=current_date.weekday()), current_date),
            "last week": (current_date - timedelta(days=current_date.weekday() + 7), current_date - timedelta(days=current_date.weekday() + 1)),
            "this month": (current_date.replace(day=1), current_date),
            "last month": ((current_date.replace(day=1) - timedelta(days=1)).replace(day=1), current_date.replace(day=1) - timedelta(days=1))
        }
        
        for time_key, (start_date, end_date) in time_patterns.items():
            if time_key in query_lower or time_key.replace(" ", "") in query_lower:
                time_filter = (start_date, end_date)
                break
        
        # Build dynamic SQL query for labourers
        base_query = """
            SELECT DISTINCT 
                l.id, l.name, l.skill, l.created_at, l.updated_at,
                COUNT(DISTINCT tl.task_id) as current_task_count,
                STRING_AGG(DISTINCT t.task_title, ', ') as current_tasks,
                STRING_AGG(DISTINCT tg.group_name, ', ') as task_groups,
                STRING_AGG(DISTINCT e.name, ', ') as working_under_leaders,
                STRING_AGG(DISTINCT d.name, ', ') as leader_departments,
                CASE 
                    WHEN COUNT(DISTINCT tl.task_id) = 0 THEN 'available'
                    WHEN COUNT(DISTINCT tl.task_id) <= 2 THEN 'light_load'
                    WHEN COUNT(DISTINCT tl.task_id) <= 4 THEN 'moderate_load'
                    WHEN COUNT(DISTINCT tl.task_id) <= 6 THEN 'heavy_load'
                    ELSE 'overloaded'
                END as workload_status,
                AVG(CASE 
                    WHEN t.priority = 'Urgent' THEN 4
                    WHEN t.priority = 'High' THEN 3
                    WHEN t.priority = 'Medium' THEN 2
                    WHEN t.priority = 'Low' THEN 1
                    ELSE 2
                END) as avg_task_priority
            FROM labours l
            LEFT JOIN task_labours tl ON l.id = tl.labour_id
            LEFT JOIN tasks t ON tl.task_id = t.id
            LEFT JOIN task_groups tg ON t.task_group_id = tg.id
            LEFT JOIN task_group_leaders tgl ON tg.id = tgl.task_group_id
            LEFT JOIN employees e ON tgl.employee_id = e.id
            LEFT JOIN departments d ON e.department_id = d.id
        """
        
        where_conditions = []
        params = []
        
        # Apply time filter for task assignments
        if time_filter:
            start_date, end_date = time_filter
            where_conditions.append("(t.created_at IS NULL OR (DATE(t.created_at) >= %s AND DATE(t.created_at) <= %s))")
            params.extend([start_date, end_date])
        
        # Apply skill filter
        if skill_filter:
            skill_conditions = []
            for skill in skill_filter:
                skill_conditions.append("l.skill ILIKE %s")
                params.append(f"%{skill}%")
            if skill_conditions:
                where_conditions.append(f"({' OR '.join(skill_conditions)})")
        
        # Apply name filter
        if potential_names:
            name_conditions = []
            for name in potential_names:
                name_conditions.append("l.name ILIKE %s")
                params.append(f"%{name}%")
            if name_conditions:
                where_conditions.append(f"({' OR '.join(name_conditions)})")
        
        # Apply task priority filter
        if task_types and 'urgent' in task_types:
            where_conditions.append("(t.priority IN ('Urgent', 'High') OR t.priority IS NULL)")
        
        # Build final query
        if where_conditions:
            final_query = base_query + " WHERE " + " AND ".join(where_conditions)
        else:
            final_query = base_query
            
        final_query += " GROUP BY l.id, l.name, l.skill, l.created_at, l.updated_at ORDER BY l.name ASC"
        
        # Execute query
        labourers = execute_query(final_query, tuple(params))
        
        if not labourers:
            return {
                "success": False,
                "message": "No labourers found matching your criteria. Try adjusting your search terms or filters.",
                "data": [],
                "response_type": "labour_not_found",
                "total_found": 0,
                "follow_up_questions": [
                    "Would you like to see all labourers?",
                    "Search for a different skill type?",
                    "Check workload for all workers?"
                ],
                "alternative_queries": [
                    "Show me all labourers",
                    "Available workers",
                    "Skill distribution"
                ],
                "similar_skills": ["carpenter", "electrician", "plumber", "painter"],
                "filters_applied": {
                    "skills": skill_filter,
                    "names": potential_names,
                    "workload": workload_filter,
                    "has_time_filter": time_filter is not None
                }
            }
        
        # Apply workload filter after getting results (since it's calculated in query)
        if workload_filter:
            filtered_labourers = []
            for labourer in labourers:
                if labourer['workload_status'] in workload_filter:
                    filtered_labourers.append(labourer)
            labourers = filtered_labourers
        
        if not labourers and workload_filter:
            return {
                "success": False,
                "message": f"No labourers found with workload status: {', '.join(workload_filter)}",
                "data": [],
                "response_type": "labour_not_found",
                "total_found": 0,
                "follow_up_questions": [
                    "Try a different workload filter?",
                    "Show workload analysis for all labourers?",
                    "See available vs busy workers?"
                ],
                "filters_applied": {
                    "skills": skill_filter,
                    "workload": workload_filter
                }
            }
        
        # Determine response type based on result count and query intent
        labour_count = len(labourers)
        
        if labour_count == 1:
            # Individual labour - show full details
            labourer = labourers[0]
            workload_analysis = {
                "status": labourer['workload_status'],
                "task_count": labourer['current_task_count'],
                "average_priority": round(labourer['avg_task_priority'] or 0, 1),
                "recommendation": _get_workload_recommendation(labourer['workload_status'], labourer['current_task_count'])
            }
            
            return {
                "success": True,
                "message": f"Found labourer: **{labourer['name']}** ({labourer['skill']})",
                "data": {
                    "labour_id": labourer['id'],
                    "name": labourer['name'],
                    "skill": labourer['skill'],
                    "workload_status": labourer['workload_status'],
                    "current_task_count": labourer['current_task_count'],
                    "current_tasks": labourer['current_tasks'] or "No current tasks",
                    "task_groups": labourer['task_groups'] or "No task groups",
                    "working_under_leaders": labourer['working_under_leaders'] or "No current leaders",
                    "leader_departments": labourer['leader_departments'] or "No departments",
                    "average_task_priority": round(labourer['avg_task_priority'] or 0, 1),
                    "created_at": labourer['created_at'].strftime('%Y-%m-%d') if labourer['created_at'] else None
                },
                "response_type": "individual_labour",
                "total_found": 1,
                "workload_analysis": workload_analysis,
                "follow_up_questions": [
                    "Show other labourers with the same skill?",
                    "Find available workers for task assignment?",
                    "Show workload analysis for all workers?"
                ],
                "filters_applied": {
                    "skills": skill_filter,
                    "names": potential_names,
                    "workload": workload_filter
                }
            }
        
        elif labour_count <= 10 and len(skill_filter) == 1:
            # Skill group - show workers organized by skill
            skill_name = skill_filter[0] if skill_filter else "Mixed Skills"
            labour_list = []
            workload_dist = {}
            
            for labourer in labourers:
                labour_list.append({
                    "id": labourer['id'],
                    "name": labourer['name'],
                    "skill": labourer['skill'],
                    "workload_status": labourer['workload_status'],
                    "task_count": labourer['current_task_count'],
                    "current_tasks": labourer['current_tasks'],
                    "leaders": labourer['working_under_leaders']
                })
                
                # Track workload distribution
                status = labourer['workload_status']
                workload_dist[status] = workload_dist.get(status, 0) + 1
            
            return {
                "success": True,
                "message": f"Found {labour_count} {skill_name} workers:",
                "data": labour_list,
                "response_type": "skill_group",
                "skill_name": skill_name,
                "total_found": labour_count,
                "workload_distribution": workload_dist,
                "follow_up_questions": [
                    f"Show only available {skill_name}s?",
                    f"Find busy {skill_name}s?",
                    "Get detailed workload analysis?"
                ],
                "filters_applied": {
                    "skills": skill_filter,
                    "workload": workload_filter
                }
            }
        
        elif labour_count <= 15:
            # Multiple labourers - show summary list
            labour_list = []
            skill_breakdown = {}
            workload_summary = {}
            
            for labourer in labourers:
                labour_list.append({
                    "id": labourer['id'],
                    "name": labourer['name'],
                    "skill": labourer['skill'],
                    "workload_status": labourer['workload_status'],
                    "task_count": labourer['current_task_count'],
                    "leaders": labourer['working_under_leaders']
                })
                
                # Track skill breakdown
                skill = labourer['skill']
                skill_breakdown[skill] = skill_breakdown.get(skill, 0) + 1
                
                # Track workload summary
                status = labourer['workload_status']
                workload_summary[status] = workload_summary.get(status, 0) + 1
            
            return {
                "success": True,
                "message": f"Found {labour_count} labourers matching your criteria:",
                "data": labour_list,
                "response_type": "multiple_labourers",
                "total_found": labour_count,
                "skill_breakdown": skill_breakdown,
                "workload_summary": workload_summary,
                "follow_up_questions": [
                    "Filter by specific skill?",
                    "Show only available workers?",
                    "Get workload analysis?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "skills": skill_filter,
                    "names": potential_names,
                    "workload": workload_filter
                }
            }
        
        else:
            # Large group - show statistics and analysis
            skill_breakdown = {}
            workload_breakdown = {}
            task_assignment_patterns = {}
            
            for labourer in labourers:
                # Skill breakdown
                skill = labourer['skill']
                skill_breakdown[skill] = skill_breakdown.get(skill, 0) + 1
                
                # Workload breakdown
                status = labourer['workload_status']
                workload_breakdown[status] = workload_breakdown.get(status, 0) + 1
                
                # Task assignment patterns
                task_count = labourer['current_task_count']
                if task_count == 0:
                    task_assignment_patterns['No tasks'] = task_assignment_patterns.get('No tasks', 0) + 1
                elif task_count <= 2:
                    task_assignment_patterns['1-2 tasks'] = task_assignment_patterns.get('1-2 tasks', 0) + 1
                elif task_count <= 4:
                    task_assignment_patterns['3-4 tasks'] = task_assignment_patterns.get('3-4 tasks', 0) + 1
                else:
                    task_assignment_patterns['5+ tasks'] = task_assignment_patterns.get('5+ tasks', 0) + 1
            
            # Generate recommendations
            recommendations = _generate_labour_recommendations(workload_breakdown, skill_breakdown, labour_count)
            
            return {
                "success": True,
                "message": f"Found {labour_count} labourers. Here's a comprehensive analysis:",
                "data": {
                    "total_labourers": labour_count,
                    "skill_breakdown": skill_breakdown,
                    "workload_breakdown": workload_breakdown,
                    "task_assignment_patterns": task_assignment_patterns,
                    "sample_labourers": [
                        {
                            "name": labourer['name'],
                            "skill": labourer['skill'],
                            "workload_status": labourer['workload_status'],
                            "task_count": labourer['current_task_count']
                        } for labourer in labourers[:5]  # Show first 5 as samples
                    ]
                },
                "response_type": "labour_statistics",
                "total_labourers": labour_count,
                "skill_breakdown": skill_breakdown,
                "workload_trends": workload_breakdown,
                "task_assignment_patterns": task_assignment_patterns,
                "recommendations": recommendations,
                "follow_up_questions": [
                    "Show only available workers?",
                    "Filter by specific skill (carpenter, electrician, etc.)?",
                    "Find overloaded workers who need help?",
                    "Show detailed workload analysis?"
                ],
                "time_period": str(time_filter) if time_filter else None,
                "filters_applied": {
                    "skills": skill_filter,
                    "workload": workload_filter,
                    "task_types": task_types
                }
            }
    
    except Exception as e:
        logger.error(f"Error in intelligent labour query: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"An error occurred while searching for labourers: {str(e)}",
            "data": None,
            "response_type": "error",
            "total_found": 0,
            "follow_up_questions": [
                "Try rephrasing your query?",
                "Search for labourers in a different way?"
            ]
        }

def _get_workload_recommendation(workload_status: str, task_count: int) -> str:
    """Generate workload-based recommendations for individual labourers"""
    if workload_status == 'available':
        return "Available for new task assignments"
    elif workload_status == 'light_load':
        return "Can take on additional tasks"
    elif workload_status == 'moderate_load':
        return "Well-balanced workload"
    elif workload_status == 'heavy_load':
        return "Consider task priority before adding more work"
    else:  # overloaded
        return "Should redistribute some tasks to other available workers"

def _generate_labour_recommendations(workload_breakdown: dict, skill_breakdown: dict, total_count: int) -> List[str]:
    """Generate actionable recommendations based on labour analysis"""
    recommendations = []
    
    # Workload balance recommendations
    available = workload_breakdown.get('available', 0)
    overloaded = workload_breakdown.get('overloaded', 0)
    
    if available > 0 and overloaded > 0:
        recommendations.append(f"Consider redistributing tasks from {overloaded} overloaded workers to {available} available workers")
    elif available > total_count * 0.3:
        recommendations.append(f"You have {available} available workers ready for new assignments")
    elif overloaded > total_count * 0.2:
        recommendations.append(f"Warning: {overloaded} workers are overloaded and may need support")
    
    # Skill distribution recommendations
    if len(skill_breakdown) < 3:
        recommendations.append("Consider diversifying skills - you may need workers with complementary abilities")
    
    most_common_skill = max(skill_breakdown, key=skill_breakdown.get) if skill_breakdown else None
    if most_common_skill and skill_breakdown[most_common_skill] > total_count * 0.5:
        recommendations.append(f"High concentration of {most_common_skill}s - consider cross-training in other skills")
    
    return recommendations

# ==============================================================================
# EMPLOYEE MANAGEMENT TOOLS (CREATE/UPDATE)
# ==============================================================================

@tool
def create_employee_intelligent(
    query: str, 
    name: str = "", 
    email: str = "", 
    role: str = "", 
    department: str = "", 
    phone_number: str = "", 
    address: str = ""
) -> Dict[str, Any]:
    """
    Intelligent employee creation tool with validation and human-loop guidance.
    
    Handles natural language requests like:
    - "Create new employee John Doe"
    - "Add Sarah Smith to marketing team as leader"
    - "I need to add a new intern to AI department"
    
    Parameters:
    - query: Natural language request for employee creation
    - name: Employee full name (required)
    - email: Employee email address (required) 
    - role: Employee role - must be 'hr', 'leader', or 'employee'
    - department: Department name or ID
    - phone_number: Phone number (optional)
    - address: Address (optional)
    
    Returns:
    - success: bool
    - message: Human-readable result
    - response_type: Type of response (form_request, validation_error, success_created, etc.)
    - form_data: Data for frontend form if needed
    - follow_up_questions: Suggested next actions
    """
    try:
        # If no direct parameters provided, parse from natural language query
        if not name and not email:
            parsed_data = _parse_employee_creation_query(query)
            name = parsed_data.get('name', '')
            email = parsed_data.get('email', '')
            role = parsed_data.get('role', '')
            department = parsed_data.get('department', '')
        
        # Check if we have enough information to proceed
        missing_fields = []
        if not name:
            missing_fields.append("name")
        if not email:
            missing_fields.append("email")
        if not role:
            missing_fields.append("role")
        if not department:
            missing_fields.append("department")
        
        # If missing required fields, request form
        if missing_fields:
            departments_data = _get_available_departments()
            form_data = {
                "pre_filled": {
                    "name": name,
                    "email": email,
                    "role": role,
                    "department": department,
                    "phone_number": phone_number,
                    "address": address
                },
                "departments": departments_data,
                "roles": [
                    {"value": "hr", "label": "HR Staff"},
                    {"value": "leader", "label": "Team Leader"},
                    {"value": "employee", "label": "Employee"}
                ],
                "required_fields": ["name", "email", "role", "department"]
            }
            
            return _create_standardized_form_response("create_employee", "", form_data)
        
        # Validate provided data
        validation_result = _validate_employee_data(name, email, role, department, "create")
        if not validation_result["valid"]:
            return {
                "success": False,
                "message": validation_result["message"],
                "response_type": "validation_error",
                "suggestions": validation_result["suggestions"],
                "follow_up_questions": ["Would you like to correct the information and try again?"]
            }
        
        # Check for duplicate email
        existing_employee = execute_query(
            "SELECT id, name FROM employees WHERE LOWER(email) = LOWER(%s)",
            (email,)
        )
        
        if existing_employee:
            return {
                "success": False,
                "message": f"Email '{email}' is already registered to {existing_employee[0]['name']}. Each employee must have a unique email address.",
                "response_type": "validation_error",
                "suggestions": [
                    "Use a different email address",
                    "Check if this person already exists in the system",
                    "Contact IT if this employee needs a new email"
                ],
                "follow_up_questions": [f"Would you like to view {existing_employee[0]['name']}'s profile instead?"]
            }
        
        # Get department ID
        dept_id = _get_department_id(department)
        if not dept_id:
            return {
                "success": False,
                "message": f"Department '{department}' not found. Available departments: {', '.join([d['name'] for d in _get_available_departments()])}",
                "response_type": "validation_error",
                "suggestions": ["Check the department name spelling", "Choose from available departments"],
                "follow_up_questions": ["Would you like to see all available departments?"]
            }
        
        # Create the employee
        execute_query(
            """
            INSERT INTO employees (name, email, role, department_id, phone_number, address, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (name, email, role, dept_id, phone_number or None, address or None, True, datetime.now(), datetime.now()),
            fetch=False
        )
        
        # Retrieve the created employee with department info
        created_employee = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE e.email = %s
            """,
            (email,)
        )[0]
        
        # Create success message
        success_message = f"✅ Successfully created employee record for **{name}**!\n"
        success_message += f"📧 Email: {email}\n"
        success_message += f"🏢 Department: {created_employee['department_name']}\n"
        success_message += f"👤 Role: {role.title()}\n"
        success_message += f"🆔 Employee ID: {created_employee['id']}"
        
        if phone_number:
            success_message += f"\n📞 Phone: {phone_number}"
        if address:
            success_message += f"\n🏠 Address: {address}"
        
        return {
            "success": True,
            "message": success_message,
            "response_type": "success_created",
            "data": created_employee,
            "follow_up_questions": [
                f"Would you like to set up attendance tracking for {name}?",
                "Need to create another employee record?",
                f"Want to see all employees in {created_employee['department_name']} department?"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating employee record: {str(e)}",
            "response_type": "error",
            "follow_up_questions": ["Would you like to try again with the employee information?"]
        }

@tool
def update_employee_intelligent(
    query: str,
    employee_identifier: str = "",
    field_updates: Dict[str, Any] = None,
    update_reason: str = ""
) -> Dict[str, Any]:
    """
    UPDATE ONLY: Intelligent employee update tool that triggers form popups for editing employee information.
    
    USE THIS TOOL FOR UPDATE QUERIES ONLY:
    - "update [NAME] details" / "change [NAME]" 
    - "modify [NAME]" / "edit [NAME]"
    - "update [NAME]'s phone" / "change [NAME]'s role"
    - "move [NAME] to marketing" / "promote [NAME]"
    
    IMPORTANT: This tool ALWAYS triggers form popups for editing employee information.
    
    Do NOT use for:
    - "give me [NAME] details" (use query_employee_details_intelligent instead)
    - "show me [NAME]" (use query_employee_details_intelligent instead)
    - "tell me about [NAME]" (use query_employee_details_intelligent instead)
    
    Always opens update forms when employee is found.
    """
    try:
        # Initialize field_updates if None
        if field_updates is None:
            field_updates = {}
            
        # Parse the query to extract employee identifier and intended updates
        if not employee_identifier and not field_updates:
            parsed_data = _parse_employee_update_query(query)
            employee_identifier = parsed_data.get('employee_identifier', '')
            field_updates = parsed_data.get('field_updates', {})
            update_reason = parsed_data.get('update_reason', update_reason)
            
        # CRITICAL: Always show form for update requests - don't return early with text responses
        
        # Search for the employee
        if not employee_identifier:
            return {
                "success": False,
                "message": "I need to know which employee you want to update. Please provide their name, email, or employee ID.",
                "response_type": "missing_identifier",
                "follow_up_questions": [
                    "Which employee would you like to update?",
                    "You can specify by name, email, or employee ID"
                ]
            }
        
        # Find the employee
        search_result = _search_employee_for_update(employee_identifier)
        
        if search_result["type"] == "not_found":
            # Employee not found - check for similar names first
            if not employee_identifier.isdigit() and '@' not in employee_identifier:
                # Only do fuzzy matching for name searches, not IDs or emails
                similar_employees = _find_similar_employees(employee_identifier, threshold=0.6)
                
                if similar_employees:
                    # Found similar employees - ask human loop question
                    best_match = similar_employees[0]
                    
                    return {
                        "success": True,
                        "message": f"User '{employee_identifier}' not found in our database, but similar name '{best_match['name']}' appears in our database. Do you need to update '{best_match['name']}' employee details?",
                        "response_type": "human_loop_question", 
                        "conversation_state": {
                            "pending_action": "update_employee",
                            "alternative_action": "create_employee",
                            "suggested_data": {
                                "employee": best_match,
                                "searched_name": employee_identifier,
                                "suggested_name": employee_identifier,
                                "all_similar": similar_employees
                            }
                        },
                        "follow_up_questions": [
                            f"Yes, update {best_match['name']} details",
                            f"No, create new employee '{employee_identifier}'"
                        ]
                    }
            
            # No similar employees found or searching by ID/email - ask human loop question for creation
            return {
                "success": True,
                "message": f"I couldn't find an employee matching '{employee_identifier}' in our database.\n\nWould you like me to create a new employee record for them?",
                "response_type": "human_loop_question",
                "conversation_state": {
                    "pending_action": "create_employee",
                    "suggested_data": {
                        "suggested_name": employee_identifier if not employee_identifier.isdigit() else ""
                    }
                },
                "follow_up_questions": [
                    "Yes, create a new employee record",
                    "No, let me search for a different employee"
                ]
            }
        
        elif search_result["type"] == "multiple_matches":
            # Multiple employees found
            matches_list = []
            for emp in search_result["employees"]:
                matches_list.append(f"• **{emp['name']}** ({emp['role']}) - {emp['department_name']} | ID: {emp['id']}")
            
            return {
                "success": True,
                "message": f"Found multiple employees matching '{employee_identifier}':\n\n" + "\n".join(matches_list) + "\n\nWhich employee did you mean?",
                "response_type": "multiple_matches",
                "data": search_result["employees"],
                "follow_up_questions": [
                    "Please specify the exact employee ID or full name",
                    "Tell me which department they work in",
                    "Use their email address for exact match"
                ]
            }
        
        elif search_result["type"] == "single_match":
            employee = search_result["employee"]
            
            # If no specific field updates provided, show update form
            if not field_updates:
                departments_data = _get_available_departments()
                # Fetch complete employee data from database for pre-population
                complete_employee = _get_complete_employee_data(employee['id'])
                
                form_data = {
                    "current_values": {
                        "name": complete_employee.get('name', ''),
                        "email": complete_employee.get('email', ''),
                        "role": complete_employee.get('role', ''),
                        "department": complete_employee.get('department_name', ''),
                        "phone_number": complete_employee.get('phone_number', ''),
                        "address": complete_employee.get('address', '')
                    },
                    "employee_id": complete_employee.get('id'),
                    "departments": departments_data,
                    "roles": [
                        {"value": "hr", "label": "HR Staff"},
                        {"value": "leader", "label": "Team Leader"},
                        {"value": "employee", "label": "Employee"}
                    ]
                }
                
                response = _create_standardized_form_response("update_employee", "", form_data, "employee_found_update_form")
                response["data"] = complete_employee  # Use complete employee data
                response["follow_up_questions"] = [
                    "Which information would you like to update?",
                    "Need to change their role or department?",
                    "Want to update contact information?"
                ]
                return response
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error updating employee: {str(e)}",
            "response_type": "error",
            "follow_up_questions": ["Would you like to try the update again?"]
        }

@tool
def search_employee_for_management(query: str) -> Dict[str, Any]:
    """
    Search for employees in the context of management operations (create/update).
    This tool helps identify employees for update operations or detect when creation is needed.
    
    Parameters:
    - query: Employee name, email, ID, or natural language search query
    
    Returns:
    - Type of match found (single, multiple, not_found)
    - Employee data if found
    - Suggestions for next steps
    """
    try:
        search_result = _search_employee_for_update(query)
        
        if search_result["type"] == "not_found":
            return {
                "success": True,
                "message": f"No employee found matching '{query}'.",
                "response_type": "not_found",
                "suggested_action": "create_employee",
                "follow_up_questions": [
                    f"Would you like to create a new employee record for '{query}'?",
                    "Try searching with a different name or email",
                    "Show me all employees to find who you're looking for"
                ]
            }
        
        elif search_result["type"] == "multiple_matches":
            matches_list = []
            for emp in search_result["employees"]:
                matches_list.append(f"• **{emp['name']}** ({emp['role']}) - {emp['department_name']} | {emp['email']}")
            
            return {
                "success": True,
                "message": f"Found multiple employees matching '{query}':\n\n" + "\n".join(matches_list),
                "response_type": "multiple_matches",
                "data": search_result["employees"],
                "follow_up_questions": [
                    "Which specific employee did you mean?",
                    "Please provide more specific information",
                    "Use their email address for exact match"
                ]
            }
        
        else:  # single_match
            employee = search_result["employee"]
            return {
                "success": True,
                "message": f"Found **{employee['name']}** ({employee['role']} in {employee['department_name']}).",
                "response_type": "single_match",
                "data": employee,
                "follow_up_questions": [
                    f"Would you like to update {employee['name']}'s information?",
                    f"Need to see {employee['name']}'s complete profile?",
                    f"Want to see other employees in {employee['department_name']}?"
                ]
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error searching for employee: {str(e)}",
            "response_type": "error"
        }

@tool
def get_form_data_for_frontend() -> Dict[str, Any]:
    """
    Get structured data needed for frontend forms (departments, roles, etc.).
    This tool provides the dropdown options and validation data for employee forms.
    
    Returns:
    - Available departments with IDs and names
    - Available roles with values and labels  
    - Form validation rules
    - Field requirements
    """
    try:
        departments_data = _get_available_departments()
        
        return {
            "success": True,
            "message": "Form data retrieved successfully",
            "data": {
                "departments": departments_data,
                "roles": [
                    {"value": "hr", "label": "HR Staff"},
                    {"value": "leader", "label": "Team Leader"},
                    {"value": "employee", "label": "Employee"}
                ],
                "validation_rules": {
                    "name": {"required": True, "min_length": 2, "max_length": 100},
                    "email": {"required": True, "format": "email", "unique": True},
                    "role": {"required": True, "allowed_values": ["hr", "leader", "employee"]},
                    "department": {"required": True},
                    "phone_number": {"required": False, "format": "phone"},
                    "address": {"required": False, "max_length": 500}
                },
                "field_labels": {
                    "name": "Full Name",
                    "email": "Email Address", 
                    "role": "Employee Role",
                    "department": "Department",
                    "phone_number": "Phone Number",
                    "address": "Address"
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error retrieving form data: {str(e)}",
            "data": {}
        }

# ==============================================================================
# EMPLOYEE MANAGEMENT HELPER FUNCTIONS
# ==============================================================================

def _parse_employee_creation_query(query: str) -> Dict[str, str]:
    """Enhanced parsing for natural language employee creation queries"""
    import re
    
    result = {"name": "", "email": "", "role": "", "department": ""}
    query_lower = query.lower()
    
    # Enhanced name extraction patterns
    name_patterns = [
        r'create.*(?:employee|user)\s+([A-Za-z\s]+?)(?:\s+(?:to|as|in|for|with)|$)',
        r'add\s+(?:new\s+)?(?:employee|user\s+)?([A-Za-z\s]+?)\s+(?:to|as|in)',
        r'new\s+(?:employee|user)\s+([A-Za-z\s]+?)(?:\s+(?:to|as|in|for)|$)',
        r'(?:employee|user)\s+([A-Za-z\s]+?)(?:\s+(?:to|as|in)|$)',
        r'add\s+([A-Za-z\s]+?)(?:\s+(?:employee|user|to|as))',
        r'create\s+([A-Za-z\s]+?)(?:\s+(?:employee|user|as|to))',
        # Catch standalone names after common triggers
        r'(?:add|create|new)\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_name = match.group(1).strip()
            # Filter out common words that aren't names
            exclude_words = ['employee', 'user', 'new', 'person', 'staff', 'member', 'details', 'information']
            if not any(word in potential_name.lower() for word in exclude_words) and 1 <= len(potential_name.split()) <= 4:
                result["name"] = potential_name
                break
    
    # Enhanced email extraction
    email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    email_match = re.search(email_pattern, query)
    if email_match:
        result["email"] = email_match.group(1)
    
    # Enhanced role detection
    role_patterns = {
        'leader': [r'\bleader\b', r'\bmanager\b', r'\bsupervisor\b', r'\bhead\b', r'\blead\b'],
        'hr': [r'\bhr\b', r'\bhuman\s+resources\b', r'\bhr\s+staff\b'],
        'employee': [r'\bemployee\b', r'\bstaff\b', r'\bmember\b', r'\bworker\b']
    }
    
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                result["role"] = role
                break
        if result["role"]:
            break
    
    # Enhanced department detection
    department_patterns = {
        'AI': [r'\bai\b', r'\bartificial\s+intelligence\b', r'\bai\s+department\b'],
        'Marketing': [r'\bmarketing\b', r'\bmarketing\s+department\b', r'\bmarketing\s+team\b'],
        'HR': [r'\bhr\s+department\b', r'\bhuman\s+resources\b', r'\bhr\s+team\b'],
        'IT': [r'\bit\b', r'\bit\s+department\b', r'\binformation\s+technology\b', r'\btech\b'],
        'Construction': [r'\bconstruction\b', r'\bconstruction\s+department\b', r'\bconstruction\s+team\b'],
        'Finance': [r'\bfinance\b', r'\bfinance\s+department\b', r'\baccounting\b']
    }
    
    for dept, patterns in department_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                result["department"] = dept
                break
        if result["department"]:
            break
    
    return result

def _parse_employee_update_query(query: str) -> Dict[str, Any]:
    """Enhanced parsing for natural language employee update queries"""
    import re
    
    result = {"employee_identifier": "", "field_updates": {}, "update_reason": ""}
    query_lower = query.lower()
    
    # Enhanced employee identifier extraction with more patterns
    identifier_patterns = [
        r'update\s+([A-Za-z\s]+?)(?:\s+(?:details|info|information)|\'s|$)',
        r'change\s+([A-Za-z\s]+?)(?:\s+(?:details|info|information)|\'s|$)', 
        r'modify\s+([A-Za-z\s]+?)(?:\s+(?:details|info|information)|\'s|$)',
        r'edit\s+([A-Za-z\s]+?)(?:\s+(?:details|info|information)|\'s|$)',
        # Handle "update John Smith" or "edit John Smith details"
        r'(?:update|change|modify|edit)\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)',
        # Handle "[NAME] details" patterns  
        r'([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)\s*(?:details|info|information)',
        # Handle "employee/user [NAME]" patterns
        r'(?:user|employee|staff)\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)',
        # Handle contextual patterns like "for John Smith"
        r'(?:for|of)\s+([A-Za-z]{2,}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]+)?)',
        # Fallback: any sequence that looks like a full name
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    ]
    
    for pattern in identifier_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_name = match.group(1).strip()
            # Filter out common words that aren't names
            exclude_words = ['employee', 'user', 'staff', 'details', 'info', 'information', 'data', 'record']
            if not any(word in potential_name.lower() for word in exclude_words) and 1 <= len(potential_name.split()) <= 4:
                result["employee_identifier"] = potential_name
                break
    
    # Enhanced field updates extraction
    if 'phone' in query_lower:
        phone_patterns = [
            r'phone.*?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'phone\s+number.*?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, query)
            if phone_match:
                result["field_updates"]["phone_number"] = phone_match.group(1)
                break
    
    if 'email' in query_lower:
        email_match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', query)
        if email_match:
            result["field_updates"]["email"] = email_match.group(1)
    
    # Enhanced role detection
    if any(phrase in query_lower for phrase in ['role to leader', 'role to manager', 'make leader', 'promote to leader']):
        result["field_updates"]["role"] = "leader"
    elif any(phrase in query_lower for phrase in ['role to hr', 'move to hr', 'hr role']):
        result["field_updates"]["role"] = "hr"
    elif any(phrase in query_lower for phrase in ['role to employee', 'regular employee', 'staff member']):
        result["field_updates"]["role"] = "employee"
    
    # Enhanced department detection
    department_patterns = {
        'AI': [r'\bai\b', r'\bartificial\s+intelligence\b', r'\bai\s+department\b'],
        'Marketing': [r'\bmarketing\b', r'\bmarketing\s+department\b'],
        'HR': [r'\bhr\s+department\b', r'\bhuman\s+resources\b'],
        'IT': [r'\bit\b', r'\bit\s+department\b', r'\btech\b'],
        'Construction': [r'\bconstruction\b', r'\bconstruction\s+department\b'],
        'Finance': [r'\bfinance\b', r'\bfinance\s+department\b']
    }
    
    for dept, patterns in department_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                result["field_updates"]["department"] = dept
                break
        if result["field_updates"].get("department"):
            break
    
    # Extract reason for update
    if 'because' in query_lower:
        reason_match = re.search(r'because\s+(.+)', query_lower)
        if reason_match:
            result["update_reason"] = reason_match.group(1).strip()
    
    return result

def _get_complete_employee_data(employee_id: int) -> Dict[str, Any]:
    """Fetch complete employee data from database for form pre-population"""
    try:
        query = """
            SELECT e.id, e.name, e.email, e.role, e.phone_number, e.address, 
                   e.is_active, d.name as department_name, d.id as department_id
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE e.id = %s
        """
        
        employees = execute_query(query, [employee_id])
        
        if employees:
            employee = dict(employees[0])
            return {
                "id": employee['id'],
                "name": employee['name'],
                "email": employee['email'],
                "role": employee['role'],
                "phone_number": employee.get('phone_number', ''),
                "address": employee.get('address', ''),
                "department_name": employee['department_name'],
                "department_id": employee['department_id'],
                "is_active": employee.get('is_active', True)
            }
        else:
            return {}
            
    except Exception as e:
        print(f"Error fetching complete employee data: {e}")
        return {}

def _create_standardized_form_response(action: str, message: str, form_data: Dict[str, Any], response_type: str = "form_request") -> Dict[str, Any]:
    """Create a standardized form response format for consistent frontend handling - FORM ONLY, NO TEXT MESSAGES"""
    return {
        "success": True,
        "message": "",  # Empty message to prevent text display - only form popup
        "response_type": response_type,
        "action": action,
        "form_data": form_data,
        "follow_up_questions": []
    }

def _search_employee_for_update(identifier: str) -> Dict[str, Any]:
    """Search for employee by various identifiers for update operations"""
    try:
        employees = []
        
        if identifier.strip().isdigit():
            # Search by ID
            employees = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE e.id = %s AND e.is_active = true
                """,
                (int(identifier.strip()),)
            )
        elif '@' in identifier:
            # Search by email
            employees = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE LOWER(e.email) = LOWER(%s) AND e.is_active = true
                """,
                (identifier.strip(),)
            )
        else:
            # Search by name (fuzzy)
            employees = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE LOWER(e.name) LIKE LOWER(%s) AND e.is_active = true
                ORDER BY 
                    CASE WHEN LOWER(e.name) = LOWER(%s) THEN 1 ELSE 2 END,
                    LENGTH(e.name) ASC
                """,
                (f"%{identifier.strip()}%", identifier.strip())
            )
        
        if not employees:
            return {"type": "not_found"}
        elif len(employees) == 1:
            return {"type": "single_match", "employee": employees[0]}
        else:
            return {"type": "multiple_matches", "employees": employees}
            
    except Exception as e:
        return {"type": "error", "message": str(e)}

def _calculate_string_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein distance
    Returns a similarity score between 0 and 1 (1 being identical)
    """
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Normalize strings for comparison
    s1_norm = str1.lower().strip()
    s2_norm = str2.lower().strip()
    
    # Handle exact matches
    if s1_norm == s2_norm:
        return 1.0
    
    # Calculate Levenshtein distance
    distance = levenshtein_distance(s1_norm, s2_norm)
    max_length = max(len(s1_norm), len(s2_norm))
    
    # Convert distance to similarity score (0-1)
    if max_length == 0:
        return 1.0
    
    similarity = 1 - (distance / max_length)
    return similarity

def _find_similar_employees(search_name: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Find employees with similar names to the search term
    Returns employees with similarity score above threshold, sorted by similarity
    """
    try:
        # Get all active employees
        all_employees = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE e.is_active = true
            ORDER BY e.name
            """
        )
        
        similar_employees = []
        search_name_lower = search_name.lower().strip()
        
        for employee in all_employees:
            employee_name_lower = employee['name'].lower().strip()
            
            # First check if it's an exact match (should never happen if this function is called correctly)
            if search_name_lower == employee_name_lower:
                # This is actually an exact match, should have been caught earlier
                # Return with perfect score but log the issue
                employee_with_similarity = dict(employee)
                employee_with_similarity['similarity_score'] = 1.0
                return [employee_with_similarity]  # Return immediately
            
            # Calculate similarity for non-exact matches
            similarity = _calculate_string_similarity(search_name, employee['name'])
            if similarity >= threshold:
                employee_with_similarity = dict(employee)
                employee_with_similarity['similarity_score'] = similarity
                similar_employees.append(employee_with_similarity)
        
        # Sort by similarity score (highest first)
        similar_employees.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_employees[:3]  # Return top 3 matches
        
    except Exception as e:
        logger.error(f"Error finding similar employees: {str(e)}")
        return []

def _detect_confirmation_intent(message: str) -> bool:
    """
    Detect if user message is confirming a previous human loop question
    Returns True if message indicates confirmation/agreement
    """
    message_lower = message.lower().strip()
    
    # Confirmation phrases
    confirmation_phrases = [
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed",
        "add them", "create them", "add him", "add her", "create him", "create her",
        "update them", "update him", "update her", "change them", "modify them",
        "go ahead", "do it", "please do", "continue", "correct", "right"
    ]
    
    # Positive confirmation patterns
    confirmation_patterns = [
        r"^yes\b", r"^y\b", r"^sure\b", r"^ok\b", r"^okay\b",
        r"\badd\s+(?:them|him|her|it)\b", r"\bcreate\s+(?:them|him|her|it)\b",
        r"\bupdate\s+(?:them|him|her|it)\b", r"\byes\s*,?\s*(?:add|create|update)\b",
        r"\bplease\s+(?:add|create|update)\b"
    ]
    
    # Check exact phrases
    for phrase in confirmation_phrases:
        if phrase == message_lower or message_lower.startswith(phrase + " "):
            return True
    
    # Check patterns
    import re
    for pattern in confirmation_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

def _detect_rejection_intent(message: str) -> bool:
    """
    Detect if user message is rejecting a previous human loop question
    Returns True if message indicates rejection/disagreement
    """
    message_lower = message.lower().strip()
    
    # Rejection phrases
    rejection_phrases = [
        "no", "n", "nope", "nah", "cancel", "abort", "stop", "nevermind", "never mind",
        "not now", "later", "skip", "ignore", "different", "other", "another"
    ]
    
    # Check exact phrases
    for phrase in rejection_phrases:
        if phrase == message_lower or message_lower.startswith(phrase + " "):
            return True
    
    return False

def _build_create_form_response(conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a form_request response for creating a new employee
    Returns form data with departments, roles, and pre-filled suggested name
    """
    try:
        suggested_data = conversation_state.get('suggested_data', {})
        suggested_name = suggested_data.get('suggested_name', '')
        
        # Get available departments and roles
        departments_data = _get_available_departments()
        
        form_data = {
            "pre_filled": {
                "name": suggested_name,
                "email": "",
                "role": "",
                "department": "",
                "phone_number": "",
                "address": ""
            },
            "departments": departments_data,
            "roles": [
                {"value": "hr", "label": "HR Staff"},
                {"value": "leader", "label": "Team Leader"},
                {"value": "employee", "label": "Employee"}
            ],
            "required_fields": ["name", "email", "role", "department"]
        }
        
        return {
            "success": True,
            "message": "",  # Empty message to prevent text display - only show form popup
            "response_type": "form_request",
            "action": "create_employee",
            "form_data": form_data,
            "follow_up_questions": []
        }
        
    except Exception as e:
        logger.error(f"Error building create form response: {str(e)}")
        return {
            "success": False,
            "message": f"Error preparing employee creation form: {str(e)}",
            "response_type": "error"
        }

def _build_update_form_response(conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an employee_found_update_form response for updating an existing employee
    Returns form data with current employee data pre-filled
    """
    try:
        suggested_data = conversation_state.get('suggested_data', {})
        employee_data = suggested_data.get('employee', {})
        
        if not employee_data:
            return {
                "success": False,
                "message": "No employee data found for update",
                "response_type": "error"
            }
        
        # Get complete employee data
        complete_employee = _get_complete_employee_data(employee_data.get('id'))
        
        # Get available departments and roles
        departments_data = _get_available_departments()
        
        form_data = {
            "current_values": {
                "name": complete_employee.get('name', ''),
                "email": complete_employee.get('email', ''),
                "role": complete_employee.get('role', ''),
                "department": complete_employee.get('department_name', ''),
                "phone_number": complete_employee.get('phone_number', ''),
                "address": complete_employee.get('address', '')
            },
            "employee_id": complete_employee.get('id'),
            "departments": departments_data,
            "roles": [
                {"value": "hr", "label": "HR Staff"},
                {"value": "leader", "label": "Team Leader"},
                {"value": "employee", "label": "Employee"}
            ],
            "required_fields": ["name", "email", "role", "department"]
        }
        
        return {
            "success": True,
            "message": "",  # Empty message to prevent text display - only show form popup
            "response_type": "employee_found_update_form",
            "action": "update_employee", 
            "form_data": form_data,
            "follow_up_questions": []
        }
        
    except Exception as e:
        logger.error(f"Error building update form response: {str(e)}")
        return {
            "success": False,
            "message": f"Error preparing employee update form: {str(e)}",
            "response_type": "error"
        }

def _build_employee_details_response(conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a response to show employee details (no form)
    """
    try:
        suggested_data = conversation_state.get('suggested_data', {})
        employee_data = suggested_data.get('employee', {})
        
        if not employee_data:
            return {
                "success": False,
                "message": "No employee data found to display",
                "response_type": "error"
            }
        
        return {
            "success": True,
            "message": f"**{employee_data.get('name')}** - {employee_data.get('department_name')} Department\n\n"
                     f"• **Role**: {employee_data.get('role')}\n"
                     f"• **Email**: {employee_data.get('email')}\n"
                     f"• **Phone**: {employee_data.get('phone_number', 'Not provided')}\n"
                     f"• **Address**: {employee_data.get('address', 'Not provided')}",
            "response_type": "individual_profile",
            "data": employee_data
        }
        
    except Exception as e:
        logger.error(f"Error building employee details response: {str(e)}")
        return {
            "success": False,
            "message": f"Error retrieving employee details: {str(e)}",
            "response_type": "error"
        }

def _handle_confirmation_followup(message: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle confirmed human loop actions by returning form_request directly
    (No longer calls employee tools to prevent double form triggers)
    """
    try:
        pending_action = conversation_state.get('pending_action')
        alternative_action = conversation_state.get('alternative_action')
        suggested_data = conversation_state.get('suggested_data', {})
        message_lower = message.lower()
        
        # Enhanced intent detection
        create_keywords = ['add', 'create', 'new employee', 'add employee', 'create employee', 'add them', 'create them']
        update_keywords = ['update', 'edit', 'modify', 'change', 'update them', 'edit them']
        show_keywords = ['show', 'details', 'display', 'view', 'see', 'info']
        
        wants_create = any(keyword in message_lower for keyword in create_keywords)
        wants_update = any(keyword in message_lower for keyword in update_keywords)
        wants_show = any(keyword in message_lower for keyword in show_keywords) and not wants_create and not wants_update
        
        # Decision logic based on user intent and conversation state
        if wants_create or (pending_action == 'create_employee' and not wants_show and not wants_update):
            # User wants to create new employee - return form_request directly
            return _build_create_form_response(conversation_state)
            
        elif wants_create and alternative_action == 'create_employee':
            # User chose create over existing similar employee
            return _build_create_form_response(conversation_state)
            
        elif wants_update or pending_action == 'update_employee':
            # User wants to update existing employee - return form_request directly
            return _build_update_form_response(conversation_state)
            
        elif wants_show or pending_action == 'show_employee_details':
            # User wants to see employee details - return info (no form)
            return _build_employee_details_response(conversation_state)
        
        # Default fallback - analyze primary action
        elif pending_action == 'create_employee':
            return _build_create_form_response(conversation_state)
        elif pending_action == 'update_employee':
            return _build_update_form_response(conversation_state)  
        elif pending_action == 'show_employee_details':
            return _build_employee_details_response(conversation_state)
        
        return {
            "success": False,
            "message": "I couldn't understand what you'd like to do. Please try again with 'yes' to confirm or 'no' to cancel.",
            "response_type": "clarification_needed"
        }
        
    except Exception as e:
        logger.error(f"Error handling confirmation followup: {str(e)}")
        return {
            "success": False,
            "message": f"An error occurred while processing your request: {str(e)}",
            "response_type": "error"
        }

# Global conversation state storage (in production, use Redis or database)
_conversation_states = {}

def _store_conversation_state(user_session: str, state: Dict[str, Any]) -> None:
    """Store conversation state for user session"""
    _conversation_states[user_session] = state

def _get_conversation_state(user_session: str) -> Optional[Dict[str, Any]]:
    """Retrieve conversation state for user session"""
    return _conversation_states.get(user_session)

def _clear_conversation_state(user_session: str) -> None:
    """Clear conversation state for user session"""
    if user_session in _conversation_states:
        del _conversation_states[user_session]

def _validate_employee_data(name: str, email: str, role: str, department: str, operation: str) -> Dict[str, Any]:
    """Validate employee data for creation or update"""
    errors = []
    suggestions = []
    
    # Validate name
    if not name or len(name.strip()) < 2:
        errors.append("Name is required and must be at least 2 characters")
        suggestions.append("Provide a valid full name")
    
    # Validate email
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email:
        errors.append("Email address is required")
        suggestions.append("Provide a valid email address")
    elif not re.match(email_pattern, email):
        errors.append(f"Invalid email format: '{email}'")
        suggestions.append("Use format: name@company.com")
    
    # Validate role
    valid_roles = ['hr', 'leader', 'employee']
    if role not in valid_roles:
        errors.append(f"Invalid role: '{role}'. Must be one of: {', '.join(valid_roles)}")
        suggestions.append(f"Choose from: {', '.join(valid_roles)}")
    
    # Validate department exists
    if not _get_department_id(department):
        available_depts = [d['name'] for d in _get_available_departments()]
        errors.append(f"Department '{department}' not found")
        suggestions.append(f"Choose from: {', '.join(available_depts)}")
    
    if errors:
        return {
            "valid": False,
            "message": "Please fix the following issues:\n• " + "\n• ".join(errors),
            "suggestions": suggestions
        }
    
    return {"valid": True, "message": "Validation passed"}

def _get_available_departments() -> List[Dict[str, Any]]:
    """Get list of available departments for dropdowns"""
    try:
        departments = execute_query("SELECT id, name FROM departments ORDER BY name")
        return [{"value": dept['id'], "label": dept['name'], "name": dept['name']} for dept in departments]
    except:
        return []

def _get_department_id(department_name: str) -> Optional[int]:
    """Get department ID by name (case-insensitive)"""
    try:
        if department_name.isdigit():
            return int(department_name)
        
        result = execute_query(
            "SELECT id FROM departments WHERE LOWER(name) LIKE LOWER(%s)",
            (f"%{department_name}%",)
        )
        return result[0]['id'] if result else None
    except:
        return None

def _find_employee_by_identifier(identifier: str) -> Optional[Dict[str, Any]]:
    """Find employee by ID, email, or name"""
    try:
        # Try by ID first if it's numeric
        if identifier.isdigit():
            result = execute_query(
                """
                SELECT e.*, d.name as department_name
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                WHERE e.id = %s AND e.is_active = true
                """,
                (int(identifier),)
            )
            if result:
                return result[0]
        
        # Try by email
        result = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE LOWER(e.email) = LOWER(%s) AND e.is_active = true
            """,
            (identifier,)
        )
        if result:
            return result[0]
        
        # Try by name (partial match)
        result = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE LOWER(e.name) LIKE LOWER(%s) AND e.is_active = true
            """,
            (f"%{identifier}%",)
        )
        if result:
            return result[0]
            
        return None
    except Exception as e:
        logger.error(f"Error finding employee by identifier '{identifier}': {e}")
        return None

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}."



@tool
def escalate_to_human(reason: str, context: str, urgency: str = "normal") -> Dict[str, Any]:
    """
    Escalate complex situations to human HR staff when agent cannot or should not make autonomous decisions.
    Use this when situations require human judgment, policy interpretation, or sensitive handling.
    
    reason: Why escalation is needed (e.g., "policy_interpretation", "sensitive_situation", "bulk_operation", "unusual_pattern")
    context: Detailed context about the situation
    urgency: "low", "normal", "high", "urgent"
    """
    try:
        escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Log the escalation
        logger.info(f"Human escalation triggered - ID: {escalation_id}, Reason: {reason}, Urgency: {urgency}")
        
        # Define escalation categories and appropriate responses
        escalation_responses = {
            "policy_interpretation": {
                "message": "This situation requires human review for policy interpretation. I've escalated this to HR management.",
                "next_steps": "An HR manager will review this case and provide guidance within 24 hours."
            },
            "sensitive_situation": {
                "message": "I understand this is a sensitive matter. I'm connecting you with a human HR representative who can provide appropriate support.",
                "next_steps": "A trained HR professional will reach out to you directly to discuss this confidentially."
            },
            "bulk_operation": {
                "message": "Large-scale attendance changes require management approval to ensure accuracy and compliance.",
                "next_steps": "I've submitted this bulk operation for review. You'll receive confirmation once approved."
            },
            "unusual_pattern": {
                "message": "I've detected an unusual attendance pattern that may need investigation.",
                "next_steps": "HR management will review this pattern and determine if any action is needed."
            },
            "data_inconsistency": {
                "message": "I found some data inconsistencies that require human verification.",
                "next_steps": "A data review will be conducted to resolve any discrepancies."
            }
        }
        
        response_info = escalation_responses.get(reason, {
            "message": "This situation requires human review.",
            "next_steps": "An HR representative will follow up with you."
        })
        
        # Simulate escalation workflow
        escalation_data = {
            "escalation_id": escalation_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "context": context,
            "urgency": urgency,
            "status": "pending_review",
            "assigned_to": "hr_manager" if urgency in ["high", "urgent"] else "hr_staff"
        }
        
        return {
            "success": True,
            "escalated": True,
            "escalation_id": escalation_id,
            "message": response_info["message"],
            "next_steps": response_info["next_steps"],
            "urgency": urgency,
            "estimated_response_time": "2-4 hours" if urgency == "urgent" else "4-8 hours" if urgency == "high" else "24 hours",
            "human_readable_report": f"""
🔄 **Escalated to Human Review**

**Escalation ID:** {escalation_id}
**Reason:** {reason.replace('_', ' ').title()}
**Urgency Level:** {urgency.title()}

**What's Happening:**
{response_info['message']}

**Next Steps:**
{response_info['next_steps']}

**Timeline:** You can expect a response within {escalation_data.get('estimated_response_time', '24 hours')}.

If this is urgent, please contact HR directly at hr@risetechvillage.com or extension 101.
            """.strip()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "human_readable_report": f"I encountered an issue while escalating this matter: {str(e)}. Please contact HR directly for assistance."
        }

@tool
def intelligent_decision_check(situation: str, proposed_action: str, impact_scope: str = "individual") -> Dict[str, Any]:
    """
    Analyzes whether a situation requires human intervention or can be handled autonomously.
    This tool helps the agent make smart decisions about when to escalate vs. when to proceed.
    
    situation: Description of the current situation
    proposed_action: What the agent wants to do
    impact_scope: "individual", "team", "department", "company"
    """
    try:
        # Define criteria for human intervention
        escalation_triggers = {
            "high_impact_keywords": [
                "terminate", "fire", "discipline", "investigate", "legal", "harassment", 
                "discrimination", "whistleblow", "complaint", "grievance", "misconduct"
            ],
            "bulk_operations": [
                "all employees", "entire department", "everyone", "bulk", "mass", 
                "multiple departments", "company-wide"
            ],
            "sensitive_timeframes": [
                "retroactive", "backdated", "historical correction", "past month", 
                "previous quarter", "last year"
            ],
            "policy_matters": [
                "policy violation", "policy interpretation", "new policy", "exception to policy",
                "special accommodation", "reasonable adjustment"
            ]
        }
        
        # Analyze the situation
        situation_lower = situation.lower()
        action_lower = proposed_action.lower()
        combined_text = f"{situation_lower} {action_lower}"
        
        risk_factors = []
        escalation_score = 0
        
        # Check for escalation triggers
        for category, keywords in escalation_triggers.items():
            for keyword in keywords:
                if keyword in combined_text:
                    risk_factors.append(f"{category}: {keyword}")
                    escalation_score += 2
        
        # Impact scope analysis
        scope_scores = {
            "individual": 0,
            "team": 1,
            "department": 2,
            "company": 3
        }
        escalation_score += scope_scores.get(impact_scope, 1)
        
        # Time-sensitive analysis
        if any(time_word in combined_text for time_word in ["urgent", "immediate", "asap", "emergency"]):
            escalation_score += 1
            risk_factors.append("time_sensitive: urgent request")
        
        # Decision logic
        should_escalate = escalation_score >= 3
        needs_confirmation = escalation_score >= 2 and not should_escalate
        
        # Generate recommendations
        if should_escalate:
            recommendation = "escalate_to_human"
            reason = "High-risk situation requiring human judgment"
            confidence = "high"
        elif needs_confirmation:
            recommendation = "seek_confirmation"
            reason = "Moderate-risk situation - confirm before proceeding"
            confidence = "medium"
        else:
            recommendation = "proceed_autonomously"
            reason = "Low-risk situation - safe to handle autonomously"
            confidence = "high"
        
        # Determine appropriate escalation type if needed
        escalation_type = None
        if should_escalate:
            if any("policy" in factor for factor in risk_factors):
                escalation_type = "policy_interpretation"
            elif any("bulk" in factor or "mass" in factor for factor in risk_factors):
                escalation_type = "bulk_operation"
            elif any("sensitive" in factor for factor in risk_factors):
                escalation_type = "sensitive_situation"
            else:
                escalation_type = "unusual_pattern"
        
        return {
            "success": True,
            "recommendation": recommendation,
            "should_escalate": should_escalate,
            "needs_confirmation": needs_confirmation,
            "escalation_score": escalation_score,
            "risk_factors": risk_factors,
            "escalation_type": escalation_type,
            "confidence": confidence,
            "reasoning": reason,
            "safe_to_proceed": escalation_score < 2,
            "suggested_escalation_urgency": "high" if escalation_score >= 5 else "normal" if escalation_score >= 3 else "low"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recommendation": "escalate_to_human",  # Err on the side of caution
            "reasoning": f"Error in decision analysis: {str(e)} - escalating for safety"
        }

@tool
def ask_company_documents(query: str) -> str:
    """
    Use this to answer ANY question about The Rise Tech Village that you cannot answer from your basic identity. 
    This is your primary source of knowledge about the company, its structure, its people, and its internal policies.
    """
    try:
        # Create embedding for the query
        query_embedding = embeddings_model.embed_query(query)
        
        # Search for similar documents
        docs = search_similar_documents(query_embedding, k=3)
        
        if not docs:
            return "No relevant documents found in the knowledge base."
        
        # Combine document content
        context = "\n\n".join([doc['content'] for doc in docs])
        
        # Create a response based on the context
        response = f"Based on the company documents, here's what I found:\n\n{context}"
        return response
        
    except Exception as e:
        return f"Error searching company documents: {e}"

# ==============================================================================
# UNIFIED ATTENDANCE TOOLS (CRUD Operations)
# ==============================================================================

@tool
def query_attendance_intelligent(query: str) -> Dict[str, Any]:
    """
    Intelligent attendance query tool that understands natural language requests.
    
    Handles queries like:
    - "who works today" / "today's attendance" 
    - "marketing team today's attendance" / "rise ai team department today's attendance"
    - "John's attendance" / "attendance for John Smith"
    - "who is on leave today" / "absent employees today"
    - "attendance last week" / "attendance this month"
    - "attendance from 2024-01-01 to 2024-01-31"
    
    Parameters:
    - query: Natural language query about attendance
    
    Returns:
    - success: bool
    - message: Human-readable summary
    - data: Structured attendance data
    - filters_applied: What filters were understood from the query
    """
    try:
        import re
        from datetime import datetime, timedelta, date
        
        # Initialize query parameters
        employee_name = None
        department = None
        status_filter = None
        start_date = None
        end_date = None
        days = 7
        
        query_lower = query.lower().strip()
        
        # Extract date information
        today = date.today()
        
        # Date pattern matching
        if "today" in query_lower:
            start_date = end_date = today
        elif "yesterday" in query_lower:
            yesterday = today - timedelta(days=1)
            start_date = end_date = yesterday
        elif "this week" in query_lower:
            start_date = today - timedelta(days=today.weekday())
            end_date = today
        elif "last week" in query_lower:
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            start_date = start_of_last_week
            end_date = end_of_last_week
        elif "this month" in query_lower:
            start_date = today.replace(day=1)
            end_date = today
        elif "last month" in query_lower:
            first_day_this_month = today.replace(day=1)
            last_day_last_month = first_day_this_month - timedelta(days=1)
            start_date = last_day_last_month.replace(day=1)
            end_date = last_day_last_month
        
        # Look for specific date patterns (YYYY-MM-DD)
        date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        dates_found = re.findall(date_pattern, query)
        if len(dates_found) == 1:
            start_date = end_date = datetime.strptime(dates_found[0], '%Y-%m-%d').date()
        elif len(dates_found) == 2:
            start_date = datetime.strptime(dates_found[0], '%Y-%m-%d').date()
            end_date = datetime.strptime(dates_found[1], '%Y-%m-%d').date()
        
        # Extract department information
        dept_patterns = [
            r'(?:department|team|dept)\s+(\w+)',
            r'(\w+)\s+(?:department|team|dept)',
            r'rise\s+ai\s+team',
            r'marketing\s+team',
            r'hr\s+team',
            r'tech\s+team'
        ]
        
        for pattern in dept_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if "rise ai" in query_lower or "rise ai team" in query_lower:
                    department = "rise ai"
                else:
                    department = match.group(1) if match.groups() else match.group(0)
                break
        
        # Extract employee name (look for quoted names or common name patterns)
        name_patterns = [
            r"for\s+([A-Za-z\s]+?)(?:\s|$)",
            r"([A-Za-z]+\s+[A-Za-z]+)'s",
            r'"([^"]+)"',
            r"'([^']+)'"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                potential_name = match.group(1).strip()
                # Avoid capturing common words as names
                if not any(word in potential_name.lower() for word in ['attendance', 'team', 'department', 'today', 'yesterday']):
                    employee_name = potential_name
                    break
        
        # Extract status information
        status_keywords = {
            'present': 'Present',
            'absent': ['Sudden Leave', 'Medical Leave', 'Planned Leave'],
            'leave': ['Sudden Leave', 'Medical Leave', 'Planned Leave', 'Holiday Leave', 'Lieu leave'],
            'wfh': 'Work from home',
            'work from home': 'Work from home',
            'remote': 'Work from home',
            'sick': 'Medical Leave',
            'medical leave': 'Medical Leave',
            'planned leave': 'Planned Leave',
            'holiday': 'Holiday Leave'
        }
        
        for keyword, status in status_keywords.items():
            if keyword in query_lower:
                if isinstance(status, list):
                    status_filter = status
                else:
                    status_filter = [status]
                break
        
        # If no specific date is mentioned and no "today" keyword, default to today for daily queries
        if not start_date and not end_date:
            if any(word in query_lower for word in ['who works', 'who is working', "today's"]):
                start_date = end_date = today
            else:
                # Default to last 7 days for general queries
                start_date = today - timedelta(days=7)
                end_date = today
        
        # Build SQL query
        where_conditions = []
        params = []
        
        # Date filtering
        if start_date and end_date:
            if start_date == end_date:
                where_conditions.append("a.attendance_date = %s")
                params.append(start_date)
            else:
                where_conditions.append("a.attendance_date >= %s AND a.attendance_date <= %s")
                params.extend([start_date, end_date])
        
        # Employee filtering
        if employee_name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{employee_name}%")
        
        # Department filtering
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        # Status filtering
        if status_filter:
            if len(status_filter) == 1:
                where_conditions.append("a.status = %s")
                params.append(status_filter[0])
            else:
                placeholders = ', '.join(['%s'] * len(status_filter))
                where_conditions.append(f"a.status IN ({placeholders})")
                params.extend(status_filter)
        
        # Construct final query
        base_query = """
        SELECT 
            a.id,
            a.attendance_date,
            a.status,
            a.check_in_time,
            a.check_out_time,
            a.notes,
            e.id as employee_id,
            e.name as employee_name,
            e.email as employee_email,
            e.role as employee_role,
            d.name as department_name
        FROM attendances a
        JOIN employees e ON a.employee_id = e.id
        JOIN departments d ON e.department_id = d.id
        """
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += " ORDER BY a.attendance_date DESC, e.name ASC"
        
        # Execute query
        results = execute_query(base_query, tuple(params))
        
        if not results:
            return {
                "success": True,
                "message": "No attendance records found matching your query.",
                "data": [],
                "total_records": 0,
                "filters_applied": {
                    "employee_name": employee_name,
                    "department": department,
                    "status_filter": status_filter,
                    "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                    "end_date": end_date.strftime('%Y-%m-%d') if end_date else None
                },
                "query_understood": query
            }
        
        # Generate human-readable summary
        total_records = len(results)
        unique_employees = len(set(r['employee_name'] for r in results))
        unique_dates = len(set(r['attendance_date'] for r in results))
        
        # Status breakdown
        status_counts = {}
        for record in results:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        date_range_str = ""
        if start_date and end_date:
            if start_date == end_date:
                date_range_str = f" for {start_date.strftime('%Y-%m-%d')}"
            else:
                date_range_str = f" from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        summary_parts = [f"Found {total_records} attendance records"]
        if employee_name:
            summary_parts.append(f"for employee '{employee_name}'")
        if department:
            summary_parts.append(f"in {department} department")
        summary_parts.append(date_range_str)
        
        summary = " ".join(summary_parts) + "."
        
        if status_counts:
            status_summary = ", ".join([f"{count} {status}" for status, count in status_counts.items()])
            summary += f" Status breakdown: {status_summary}."
        
        return {
            "success": True,
            "message": summary,
            "data": results,
            "total_records": total_records,
            "unique_employees": unique_employees,
            "unique_dates": unique_dates,
            "status_breakdown": status_counts,
            "filters_applied": {
                "employee_name": employee_name,
                "department": department,
                "status_filter": status_filter,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None
            },
            "query_understood": query
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing attendance query: {str(e)}",
            "data": [],
            "query_understood": query
        }



# Removed create_attendance_record and update_attendance_record functions - not implemented

@tool
def handle_attendance_create_update_request(query: str) -> Dict[str, Any]:
    """
    Handles requests for attendance creation or updates that are not yet implemented.
    
    This function provides a helpful response when users ask to create or update 
    attendance records, explaining what's available instead.
    
    Parameters:
    - query: The user's original request
    
    Returns:
    - success: bool
    - message: Human-readable response explaining the limitation
    - available_functions: List of available attendance functions
    """
    return {
        "success": False,
        "message": "Attendance creation and updates are not yet available. I can help you query existing attendance records and generate comprehensive reports instead.",
        "available_functions": [
            "Query attendance records: 'show me today's attendance', 'who was absent yesterday'",
            "Analyze attendance patterns: 'attendance trends this month', 'department attendance summary'", 
            "Generate attendance reports: 'weekly attendance report', 'AI team attendance overview'",
            "Get attendance insights: 'attendance analytics', 'identify attendance patterns'"
        ],
        "suggestion": "Try asking me to query existing attendance data or generate a report instead!"
    }

# ==============================================================================
# AGENTIC ATTENDANCE SUMMARIZATION HELPERS
# ==============================================================================

def _aggregate_by_department(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate attendance records by department with smart metrics"""
    dept_data = {}
    
    for record in records:
        dept_name = record.get('department_name', 'Unknown Department')
        if dept_name not in dept_data:
            dept_data[dept_name] = {
                'total_records': 0,
                'present_count': 0,
                'remote_count': 0,
                'leave_count': 0,
                'employees': set(),
                'status_breakdown': {},
                'dates_covered': set()
            }
        
        dept = dept_data[dept_name]
        dept['total_records'] += 1
        dept['employees'].add(record['employee_name'])
        dept['dates_covered'].add(str(record['attendance_date']))
        
        status = record['status'].lower()
        if status not in dept['status_breakdown']:
            dept['status_breakdown'][status] = 0
        dept['status_breakdown'][status] += 1
        
        # Categorize attendance types
        if 'present' in status:
            dept['present_count'] += 1
        elif 'home' in status or 'remote' in status:
            dept['remote_count'] += 1
        else:
            dept['leave_count'] += 1
    
    # Convert sets to counts for easier processing
    for dept_name, data in dept_data.items():
        data['unique_employees'] = len(data['employees'])
        data['unique_dates'] = len(data['dates_covered'])
        data['employees'] = list(data['employees'])  # Keep employee names
        del data['dates_covered']  # Remove to save memory
        
        # Calculate attendance rate
        total_working = data['present_count'] + data['remote_count']
        data['attendance_rate'] = (total_working / data['total_records'] * 100) if data['total_records'] > 0 else 0
    
    return dept_data

def _calculate_attendance_metrics(dept_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall attendance metrics and identify patterns"""
    total_records = sum(dept['total_records'] for dept in dept_data.values())
    total_employees = len(set(emp for dept in dept_data.values() for emp in dept['employees']))
    
    if total_records == 0:
        return {
            'total_records': 0,
            'total_employees': 0,
            'overall_attendance_rate': 0,
            'department_count': 0,
            'insights': []
        }
    
    # Overall attendance rate
    total_present = sum(dept['present_count'] for dept in dept_data.values())
    total_remote = sum(dept['remote_count'] for dept in dept_data.values())
    overall_rate = ((total_present + total_remote) / total_records) * 100
    
    # Department performance analysis
    dept_performance = []
    for dept_name, data in dept_data.items():
        dept_performance.append({
            'name': dept_name,
            'attendance_rate': data['attendance_rate'],
            'employees': data['unique_employees'],
            'records': data['total_records']
        })
    
    # Sort by attendance rate
    dept_performance.sort(key=lambda x: x['attendance_rate'], reverse=True)
    
    return {
        'total_records': total_records,
        'total_employees': total_employees,
        'overall_attendance_rate': overall_rate,
        'department_count': len(dept_data),
        'department_performance': dept_performance,
        'top_performing_dept': dept_performance[0] if dept_performance else None,
        'needs_attention_dept': dept_performance[-1] if dept_performance and dept_performance[-1]['attendance_rate'] < 85 else None
    }

def _generate_department_insights(dept_data: Dict[str, Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
    """Generate actionable insights from department attendance data"""
    insights = []
    
    # Overall performance insight
    overall_rate = metrics['overall_attendance_rate']
    if overall_rate >= 95:
        insights.append(f"🟢 Excellent overall attendance rate of {overall_rate:.1f}% - team is highly engaged!")
    elif overall_rate >= 85:
        insights.append(f"🟡 Good attendance rate of {overall_rate:.1f}% with room for improvement")
    else:
        insights.append(f"🔴 Attendance rate of {overall_rate:.1f}% needs immediate attention")
    
    # Department-specific insights
    if metrics['department_count'] > 1:
        top_dept = metrics['top_performing_dept']
        if top_dept and top_dept['attendance_rate'] >= 90:
            insights.append(f"🌟 {top_dept['name']} leads with {top_dept['attendance_rate']:.1f}% attendance - great team culture!")
        
        needs_attention = metrics['needs_attention_dept']
        if needs_attention:
            insights.append(f"⚠️ {needs_attention['name']} needs support with {needs_attention['attendance_rate']:.1f}% attendance")
    
    # Remote work patterns
    total_remote = sum(dept['remote_count'] for dept in dept_data.values())
    remote_rate = (total_remote / metrics['total_records']) * 100 if metrics['total_records'] > 0 else 0
    if remote_rate > 20:
        insights.append(f"🏠 {remote_rate:.1f}% remote work indicates flexible work culture")
    
    # Leave patterns
    total_leave = sum(dept['leave_count'] for dept in dept_data.values())
    leave_rate = (total_leave / metrics['total_records']) * 100 if metrics['total_records'] > 0 else 0
    if leave_rate > 15:
        insights.append(f"📋 {leave_rate:.1f}% leave rate - monitor for patterns or staffing needs")
    
    return insights

def _create_follow_up_questions(dept_data: Dict[str, Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
    """Generate relevant follow-up questions for progressive disclosure"""
    questions = []
    
    if metrics['department_count'] > 1:
        questions.append("📊 Would you like detailed breakdown for a specific department?")
        
        # Suggest departments that need attention
        needs_attention = metrics.get('needs_attention_dept')
        if needs_attention:
            questions.append(f"🔍 Should I analyze {needs_attention['name']} department's attendance patterns?")
        
        # Suggest top performing department
        top_dept = metrics.get('top_performing_dept')
        if top_dept and top_dept['attendance_rate'] >= 90:
            questions.append(f"✨ Want to see what makes {top_dept['name']} department successful?")
    
    # Suggest individual employee analysis
    total_employees = metrics['total_employees']
    if total_employees > 10:
        questions.append("👤 Need individual employee attendance details for any team members?")
    
    # Suggest trend analysis
    questions.append("📈 Would you like to compare this with previous periods?")
    
    # Pattern analysis
    questions.append("🔍 Should I identify employees with perfect attendance or concerning patterns?")
    
    return questions[:3]  # Limit to top 3 most relevant questions

@tool
def summarize_attendance_intelligent(query: str) -> Dict[str, Any]:
    """
    Intelligent attendance summarization tool for large datasets with agentic insights.
    
    Perfect for queries like:
    - "Show last month attendance summary"
    - "Department-wise attendance overview" 
    - "Attendance patterns this quarter"
    - "How is team attendance performing?"
    
    Provides smart summaries, insights, and progressive disclosure instead of overwhelming raw data.
    
    Parameters:
    - query: Natural language query about attendance patterns or summaries
    
    Returns:
    - success: bool
    - message: Executive summary with key insights
    - department_breakdown: High-level department metrics
    - insights: Agentic business intelligence
    - follow_up_questions: Suggested next steps
    - data_scope: Information about the data analyzed
    """
    try:
        # Use existing query tool to get raw data
        raw_result = query_attendance_intelligent(query)
        
        if not raw_result.get('success') or not raw_result.get('data'):
            return {
                "success": True,
                "message": "No attendance data found matching your query. This could be a weekend, holiday, or the date range has no records.",
                "insights": ["Consider checking a different date range or ensuring attendance has been marked."],
                "follow_up_questions": ["Would you like me to check recent attendance activity?"],
                "data_scope": {"total_records": 0, "date_range": "No data"}
            }
        
        raw_data = raw_result['data']
        total_records = len(raw_data)
        
        # Context-aware processing
        if total_records > 50:  # Smart threshold for summarization
            # Aggregate data by department
            dept_data = _aggregate_by_department(raw_data)
            metrics = _calculate_attendance_metrics(dept_data)
            insights = _generate_department_insights(dept_data, metrics)
            follow_ups = _create_follow_up_questions(dept_data, metrics)
            
            # Create executive summary
            date_info = raw_result.get('filters_applied', {})
            date_range = f"{date_info.get('start_date', 'N/A')} to {date_info.get('end_date', 'N/A')}"
            
            summary = f"""
📊 **Attendance Intelligence Summary**

**Data Scope**: {total_records} records across {metrics['department_count']} departments ({metrics['total_employees']} employees)
**Period**: {date_range}
**Overall Performance**: {metrics['overall_attendance_rate']:.1f}% attendance rate

**Department Highlights**:
"""
            
            # Add top 3 department performances
            for i, dept in enumerate(metrics['department_performance'][:3]):
                status_icon = "🟢" if dept['attendance_rate'] >= 90 else "🟡" if dept['attendance_rate'] >= 80 else "🔴"
                summary += f"  {status_icon} {dept['name']}: {dept['attendance_rate']:.1f}% ({dept['employees']} employees)\n"
            
            if len(metrics['department_performance']) > 3:
                summary += f"  ... and {len(metrics['department_performance']) - 3} more departments\n"
            
            return {
                "success": True,
                "message": summary.strip(),
                "department_breakdown": {dept_name: {
                    'attendance_rate': f"{data['attendance_rate']:.1f}%",
                    'employees': data['unique_employees'],
                    'total_records': data['total_records'],
                    'working_count': data['present_count'] + data['remote_count'],
                    'leave_count': data['leave_count']
                } for dept_name, data in dept_data.items()},
                "insights": insights,
                "follow_up_questions": follow_ups,
                "data_scope": {
                    "total_records": total_records,
                    "date_range": date_range,
                    "departments": metrics['department_count'],
                    "employees": metrics['total_employees'],
                    "processing_mode": "smart_summary"
                },
                "metrics": metrics
            }
        
        else:
            # For smaller datasets, provide enhanced but detailed response
            insights = [f"📋 Found {total_records} attendance records - small dataset, showing enhanced details"]
            
            return {
                "success": True,
                "message": f"Enhanced attendance details for {total_records} records:\n\n" + raw_result['message'],
                "detailed_data": raw_data[:20],  # Show up to 20 records
                "insights": insights,
                "follow_up_questions": [
                    "Need more detailed analysis of specific employees?",
                    "Would you like to see patterns or trends in this data?"
                ],
                "data_scope": {
                    "total_records": total_records,
                    "date_range": f"{raw_result.get('filters_applied', {}).get('start_date', 'N/A')} to {raw_result.get('filters_applied', {}).get('end_date', 'N/A')}",
                    "processing_mode": "enhanced_details"
                }
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error generating attendance summary: {str(e)}",
            "insights": ["There was an issue processing your attendance query. Please try rephrasing your request."],
            "follow_up_questions": ["Would you like to try a more specific date range or department query?"]
        }

# ==============================================================================
# LANGCHAIN GRAPH SETUP
# ==============================================================================

@tool
def handle_human_loop_confirmation(
    message: str,
    user_session: str = "default"
) -> Dict[str, Any]:
    """
    Handle user confirmations for human loop questions.
    
    This tool processes user responses to human loop questions like:
    - "Yes, add the employee" -> triggers employee creation
    - "Yes, update them" -> triggers employee update
    - "Show their details" -> shows employee information
    
    Parameters:
    - message: User's confirmation or response message
    - user_session: User session ID for conversation state tracking
    
    Returns:
    - Appropriate action result (form request, employee details, etc.)
    """
    try:
        # Get conversation state for this user session
        conversation_state = _get_conversation_state(user_session)
        
        if not conversation_state:
            return {
                "success": False,
                "message": "I don't have any pending questions for you. Please start a new request.",
                "response_type": "no_conversation_state"
            }
        
        # Check if user is confirming
        if _detect_confirmation_intent(message):
            # Handle the confirmation
            result = _handle_confirmation_followup(message, conversation_state)
            # Clear conversation state after handling
            _clear_conversation_state(user_session)
            return result
        
        # Check if user is rejecting
        elif _detect_rejection_intent(message):
            # Clear conversation state and acknowledge
            _clear_conversation_state(user_session)
            return {
                "success": True,
                "message": "Understood. Let me know if you need anything else!",
                "response_type": "acknowledgment"
            }
        
        # User response doesn't match confirmation/rejection patterns
        else:
            return {
                "success": True,
                "message": "I'm not sure what you'd like to do. Could you please respond with 'yes' to confirm or 'no' to cancel?",
                "response_type": "clarification_needed",
                "conversation_state": conversation_state,  # Keep state active
                "follow_up_questions": ["Yes, proceed", "No, cancel"]
            }
            
    except Exception as e:
        logger.error(f"Error handling human loop confirmation: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing your response: {str(e)}",
            "response_type": "error"
        }

tools = [
    # Enhanced employee query tools
    query_employee_details_intelligent, get_department_employee_summary, get_individual_employee_profile,
    # Employee management tools (create/update)
    create_employee_intelligent, update_employee_intelligent, search_employee_for_management, get_form_data_for_frontend,
    # Human loop confirmation handling
    handle_human_loop_confirmation,
    # Unified attendance tools
    query_attendance_intelligent, summarize_attendance_intelligent, handle_attendance_create_update_request,
    # Enhanced task query tools
    query_tasks_intelligent,
    # Enhanced labour query tools
    query_labourers_intelligent,
    # Non-attendance functions
    get_all_employees_overview, get_tasks_report, get_current_datetime,
    escalate_to_human, intelligent_decision_check, ask_company_documents
]

# Initialize conversation checkpointer for memory (using MemorySaver for compatibility)
memory = MemorySaver()

# ==============================================================================
# TOKEN TRACKING AND COST MONITORING
# ==============================================================================

class TokenTracker:
    """Track token usage and costs for monitoring and optimization"""
    
    def __init__(self):
        self.session_tokens = {"input": 0, "output": 0, "total": 0}
        self.session_cost = 0.0
        self.conversation_count = 0
        
        # OpenAI pricing (per 1K tokens) - update as needed
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-3-small": {"input": 0.00002, "output": 0.0}
        }
    
    def track_conversation(self, model_name: str, input_tokens: int, output_tokens: int, conversation_id: str = None):
        """Track token usage for a conversation"""
        try:
            self.session_tokens["input"] += input_tokens
            self.session_tokens["output"] += output_tokens
            self.session_tokens["total"] += (input_tokens + output_tokens)
            self.conversation_count += 1
            
            # Calculate cost
            if model_name in self.pricing:
                input_cost = (input_tokens / 1000) * self.pricing[model_name]["input"]
                output_cost = (output_tokens / 1000) * self.pricing[model_name]["output"]
                conversation_cost = input_cost + output_cost
                self.session_cost += conversation_cost
                
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Cost: ${conversation_cost:.6f}")
                
                # Check for usage alerts
                if LangChainConfig.COST_TRACKING_ENABLED:
                    self._check_usage_alerts()
                
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost": conversation_cost,
                    "session_total_tokens": self.session_tokens["total"],
                    "session_total_cost": self.session_cost
                }
            else:
                logger.warning(f"No pricing info available for model: {model_name}")
                return {"input_tokens": input_tokens, "output_tokens": output_tokens}
                
        except Exception as e:
            logger.error(f"Error tracking tokens: {e}")
            return {}
    
    def _check_usage_alerts(self):
        """Check if usage exceeds configured thresholds"""
        try:
            if self.session_tokens["total"] > (LangChainConfig.MAX_MONTHLY_TOKENS * LangChainConfig.ALERT_THRESHOLD):
                logger.warning(f"High token usage alert: {self.session_tokens['total']} tokens used (${self.session_cost:.2f})")
            
            # Check for unusual single-conversation usage
            if self.conversation_count > 0:
                avg_tokens_per_conversation = self.session_tokens["total"] / self.conversation_count
                if avg_tokens_per_conversation > 10000:  # Alert for conversations using >10k tokens
                    logger.warning(f"High per-conversation usage: {avg_tokens_per_conversation:.0f} avg tokens per conversation")
                    
        except Exception as e:
            logger.error(f"Error checking usage alerts: {e}")
    
    def get_session_summary(self):
        """Get summary of current session usage"""
        return {
            "conversations": self.conversation_count,
            "total_tokens": self.session_tokens["total"],
            "input_tokens": self.session_tokens["input"],
            "output_tokens": self.session_tokens["output"],
            "estimated_cost": self.session_cost,
            "avg_tokens_per_conversation": self.session_tokens["total"] / max(self.conversation_count, 1)
        }

# Initialize token tracker
token_tracker = TokenTracker() if LangChainConfig.ENABLE_TOKEN_TRACKING else None

# Initialize LangChain components with configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def initialize_model():
    """Initialize ChatOpenAI model with retry logic"""
    return ChatOpenAI(**LangChainConfig.get_model_kwargs())

try:
    tool_node = ToolNode(tools)
    model = initialize_model()
    model_with_tools = model.bind_tools(tools)
    logger.info("LangChain model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangChain model: {e}")
    raise RuntimeError("LangChain model initialization failed")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

def call_model(state: AgentState):
    return {"messages": [model_with_tools.invoke(state['messages'])]}

def should_continue(state: AgentState):
    return "call_tools" if state['messages'][-1].tool_calls else "end"

workflow = StateGraph(AgentState)
workflow.add_node("call_model", call_model)
workflow.add_node("call_tools", tool_node)
workflow.set_entry_point("call_model")

workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {"call_tools": "call_tools", "end": END}
)
workflow.add_edge("call_tools", "call_model")

# Compile with memory checkpointer for conversation history
langgraph_app = workflow.compile(checkpointer=memory)

SYSTEM_PROMPT = """Hi there! I'm Alex, your dedicated HR Assistant at The Rise Tech Village. Think of me as your knowledgeable colleague who's always here to help with attendance matters, employee insights, and workplace analytics.

## Who I Am & How I Think 🤗
I'm designed to be genuinely helpful and conversational - not robotic. I understand context, remember our conversation, and can pick up on subtle cues in how you communicate. Whether you're a manager checking team performance or an employee updating your status, I adapt my responses to be most helpful for your specific situation.

## What I'm Great At 🌟

**👥 Complete Employee Management System:**
- Find employees naturally - "show me AI department employees", "give me Simon details", "all leaders"
- Department insights - "marketing team", "who works in ramstudios", "AI department employee details"
- Role-based queries - "show me all HR staff", "leaders in AI department"
- Individual profiles - "Simon details", "give me John's information" with complete contact info
- Smart human-loop confirmation - when you ask about a specific person, I first confirm I recognize them, then ask if you want their full details
- Enhanced responses: summaries for large groups, detailed info for individuals after confirmation, names first for departments

**✏️ Employee Creation & Updates:**
- Create employees naturally - "add new employee John Doe to marketing", "create Sarah as AI leader"
- Update any information - "change Mike's role to leader", "update Alice's phone number"
- Smart validation - validates emails, prevents duplicates, checks department existence
- Human-loop guidance - "John not found, create new record?", shows forms when needed
- Complete change tracking - logs all modifications with timestamps and reasons

**🚨 CRITICAL: Employee Management Tool Usage - IMMEDIATE FORM TRIGGERING**
MANDATORY: ALWAYS use these specific tools for employee operations WITHOUT ANY PRELIMINARY QUESTIONS:
- For "add employee", "create employee", "new employee", "add new user", "create user" → IMMEDIATELY call create_employee_intelligent() tool first
- For "update employee", "change employee info", "edit employee", "modify employee", "update user", "change user", "edit user", "modify user" → IMMEDIATELY call update_employee_intelligent() tool first  
- For "find employee", "show employee", "give me [NAME] details", "employee details", "employee information", "[NAME] information", "tell me about [NAME]" → IMMEDIATELY call query_employee_details_intelligent() tool first
- ABSOLUTELY NEVER ask users manually for employee details - the tools automatically trigger frontend forms when needed
- DO NOT provide text responses like "I can help you add employees" or "What would you like to update?" - ALWAYS trigger the appropriate tool immediately
- CRITICAL: DISTINGUISH INTENTS CAREFULLY:
  * "give me [NAME] details", "show me [NAME]", "tell me about [NAME]" → LOOKUP (use query_employee_details_intelligent)
  * "update [NAME] details", "change [NAME]", "modify [NAME]", "edit [NAME]" → UPDATE (use update_employee_intelligent)

**🔄 Human-Loop Employee Management Workflow:**
When a user asks to update an employee that doesn't exist:
1. First, search for the employee using update_employee_intelligent or search_employee_for_management
2. If not found, offer to create a new record: "Employee not found. Would you like to create them?"
3. If user confirms (says "yes", "create", "add them", etc.), immediately call create_employee_intelligent with the suggested name
4. When calling create_employee_intelligent or update_employee_intelligent, if missing required information, the tool will automatically return a form_request response
5. Always ensure that form requests include proper form_data with pre_filled values and available options

**🎯 Form Request Guidelines:**
- For CREATE operations: form should be mostly blank except for any extracted information from the user query
- For UPDATE operations: form should be pre-filled with current employee data  
- Always include departments and roles data in form_data
- Response type should be 'form_request' for new employees or 'employee_found_update_form' for existing employees

**📋 Intelligent Task Management & Query System:**
- Smart task queries - "today's tasks", "urgent tasks", "tasks led by John", "carpentry work"
- Time-based filtering - "yesterday's tasks", "this week", "last month", "tasks from last quarter"
- Priority-based searches - "high priority tasks", "urgent work", "low priority items"
- People & skill-based queries - "Sarah's tasks", "electrical work", "construction tasks"
- Location-aware searches - "office tasks", "remote work", "downtown site tasks"
- Comprehensive task details - shows leaders, departments, labourers, skills, and timelines
- Smart response patterns: Individual task details, small group summaries, large group statistics
- Human-loop guidance - suggests refinements when queries are ambiguous or return too many results

**🔍 CRITICAL: Task Query Tool Usage - IMMEDIATE INTELLIGENT SEARCH:**
MANDATORY: For ANY task-related queries, IMMEDIATELY call query_tasks_intelligent() tool:
- "show me today's tasks" → IMMEDIATELY call query_tasks_intelligent("show me today's tasks")  
- "urgent tasks" → IMMEDIATELY call query_tasks_intelligent("urgent tasks")
- "tasks led by John" → IMMEDIATELY call query_tasks_intelligent("tasks led by John")
- "carpentry work this week" → IMMEDIATELY call query_tasks_intelligent("carpentry work this week")
- "all tasks" → IMMEDIATELY call query_tasks_intelligent("all tasks")
- NEVER ask users for clarification first - the tool handles natural language and provides follow-up questions
- The tool automatically suggests refinements and alternative queries when needed
- Always trust the tool's response structure and provide the human-readable summary it generates

**🔧 Intelligent Labour Management & Query System:**
- Smart labour queries - "show me all carpenters", "available electricians", "busy workers"
- Workload analysis - "overloaded staff", "available labourers", "light workload workers"
- Skill-based searches - "electrical workers", "construction labourers", "maintenance staff"
- Task assignment queries - "who worked on urgent tasks", "labourers on downtown projects"
- Individual profiles - "John the carpenter details", "Sarah's current workload"
- Comprehensive labour analytics - skill distribution, workload balance, task assignment patterns
- Smart response patterns: Individual profiles, skill groups, workload analysis, statistical summaries
- Human-loop recommendations - workload balancing, skill optimization, task redistribution

**🔨 CRITICAL: Labour Query Tool Usage - IMMEDIATE INTELLIGENT SEARCH:**
MANDATORY: For ANY labour/worker-related queries, IMMEDIATELY call query_labourers_intelligent() tool:
- "show me all carpenters" → IMMEDIATELY call query_labourers_intelligent("show me all carpenters")
- "available workers" → IMMEDIATELY call query_labourers_intelligent("available workers")
- "busy labourers" → IMMEDIATELY call query_labourers_intelligent("busy labourers")
- "John the carpenter" → IMMEDIATELY call query_labourers_intelligent("John the carpenter")
- "workload analysis" → IMMEDIATELY call query_labourers_intelligent("workload analysis")
- "electricians working on urgent tasks" → IMMEDIATELY call query_labourers_intelligent("electricians working on urgent tasks")
- NEVER ask users for clarification first - the tool handles natural language and provides intelligent recommendations
- The tool automatically provides workload analysis and skill-based insights
- Always trust the tool's response structure and provide actionable recommendations it generates

**📋 Attendance Management:**
- Query attendance records and generate reports - "show me today's attendance", "who was absent yesterday"
- Analyze attendance patterns and trends across departments and time periods
- Generate comprehensive attendance summaries with insights and recommendations

**🚨 IMPORTANT: Attendance Creation/Updates Not Available**
- Creating new attendance records is not yet implemented
- Updating existing attendance records is not yet implemented  
- When users ask to create or update attendance, respond: "Attendance creation and updates are not yet available. I can help you query existing attendance records and generate reports instead."
- Available attendance functions: query_attendance_intelligent, summarize_attendance_intelligent

**📊 Intelligent Analytics & Insights:**
- Generate comprehensive attendance reports with real insights, not just numbers
- Identify patterns and trends - "I notice attendance drops on Fridays. Should we investigate?"
- Compare departments, roles, and time periods with actionable recommendations
- Provide predictive insights - spot potential issues before they become problems
- **Two-step summaries**: First show high-level overview, then ask before diving into detailed data
- **Smart follow-ups**: Automatically suggest relevant next questions based on what I find

**🧠 Human-like Understanding:**
- I understand context: "Update my status" (I'll help identify you)
- Handle natural dates: "last Monday," "two weeks ago," "yesterday"
- Recognize relationships: "How's my team doing?" (I'll analyze your department)
- Ask clarifying questions when needed, just like a human colleague would

## How I Communicate 💬

**Conversational & Contextual:**
- I remember what we've discussed in our conversation
- I ask follow-up questions to better understand your needs
- I provide explanations that make sense for your role and situation
- I suggest next steps based on what you're trying to accomplish

**Professional Yet Friendly:**
- Direct and helpful without being overly formal
- I acknowledge when situations are sensitive or require careful handling
- I celebrate good news (great attendance rates!) and address concerns thoughtfully
- I provide recommendations as suggestions, not orders

**Actionable Intelligence:**
- Every report includes "So what?" - what the data means and what you should do
- I highlight both successes worth celebrating and areas needing attention
- I provide specific, practical recommendations tailored to your situation
- I help you understand the story behind the numbers

## Valid Attendance Options 📝
- **Present** - Standard office/on-site work
- **Work from home** - Remote work arrangements
- **Work from out of Rise** - Working from external locations
- **Planned Leave** - Pre-approved time off
- **Sudden Leave** - Unexpected absence
- **Medical Leave** - Health-related absence
- **Holiday Leave** - Scheduled company holidays
- **Lieu leave** - Compensatory time off

## My Operating Principles 🎯

**Human-First Approach:**
I consider the human impact of attendance data. Behind every statistic is a person with circumstances, challenges, and contributions. I help you understand patterns while maintaining empathy and professionalism.

**Context is King:**
I don't just process requests - I understand intent. If you ask "How's attendance?" I'll consider your role, recent events, and what kind of insights would be most valuable to you.

**Continuous Learning:**
Throughout our conversation, I pick up on what matters most to you and adjust my responses accordingly. If you're focused on departmental performance, I'll emphasize those insights.

**Proactive Partnership:**
I don't just wait for specific requests. I notice patterns, suggest analyses you might find useful, and help you think about attendance data strategically.

**Interactive Human Loop:**
- When you ask for attendance data, I first provide a summary overview
- I then ask if you want detailed breakdowns before overwhelming you with data
- Based on what I find, I automatically generate relevant follow-up questions
- I adapt my responses based on your role (HR, leader, employee) and what you're trying to accomplish
- I suggest the most logical next steps to help you take action on insights

## Enhanced Employee & Attendance Capabilities 🚀

**Smart Employee Queries & Management:**
- Natural language understanding: "AI team employees", "show me all leaders", "Simon details"
- Department-wise filtering with role breakdowns
- Individual employee profiles with complete contact information and recent activity
- Human-in-the-loop responses: summaries first for large groups, detailed info for specific requests
- Context-aware suggestions: "Need details for any specific person?", "Want to see attendance records?"

**Intelligent Employee Creation:**
- Parse natural language requests: "create John as marketing leader", "add new AI intern"
- Smart form generation when information is missing
- Comprehensive validation: email format, duplicate checking, department verification
- Role validation: ensures only valid roles (hr, leader, employee) are assigned
- Pre-fills forms with extracted information from natural language

**Advanced Employee Updates:**
- Natural language update requests: "change Sarah's phone", "move Mike to HR department"
- Employee not found → automatic offer to create new record
- Multiple match disambiguation with clear employee identification
- Change tracking with before/after values and update reasons
- Form-based updates when specific changes aren't specified in natural language

**Role-Based Queries:**
- Query attendance by specific roles: leaders, HR staff, employees
- Special handling for task group leaders and their team responsibilities  
- Compare performance across different employee roles
- Cross-reference employee roles with department structures

**Smart Date Filtering:**
- Support for "today", "this week", "this month", "last month", custom date ranges
- Automatic period comparisons and trend analysis
- Flexible date queries like "show me who was present yesterday"

**Contextual Follow-ups:**
When I analyze data, I automatically suggest relevant questions like:
- "Would you like to see which employees have perfect attendance?"
- "Should I compare this with previous months?"
- "Need individual employee details from any department?"
- "Want me to identify employees needing attention?"

---

---

## 🔥 IMPORTANT TOOL USAGE RULES 🔥

**🎯 EMPLOYEE MANAGEMENT TOOL PRIORITY - ZERO DELAY EXECUTION:**
When user mentions ANY of these keywords or variations (even as part of longer sentences):
- CREATE TRIGGERS: "add employee", "create employee", "new employee", "add new employee", "need to add employee", "add user", "create user", "new user", "add new user"
- UPDATE TRIGGERS: "update employee", "change employee", "modify employee", "edit employee", "update user", "change user", "edit user", "update [NAME]", "change [NAME]", "edit [NAME]", "modify [NAME]"
- LOOKUP TRIGGERS: "find employee", "show employee", "give me [NAME] details", "get employee details", "employee information", "find user", "show user", "[NAME] information", "tell me about [NAME]"

MANDATORY IMMEDIATE ACTION (NO EXCEPTIONS - NOT EVEN FOR CLARIFICATION):
1. CALL THE TOOL IMMEDIATELY WITHOUT ANY PRELIMINARY TEXT OR EXPLANATIONS
2. create_employee_intelligent() for ANY creation-related requests
3. update_employee_intelligent() for ANY update-related requests (even with employee names)
4. query_employee_details_intelligent() for lookup-only requests
5. Let the tool handle form requests and responses automatically
6. NEVER write "I found [employee name]..." before calling the tool
7. NEVER provide ANY text content before tool calls - the tool provides all necessary messaging

**Enhanced Examples:**
User: "I need to add new employee" → INSTANTLY call create_employee_intelligent("I need to add new employee") WITHOUT ANY TEXT
User: "add new user John Smith" → INSTANTLY call create_employee_intelligent("add new user John Smith") WITHOUT ANY TEXT
User: "update employee details" → INSTANTLY call update_employee_intelligent("update employee details") WITHOUT ANY TEXT
User: "update John Smith details" → INSTANTLY call update_employee_intelligent("update John Smith details") WITHOUT ANY TEXT
User: "change Sarah's information" → INSTANTLY call update_employee_intelligent("change Sarah's information") WITHOUT ANY TEXT
User: "edit Mike Johnson" → INSTANTLY call update_employee_intelligent("edit Mike Johnson") WITHOUT ANY TEXT
User: "can you help me add employees?" → INSTANTLY call create_employee_intelligent("can you help me add employees?") WITHOUT ANY TEXT

**🔍 LOOKUP vs UPDATE Examples:**
User: "give me deshan details" → INSTANTLY call query_employee_details_intelligent("give me deshan details") WITHOUT ANY TEXT
User: "show me Simon information" → INSTANTLY call query_employee_details_intelligent("show me Simon information") WITHOUT ANY TEXT
User: "tell me about John Smith" → INSTANTLY call query_employee_details_intelligent("tell me about John Smith") WITHOUT ANY TEXT
User: "find employee Sarah" → INSTANTLY call query_employee_details_intelligent("find employee Sarah") WITHOUT ANY TEXT
User: "Yehara" → INSTANTLY call query_employee_details_intelligent("Yehara") WITHOUT ANY TEXT
vs.
User: "update deshan details" → INSTANTLY call update_employee_intelligent("update deshan details") WITHOUT ANY TEXT
User: "change Simon information" → INSTANTLY call update_employee_intelligent("change Simon information") WITHOUT ANY TEXT

❌ NEVER DO: "I found [employee]. What would you like to update?" 
❌ NEVER DO: "Here are [employee] details. Let me know what to change."
❌ NEVER DO: "I found Yehara who works in..." before calling the tool
✅ ALWAYS DO: Trigger the appropriate tool immediately WITHOUT any preliminary text - the tool provides all messaging

---

Ready to dive in? Whether you need to query attendance records, analyze trends, manage employees, or just understand what's happening with your team, I'm here to help in whatever way makes most sense for your situation. What would you like to explore?
"""

# ==============================================================================
# LANGCHAIN-POWERED UTILITY FUNCTIONS
# ==============================================================================


# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "HR Management System API", 
        "version": "2.0.0",
        "ai_agent_available": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with token usage summary"""
    health_data = {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "langsmith_enabled": LangChainConfig.LANGSMITH_TRACING,
        "token_tracking_enabled": LangChainConfig.ENABLE_TOKEN_TRACKING
    }
    
    if token_tracker:
        health_data["session_usage"] = token_tracker.get_session_summary()
    
    return health_data

@app.get("/usage")
async def get_usage_statistics():
    """Get detailed token usage and cost statistics"""
    if not token_tracker:
        return {"error": "Token tracking is disabled"}
    
    summary = token_tracker.get_session_summary()
    
    return {
        "token_usage": summary,
        "configuration": {
            "model": LangChainConfig.OPENAI_MODEL,
            "max_monthly_tokens": LangChainConfig.MAX_MONTHLY_TOKENS,
            "alert_threshold": f"{LangChainConfig.ALERT_THRESHOLD * 100:.0f}%",
            "cost_tracking_enabled": LangChainConfig.COST_TRACKING_ENABLED
        },
        "pricing_info": {
            "current_model": LangChainConfig.OPENAI_MODEL,
            "pricing_per_1k_tokens": token_tracker.pricing.get(LangChainConfig.OPENAI_MODEL, "Unknown")
        }
    }

@app.get("/employees", response_model=List[EmployeeWithDepartment])
async def get_employees(
    name: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(True)
):
    """Get all employees with optional filtering"""
    try:
        where_conditions = []
        params = []
        
        if name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{name}%")
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        if role:
            where_conditions.append("e.role = %s")
            params.append(role)
        
        if is_active is not None:
            where_conditions.append("e.is_active = %s")
            params.append(is_active)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT e.id, e.name, e.email, e.role, e.department_id, 
                   e.phone_number, e.address, e.is_active, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY e.name
        """
        
        employees = execute_query(query, params)
        return [dict(emp) for emp in employees]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/departments", response_model=List[Department])
async def get_departments():
    """Get all departments"""
    try:
        departments = execute_query("SELECT * FROM departments ORDER BY name")
        return [dict(dept) for dept in departments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/tasks", response_model=List[dict])
async def get_tasks(
    leader_name: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    location: Optional[str] = Query(None)
):
    """Get tasks with optional filtering"""
    try:
        where_conditions = []
        params = []
        
        if leader_name:
            where_conditions.append("e.name ILIKE %s AND e.role = 'leader'")
            params.append(f"%{leader_name}%")
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        if priority:
            where_conditions.append("t.priority = %s")
            params.append(priority)
        
        if location:
            where_conditions.append("t.location ILIKE %s")
            params.append(f"%{location}%")
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT DISTINCT t.id, t.task_title, t.location, t.expected_days, 
                   t.priority, t.notes, t.created_at, tg.group_name,
                   e.name as leader_name, d.name as department_name
            FROM tasks t
            LEFT JOIN task_groups tg ON t.task_group_id = tg.id
            LEFT JOIN task_group_leaders tgl ON tg.id = tgl.task_group_id
            LEFT JOIN employees e ON tgl.employee_id = e.id
            LEFT JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY t.created_at DESC
        """
        
        tasks = execute_query(query, params)
        return [dict(task) for task in tasks]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/intelligent", response_model=dict)
async def query_tasks_intelligent_endpoint(query_request: TaskIntelligentQuery):
    """Intelligent task query endpoint that accepts natural language queries"""
    try:
        # Call the intelligent task query tool
        result = query_tasks_intelligent(query_request.query)
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent task query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/intelligent", response_model=dict)
async def query_tasks_intelligent_get_endpoint(
    q: str = Query(..., description="Natural language query about tasks")
):
    """GET version of intelligent task query for simple queries"""
    try:
        # Call the intelligent task query tool
        result = query_tasks_intelligent(q)
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent task query GET endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/labourers/intelligent", response_model=dict)
async def query_labourers_intelligent_endpoint(query_request: LabourIntelligentQuery):
    """Intelligent labour query endpoint that accepts natural language queries"""
    try:
        # Call the intelligent labour query tool
        result = query_labourers_intelligent(query_request.query)
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent labour query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/labourers/intelligent", response_model=dict)
async def query_labourers_intelligent_get_endpoint(
    q: str = Query(..., description="Natural language query about labourers")
):
    """GET version of intelligent labour query for simple queries"""
    try:
        # Call the intelligent labour query tool
        result = query_labourers_intelligent(q)
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent labour query GET endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/labourers", response_model=List[dict])
async def get_labourers(
    skill: Optional[str] = Query(None, description="Filter by skill type"),
    available_only: Optional[bool] = Query(False, description="Show only available labourers"),
    name: Optional[str] = Query(None, description="Search by name")
):
    """Get labourers with optional filtering"""
    try:
        where_conditions = []
        params = []
        
        if skill:
            where_conditions.append("l.skill ILIKE %s")
            params.append(f"%{skill}%")
        
        if name:
            where_conditions.append("l.name ILIKE %s")
            params.append(f"%{name}%")
        
        # Base query with task count for availability filtering
        base_query = """
            SELECT DISTINCT 
                l.id, l.name, l.skill, l.created_at, l.updated_at,
                COUNT(DISTINCT tl.task_id) as current_task_count,
                STRING_AGG(DISTINCT t.task_title, ', ') as current_tasks,
                CASE 
                    WHEN COUNT(DISTINCT tl.task_id) = 0 THEN 'available'
                    WHEN COUNT(DISTINCT tl.task_id) <= 2 THEN 'light_load'
                    WHEN COUNT(DISTINCT tl.task_id) <= 4 THEN 'moderate_load'
                    WHEN COUNT(DISTINCT tl.task_id) <= 6 THEN 'heavy_load'
                    ELSE 'overloaded'
                END as workload_status
            FROM labours l
            LEFT JOIN task_labours tl ON l.id = tl.labour_id
            LEFT JOIN tasks t ON tl.task_id = t.id
        """
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += " GROUP BY l.id, l.name, l.skill, l.created_at, l.updated_at ORDER BY l.name ASC"
        
        labourers = execute_query(base_query, tuple(params))
        
        # Apply availability filter after query execution if needed
        if available_only:
            labourers = [l for l in labourers if l['workload_status'] == 'available']
        
        return [dict(labourer) for labourer in labourers]
        
    except Exception as e:
        logger.error(f"Error in get labourers endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Chat with HR AI Agent with conversation memory - Async optimized"""
    try:
        async def generate_response():
            # Create a unique thread ID for conversation memory
            # In production, this should be user-specific or session-specific
            thread_id = "hr_chat_session_1"
            config = RunnableConfig(configurable={"thread_id": thread_id})
            
            # Check if user is responding to a human loop question
            conversation_state = _get_conversation_state(thread_id)
            if conversation_state and (_detect_confirmation_intent(message.message) or _detect_rejection_intent(message.message)):
                # Handle human loop confirmation directly
                result = _handle_confirmation_followup(message.message, conversation_state)
                _clear_conversation_state(thread_id)
                
                # Stream the result
                response_obj = {"type": "text", "content": result.get('message', '')}
                yield f"data: {json.dumps(response_obj)}\n\n"
                
                # Check if this triggers a form
                if result.get('response_type') in ['form_request', 'employee_found_update_form']:
                    form_response = {
                        "type": "form_request",
                        "action": result.get('action', 'create_employee'),
                        "form_data": result.get('form_data', {}),
                        "content": ""
                    }
                    yield f"data: {json.dumps(form_response)}\n\n"
                
                yield f"data: [DONE]\n\n"
                return
            
            # Check if this is the first message in the conversation
            # If so, include the system prompt
            try:
                # Try to get existing conversation history
                state = langgraph_app.get_state(config)
                if not state.values.get("messages"):
                    # First message - include system prompt
                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=message.message)
                    ]
                else:
                    # Continuing conversation - just add the new message
                    messages = [HumanMessage(content=message.message)]
            except:
                # Fallback - treat as first message
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=message.message)
                ]
            
            final_response = []
            
            # Check if this is an employee query that should suppress AI text to prevent duplication
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
            
            # Temporarily disable suppression to debug blank response issue
            suppress_ai_text = False  # is_employee_query(message.message)
            logger.info(f"DEBUG: Query '{message.message}' - suppress_ai_text: {suppress_ai_text} (DISABLED FOR DEBUG)")
            
            # Async stream the response with memory
            async for chunk in langgraph_app.astream({"messages": messages}, config):
                logger.info(f"DEBUG: Processing chunk: {list(chunk.keys())}")
                
                if "call_model" in chunk:
                    ai_message = chunk["call_model"]["messages"][-1]
                    if hasattr(ai_message, 'content') and ai_message.content:
                        logger.info(f"DEBUG: AI message content: '{ai_message.content[:100]}...'")
                        if suppress_ai_text:
                            logger.info(f"DEBUG: Suppressing AI text due to employee query detection")
                        else:
                            logger.info(f"DEBUG: Streaming AI text")
                            response_obj = {"type": "text", "content": ai_message.content}
                            yield f"data: {json.dumps(response_obj)}\n\n"
                            final_response.append(response_obj)
                
                elif "call_tools" in chunk:
                    logger.info(f"DEBUG: Found tool calls chunk")
                    # Process tool calls to detect form requests
                    tool_messages = chunk["call_tools"]["messages"]
                    for tool_msg in tool_messages:
                        if hasattr(tool_msg, 'content'):
                            logger.info(f"DEBUG: Tool message content: {str(tool_msg.content)[:200]}...")
                            try:
                                # Try to parse tool response as JSON
                                if isinstance(tool_msg.content, str):
                                    tool_result = json.loads(tool_msg.content)
                                else:
                                    tool_result = tool_msg.content
                                
                                logger.info(f"DEBUG: Tool result response_type: {tool_result.get('response_type')}")
                                
                                # Check if tool result indicates a form request or human loop question
                                response_type = tool_result.get('response_type')
                                if response_type in ['form_request', 'employee_found_update_form', 'human_loop_question']:
                                    
                                    logger.info(f"DEBUG: Creating {response_type} response")
                                    
                                    # Map tool response to frontend form request format
                                    if response_type == 'form_request':
                                        form_response = {
                                            "type": "form_request",
                                            "action": tool_result.get('action', 'create_employee'),
                                            "form_data": tool_result.get('form_data', {}),
                                            "content": ""  # Empty content - only show form popup
                                        }
                                    elif response_type == 'employee_found_update_form':
                                        form_response = {
                                            "type": "form_request",
                                            "action": "update_employee",
                                            "form_data": tool_result.get('form_data', {}),
                                            "content": ""  # Empty content - only show form popup
                                        }
                                    elif response_type == 'human_loop_question':
                                        # Store conversation state and send human loop question
                                        conversation_state = tool_result.get('conversation_state', {})
                                        _store_conversation_state(thread_id, conversation_state)
                                        
                                        form_response = {
                                            "type": "human_loop_question", 
                                            "content": tool_result.get('message', ''),
                                            "follow_up_questions": tool_result.get('follow_up_questions', []),
                                            "conversation_state": conversation_state
                                        }
                                    
                                    # Stream the response to frontend
                                    yield f"data: {json.dumps(form_response)}\n\n"
                                    final_response.append(form_response)
                                    
                                    # Stop processing after form requests, but continue for human loop questions
                                    if response_type in ['form_request', 'employee_found_update_form']:
                                        yield f"data: [DONE]\n\n"
                                        return
                                    
                            except (json.JSONDecodeError, AttributeError):
                                # If tool content is not JSON or has no expected format, continue
                                pass
            
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(generate_response(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Critical chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

# Separate API functions for direct form submissions (not LangChain tools)
def create_employee_api(
    name: str,
    email: str, 
    role: str,
    department: str,
    phone_number: str = "",
    address: str = ""
) -> Dict[str, Any]:
    """
    Direct API function for employee creation from forms.
    This function is called directly by API endpoints, not through LangChain tools.
    """
    try:
        # Validate required fields
        if not name or not email or not role or not department:
            missing_fields = []
            if not name: missing_fields.append("name")
            if not email: missing_fields.append("email")
            if not role: missing_fields.append("role") 
            if not department: missing_fields.append("department")
            
            return {
                "success": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "response_type": "validation_error"
            }
        
        # Validate provided data
        validation_result = _validate_employee_data(name, email, role, department, "create")
        if not validation_result["valid"]:
            return {
                "success": False,
                "message": validation_result["message"],
                "response_type": "validation_error"
            }
        
        # Check for duplicate email
        existing_employee = execute_query(
            "SELECT id, name FROM employees WHERE LOWER(email) = LOWER(%s)",
            (email,)
        )
        
        if existing_employee:
            return {
                "success": False,
                "message": f"Email '{email}' is already registered to {existing_employee[0]['name']}. Each employee must have a unique email address.",
                "response_type": "validation_error"
            }
        
        # Get department ID
        dept_id = _get_department_id(department)
        if not dept_id:
            return {
                "success": False,
                "message": f"Department '{department}' not found.",
                "response_type": "validation_error"
            }
        
        # Create the employee
        execute_query(
            """
            INSERT INTO employees (name, email, role, department_id, phone_number, address, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (name, email, role, dept_id, phone_number or None, address or None, True, datetime.now(), datetime.now()),
            fetch=False
        )
        
        # Retrieve the created employee with department info
        created_employee = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE e.email = %s
            """,
            (email,)
        )[0]
        
        return {
            "success": True,
            "message": f"Successfully created employee record for {name}!",
            "response_type": "success_created",
            "data": created_employee
        }
        
    except Exception as e:
        logger.error(f"Error in create_employee_api: {e}")
        return {
            "success": False,
            "message": f"Error creating employee: {str(e)}",
            "response_type": "error"
        }

def update_employee_api(
    employee_identifier: str,
    field_updates: Dict[str, Any],
    update_reason: str = ""
) -> Dict[str, Any]:
    """
    Direct API function for employee updates from forms.
    This function is called directly by API endpoints, not through LangChain tools.
    """
    try:
        # Find the employee
        employee = _find_employee_by_identifier(employee_identifier)
        if not employee:
            return {
                "success": False,
                "message": f"Employee '{employee_identifier}' not found.",
                "response_type": "employee_not_found"
            }
        
        # Validate updates
        changes_made = []
        update_queries = []
        params = []
        
        for field, new_value in field_updates.items():
            if field in ['name', 'email', 'role', 'department', 'phone_number', 'address']:
                old_value = employee.get(field, '')
                
                # Handle department name to ID conversion
                if field == 'department':
                    # Compare with actual department name from database
                    current_dept_name = employee.get('department_name', employee.get('department', ''))
                    if current_dept_name != new_value:
                        dept_id = _get_department_id(new_value)
                        if not dept_id:
                            return {
                                "success": False,
                                "message": f"Department '{new_value}' not found.",
                                "response_type": "validation_error"
                            }
                        update_queries.append("department_id = %s")
                        params.append(dept_id)
                        changes_made.append(f"Department: '{current_dept_name}' → '{new_value}'")
                elif field == 'email':
                    # Check for duplicate email (excluding current employee)
                    if old_value.lower() != new_value.lower():
                        existing = execute_query(
                            "SELECT id, name FROM employees WHERE LOWER(email) = LOWER(%s) AND id != %s",
                            (new_value, employee['id'])
                        )
                        if existing:
                            return {
                                "success": False,
                                "message": f"Email '{new_value}' is already in use by {existing[0]['name']}.",
                                "response_type": "validation_error"
                            }
                        update_queries.append("email = %s")
                        params.append(new_value)
                        changes_made.append(f"Email: '{old_value}' → '{new_value}'")
                else:
                    # Handle other fields
                    if old_value != new_value:
                        update_queries.append(f"{field} = %s")
                        params.append(new_value)
                        field_display = field.replace('_', ' ').title()
                        changes_made.append(f"{field_display}: '{old_value}' → '{new_value}'")
        
        if not changes_made:
            return {
                "success": True,
                "message": "No changes were made - all values are already current.",
                "response_type": "no_changes",
                "data": employee,
                "changes_made": []
            }
        
        # Update the employee
        if update_queries:
            update_queries.append("updated_at = %s")
            params.append(datetime.now())
            params.append(employee['id'])
            
            execute_query(
                f"UPDATE employees SET {', '.join(update_queries)} WHERE id = %s",
                params,
                fetch=False
            )
        
        # Get updated employee data
        updated_employee = execute_query(
            """
            SELECT e.*, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE e.id = %s
            """,
            (employee['id'],)
        )[0]
        
        return {
            "success": True,
            "message": f"Successfully updated {employee['name']}'s information!",
            "response_type": "success_updated",
            "data": updated_employee,
            "changes_made": changes_made
        }
        
    except Exception as e:
        logger.error(f"Error in update_employee_api: {e}")
        return {
            "success": False,
            "message": f"Error updating employee: {str(e)}",
            "response_type": "error"
        }

@app.post("/tools/create_employee_intelligent")
async def create_employee_endpoint(request: EmployeeCreateRequest):
    """Direct API endpoint for employee creation from forms"""
    try:
        # Call the new API-specific function instead of the LangChain tool
        result = create_employee_api(
            name=request.name,
            email=request.email,
            role=request.role,
            department=request.department,
            phone_number=request.phone_number,
            address=request.address
        )
        
        # Return the result with consistent format
        if result.get("success", False):
            return {
                "success": True,
                "message": f"Employee {request.name} created successfully!",
                "data": result.get("data", {}),
                "employee": result.get("data", {})
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Failed to create employee"),
                "error": result.get("message", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"Employee creation error: {e}")
        return {
            "success": False,
            "message": f"Error creating employee: {str(e)}",
            "error": str(e)
        }

@app.post("/tools/update_employee_intelligent")
async def update_employee_endpoint(request: EmployeeUpdateRequest):
    """Direct API endpoint for employee updates from forms"""
    try:
        # Call the new API-specific function instead of the LangChain tool
        result = update_employee_api(
            employee_identifier=request.employee_identifier,
            field_updates=request.field_updates,
            update_reason=request.update_reason
        )
        
        # Return the result with consistent format
        if result.get("success", False):
            return {
                "success": True,
                "message": f"Employee updated successfully!",
                "data": result.get("data", {}),
                "changes_made": result.get("changes_made", [])
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Failed to update employee"),
                "error": result.get("message", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"Employee update error: {e}")
        return {
            "success": False,
            "message": f"Error updating employee: {str(e)}",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)