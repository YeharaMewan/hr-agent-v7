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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig

from database import execute_query, get_db_connection, search_similar_documents
from models import (
    Employee, EmployeeWithDepartment, EmployeeCreate, EmployeeQuery,
    Attendance, AttendanceCreate, AttendanceMarkRequest, AttendanceQuery,
    Task, TaskCreate, TaskWithDetails, TaskQuery,
    Department, ChatMessage, APIResponse, AttendanceStatistics
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

@tool
def mark_attendance(employee_identifier: str, status: str = "Present", notes: str = "", attendance_date: str = "") -> Dict[str, Any]:
    """
    Marks attendance for an employee. You can use either employee ID (number) or employee name.
    Status options: Present, Work from home, Planned Leave, Sudden Leave, Medical Leave, Holiday Leave, Lieu leave, Work from out of Rise
    If attendance_date is not provided, uses today's date. Format: YYYY-MM-DD
    """
    try:
            # Validate status
        valid_statuses = ["Present", "Work from home", "Planned Leave", "Sudden Leave", 
                        "Medical Leave", "Holiday Leave", "Lieu leave", "Work from out of Rise"]
        if status not in valid_statuses:
            return {
                "success": False, 
                "message": f"Invalid status. Valid options: {', '.join(valid_statuses)}"
            }
        
            # Determine if identifier is ID or name
        employee = None
        if employee_identifier.isdigit():
                # Lookup by ID
            employee = execute_query(
                "SELECT id, name, email FROM employees WHERE id = %s AND is_active = true",
                (int(employee_identifier),)
            )
        else:
                # Lookup by name (fuzzy matching)
            employee = execute_query(
                "SELECT id, name, email FROM employees WHERE name ILIKE %s AND is_active = true",
                (f"%{employee_identifier}%",)
            )
        
        if not employee:
            return {"success": False, "message": f"Employee '{employee_identifier}' not found or inactive."}
        
        if len(employee) > 1:
                # Multiple matches found
            matches = [f"{emp['name']} (ID: {emp['id']})" for emp in employee]
            return {
                "success": False, 
                "message": f"Multiple employees found matching '{employee_identifier}'. Please be more specific. Found: {', '.join(matches)}"
            }
        
        employee_data = employee[0]
        employee_id = employee_data['id']
        employee_name = employee_data['name']
        
            # Parse date
        if attendance_date:
            try:
                target_date = datetime.strptime(attendance_date, '%Y-%m-%d').date()
            except ValueError:
                return {"success": False, "message": "Invalid date format. Please use YYYY-MM-DD format."}
        else:
            target_date = date.today()
        
            # Check if attendance already marked for the date
        existing = execute_query(
            "SELECT id, status FROM attendances WHERE employee_id = %s AND attendance_date = %s",
            (employee_id, target_date)
        )
        
        if existing:
            old_status = existing[0]['status']
                # Update existing record
            execute_query(
                "UPDATE attendances SET status = %s WHERE employee_id = %s AND attendance_date = %s",
                (status, employee_id, target_date),
                fetch=False
            )
            action = "updated"
            message = f"Attendance for {employee_name} successfully {action} from '{old_status}' to '{status}' for {target_date.strftime('%Y-%m-%d')}."
        else:
                # Create new record
            execute_query(
                "INSERT INTO attendances (employee_id, attendance_date, status) VALUES (%s, %s, %s)",
                (employee_id, target_date, status),
                fetch=False
            )
            action = "marked"
            message = f"Attendance for {employee_name} successfully {action} as '{status}' for {target_date.strftime('%Y-%m-%d')}."
        
        return {
            "success": True, 
            "message": message,
            "employee_name": employee_name,
            "employee_id": employee_id,
            "status": status,
            "date": target_date.strftime('%Y-%m-%d'),
            "action": action
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error marking attendance: {e}"}

@tool
def update_attendance_history(employee_identifier: str, attendance_date: str, new_status: str, reason: str = "") -> Dict[str, Any]:
    """
    Updates a specific past attendance record for an employee. Use this when you need to modify attendance for a specific date.
    employee_identifier can be either employee ID (number) or employee name.
    attendance_date format: YYYY-MM-DD
    Status options: Present, Work from home, Planned Leave, Sudden Leave, Medical Leave, Holiday Leave, Lieu leave, Work from out of Rise
    """
    try:
            # Validate status
        valid_statuses = ["Present", "Work from home", "Planned Leave", "Sudden Leave", 
                        "Medical Leave", "Holiday Leave", "Lieu leave", "Work from out of Rise"]
        if new_status not in valid_statuses:
            return {
                "success": False, 
                "message": f"Invalid status. Valid options: {', '.join(valid_statuses)}"
            }
        
            # Parse and validate date
        try:
            target_date = datetime.strptime(attendance_date, '%Y-%m-%d').date()
        except ValueError:
            return {"success": False, "message": "Invalid date format. Please use YYYY-MM-DD format."}
        
            # Determine if identifier is ID or name
        employee = None
        if employee_identifier.isdigit():
                # Lookup by ID
            employee = execute_query(
                "SELECT id, name, email FROM employees WHERE id = %s AND is_active = true",
                (int(employee_identifier),)
            )
        else:
                # Lookup by name (fuzzy matching)
            employee = execute_query(
                "SELECT id, name, email FROM employees WHERE name ILIKE %s AND is_active = true",
                (f"%{employee_identifier}%",)
            )
        
        if not employee:
            return {"success": False, "message": f"Employee '{employee_identifier}' not found or inactive."}
        
        if len(employee) > 1:
                # Multiple matches found
            matches = [f"{emp['name']} (ID: {emp['id']})" for emp in employee]
            return {
                "success": False, 
                "message": f"Multiple employees found matching '{employee_identifier}'. Please be more specific. Found: {', '.join(matches)}"
            }
        
        employee_data = employee[0]
        employee_id = employee_data['id']
        employee_name = employee_data['name']
        
            # Check if attendance record exists for the date
        existing = execute_query(
            "SELECT id, status FROM attendances WHERE employee_id = %s AND attendance_date = %s",
            (employee_id, target_date)
        )
        
        if not existing:
            return {
                "success": False, 
                "message": f"No attendance record found for {employee_name} on {target_date.strftime('%Y-%m-%d')}. Use mark_attendance to create a new record."
            }
        
        old_status = existing[0]['status']
        
            # Update the record
        execute_query(
            "UPDATE attendances SET status = %s WHERE employee_id = %s AND attendance_date = %s",
            (new_status, employee_id, target_date),
            fetch=False
        )
        
        message = f"Attendance for {employee_name} on {target_date.strftime('%Y-%m-%d')} successfully updated from '{old_status}' to '{new_status}'."
        if reason:
            message += f" Reason: {reason}"
        
        return {
            "success": True, 
            "message": message,
            "employee_name": employee_name,
            "employee_id": employee_id,
            "date": target_date.strftime('%Y-%m-%d'),
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error updating attendance history: {e}"}

@tool
def get_employee_attendance_summary(employee_identifier: str, days: int = 30) -> Dict[str, Any]:
    """
    Get a comprehensive attendance summary for a specific employee including statistics, patterns, and insights.
    employee_identifier can be either employee ID (number) or employee name.
    days parameter specifies the number of days to look back (default: 30 days).
    """
    try:
            # Determine if identifier is ID or name
        employee = None
        if employee_identifier.isdigit():
                # Lookup by ID
            employee = execute_query(
                "SELECT id, name, email, role FROM employees WHERE id = %s AND is_active = true",
                (int(employee_identifier),)
            )
        else:
                # Lookup by name (fuzzy matching)
            employee = execute_query(
                "SELECT id, name, email, role FROM employees WHERE name ILIKE %s AND is_active = true",
                (f"%{employee_identifier}%",)
            )
        
        if not employee:
            return {"success": False, "message": f"Employee '{employee_identifier}' not found or inactive."}
        
        if len(employee) > 1:
                # Multiple matches found
            matches = [f"{emp['name']} (ID: {emp['id']})" for emp in employee]
            return {
                "success": False, 
                "message": f"Multiple employees found matching '{employee_identifier}'. Please be more specific. Found: {', '.join(matches)}"
            }
        
        employee_data = employee[0]
        employee_id = employee_data['id']
        employee_name = employee_data['name']
        employee_role = employee_data['role']
        
            # Calculate date range
        start_date = date.today() - timedelta(days=days)
        end_date = date.today()
        
            # Get attendance records for the period
        attendance_records = execute_query("""
            SELECT attendance_date, status 
            FROM attendances 
            WHERE employee_id = %s AND attendance_date >= %s AND attendance_date <= %s
            ORDER BY attendance_date DESC
        """, (employee_id, start_date, end_date))
        
        if not attendance_records:
            return {
                "success": True,
                "message": f"No attendance records found for {employee_name} in the last {days} days.",
                "employee_name": employee_name,
                "summary": f"**📊 Attendance Summary for {employee_name}**\n\nNo attendance records found for the last {days} days."
            }
        
            # Calculate statistics
        total_records = len(attendance_records)
        status_counts = {}
        recent_pattern = []
        
        for record in attendance_records:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
                # Keep track of recent 7 days for pattern analysis
            if len(recent_pattern) < 7:
                recent_pattern.append({
                    'date': record['attendance_date'].strftime('%Y-%m-%d (%a)'),
                    'status': status
                })
        
            # Calculate percentages
        status_percentages = {
            status: round((count / total_records) * 100, 1)
            for status, count in status_counts.items()
        }
        
            # Generate insights
        present_count = status_counts.get('Present', 0)
        wfh_count = status_counts.get('Work from home', 0)
        leave_types = ['Planned Leave', 'Sudden Leave', 'Medical Leave', 'Holiday Leave', 'Lieu leave']
        total_leaves = sum(status_counts.get(leave_type, 0) for leave_type in leave_types)
        
        attendance_rate = round(((present_count + wfh_count) / total_records) * 100, 1) if total_records > 0 else 0
        
            # Build comprehensive summary
        summary_lines = [
            f"Attendance Summary for {employee_name} ({employee_role.title()})",
            "-" * 50,
            f"Period: Last {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            f"Total Records: {total_records}",
            f"Attendance Rate: {attendance_rate}% (Present + WFH)",
            "",
            "Status Breakdown:"
        ]
        
            # Add status breakdown
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = status_percentages[status]
            summary_lines.append(f"- {status}: {count} days ({percentage}%)")
        
        summary_lines.extend(["", "Recent Activity (Last 7 Days):"])
        
            # Add recent pattern
        for day in recent_pattern:
            summary_lines.append(f"  - {day['date']}: {day['status']}")
        
            # Add insights
        summary_lines.extend(["", "Insights:"])
        
        if attendance_rate >= 90:
            summary_lines.append("  - Excellent attendance record!")
        elif attendance_rate >= 75:
            summary_lines.append("  - Good attendance record.")
        else:
            summary_lines.append("  - Attendance below expectations.")
        
        if total_leaves > 0:
            summary_lines.append(f"  - Total leave days: {total_leaves}")
        
        if wfh_count > 0:
            wfh_percentage = round((wfh_count / total_records) * 100, 1)
            summary_lines.append(f"  - Work from home: {wfh_percentage}% of working days")
        
        return {
            "success": True,
            "message": f"Attendance summary generated for {employee_name}",
            "employee_name": employee_name,
            "employee_id": employee_id,
            "employee_role": employee_role,
            "period_days": days,
            "total_records": total_records,
            "attendance_rate": attendance_rate,
            "status_breakdown": status_counts,
            "status_percentages": status_percentages,
            "recent_pattern": recent_pattern,
            "summary": "\n".join(summary_lines)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error generating attendance summary: {e}"}

@tool
def get_attendance_insights(department: str = None, days: int = 30, analysis_type: str = "general") -> Dict[str, Any]:
    """
    Provides AI-powered insights and pattern analysis on attendance data.
    department: Filter by specific department (optional)
    days: Number of days to analyze (default: 30)
    """
    try:
            # Calculate date range
        start_date = date.today() - timedelta(days=days)
        end_date = date.today()
        
            # Build base query with optional department filter
        where_conditions = ["a.attendance_date >= %s AND a.attendance_date <= %s"]
        params = [start_date, end_date]
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
            # Get comprehensive attendance data
        query = f"""
            SELECT a.attendance_date, a.status, e.name as employee_name, 
                   e.role, d.name as department_name,
                   EXTRACT(dow FROM a.attendance_date) as day_of_week,
                   EXTRACT(week FROM a.attendance_date) as week_number
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC
        """
        
        records = execute_query(query, params)
        
        if not records:
            return {
                "success": False,
                "message": f"No attendance data found for the specified criteria in the last {days} days."
            }
        
            # Perform different types of analysis based on analysis_type
        insights = []
        statistics = {}
        
            # Basic statistics
        total_records = len(records)
        unique_employees = len(set(r['employee_name'] for r in records))
        departments_analyzed = list(set(r['department_name'] for r in records))
        
            # Status distribution
        status_counts = {}
        for record in records:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
            # Day-of-week analysis
        dow_analysis = {}
        dow_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                    4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        
        for record in records:
            dow = int(record['day_of_week'])
            dow_name = dow_names[dow]
            if dow_name not in dow_analysis:
                dow_analysis[dow_name] = {'total': 0, 'present': 0, 'wfh': 0, 'leave': 0}
            
            dow_analysis[dow_name]['total'] += 1
            if record['status'] == 'Present':
                dow_analysis[dow_name]['present'] += 1
            elif record['status'] == 'Work from home':
                dow_analysis[dow_name]['wfh'] += 1
            else:
                dow_analysis[dow_name]['leave'] += 1
        
            # Department-wise analysis
        dept_analysis = {}
        for record in records:
            dept = record['department_name']
            if dept not in dept_analysis:
                dept_analysis[dept] = {'total': 0, 'present': 0, 'wfh': 0, 'leave': 0}
            
            dept_analysis[dept]['total'] += 1
            if record['status'] == 'Present':
                dept_analysis[dept]['present'] += 1
            elif record['status'] == 'Work from home':
                dept_analysis[dept]['wfh'] += 1
            else:
                dept_analysis[dept]['leave'] += 1
        
            # Role-based analysis
        role_analysis = {}
        for record in records:
            role = record['role']
            if role not in role_analysis:
                role_analysis[role] = {'total': 0, 'present': 0, 'wfh': 0, 'leave': 0}
            
            role_analysis[role]['total'] += 1
            if record['status'] == 'Present':
                role_analysis[role]['present'] += 1
            elif record['status'] == 'Work from home':
                role_analysis[role]['wfh'] += 1
            else:
                role_analysis[role]['leave'] += 1
        
            # Generate insights based on analysis type
        if analysis_type == "general" or analysis_type == "trends":
            insights.extend([
                "📊 **General Attendance Insights:**",
                f"• Analyzed {total_records} attendance records from {unique_employees} employees",
                f"• Departments covered: {', '.join(departments_analyzed)}",
                ""
            ])
            
                # Overall attendance rate
            present_wfh = status_counts.get('Present', 0) + status_counts.get('Work from home', 0)
            overall_rate = round((present_wfh / total_records) * 100, 1)
            insights.append(f"• **Overall Attendance Rate:** {overall_rate}%")
            
                # Best and worst days
            best_dow = max(dow_analysis.keys(), key=lambda x: (dow_analysis[x]['present'] + dow_analysis[x]['wfh']) / dow_analysis[x]['total'])
            worst_dow = min(dow_analysis.keys(), key=lambda x: (dow_analysis[x]['present'] + dow_analysis[x]['wfh']) / dow_analysis[x]['total'])
            
            insights.extend([
                f"• **Best Attendance Day:** {best_dow}",
                f"• **Challenging Attendance Day:** {worst_dow}",
                ""
            ])
        
        if analysis_type == "general" or analysis_type == "departments":
            insights.append("🏢 **Department Performance:**")
            for dept, data in sorted(dept_analysis.items(), key=lambda x: x[1]['total'], reverse=True):
                rate = round(((data['present'] + data['wfh']) / data['total']) * 100, 1) if data['total'] > 0 else 0
                insights.append(f"• **{dept}:** {rate}% attendance rate ({data['total']} records)")
            insights.append("")
        
        if analysis_type == "general" or analysis_type == "roles":
            insights.append("👥 **Role-based Analysis:**")
            for role, data in sorted(role_analysis.items(), key=lambda x: x[1]['total'], reverse=True):
                rate = round(((data['present'] + data['wfh']) / data['total']) * 100, 1) if data['total'] > 0 else 0
                insights.append(f"• **{role.title()}s:** {rate}% attendance rate ({data['total']} records)")
            insights.append("")
        
        if analysis_type == "general" or analysis_type == "anomalies":
            insights.append("🔍 **Pattern Observations:**")
            
                # Work from home patterns
            wfh_percentage = round((status_counts.get('Work from home', 0) / total_records) * 100, 1)
            if wfh_percentage > 20:
                insights.append(f"• High WFH adoption: {wfh_percentage}% of all records")
            elif wfh_percentage > 10:
                insights.append(f"• Moderate WFH usage: {wfh_percentage}% of all records")
            else:
                insights.append(f"• Low WFH usage: {wfh_percentage}% of all records")
            
                # Leave patterns
            total_leaves = sum(status_counts.get(leave_type, 0) for leave_type in 
                             ['Planned Leave', 'Sudden Leave', 'Medical Leave', 'Holiday Leave', 'Lieu leave'])
            leave_percentage = round((total_leaves / total_records) * 100, 1)
            
            if leave_percentage > 15:
                insights.append(f"• High leave rate detected: {leave_percentage}% of records")
            elif leave_percentage < 5:
                insights.append(f"• Low leave rate: {leave_percentage}% - employees might need encouragement to take breaks")
            else:
                insights.append(f"• Normal leave pattern: {leave_percentage}% of records")
            
            insights.append("")
        
            # Build summary report
        summary_lines = [
            f"🔍 **Attendance Insights Analysis** ({analysis_type.title()})",
            "=" * 60,
            f"**Analysis Period:** Last {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            f"**Scope:** {department if department else 'All Departments'}",
            ""
        ] + insights
        
            # Add recommendations
        summary_lines.extend([
            "💡 **Recommendations:**",
            f"• Monitor departments with attendance rates below 85%",
            f"• Consider flexible work arrangements if WFH requests are high",
            f"• Review leave policies if leave rates are unusually high or low",
            f"• Focus on improving {worst_dow} attendance patterns",
            ""
        ])
        
        return {
            "success": True,
            "message": f"Attendance insights generated for {analysis_type} analysis",
            "analysis_type": analysis_type,
            "period_days": days,
            "total_records": total_records,
            "unique_employees": unique_employees,
            "departments": departments_analyzed,
            "overall_attendance_rate": overall_rate,
            "status_distribution": status_counts,
            "day_of_week_analysis": dow_analysis,
            "department_analysis": dept_analysis,
            "role_analysis": role_analysis,
            "insights_summary": "\n".join(summary_lines)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error generating attendance insights: {e}"}

@tool
def get_attendance_report(employee_name: str = None, department: str = None, 
                         start_date_str: str = None, end_date_str: str = None, 
                         status: str = None, days: int = 7, report_type: str = "detailed",
                         include_trends: bool = True) -> str:
    """
    Fetches a comprehensive attendance report with filtering and trend analysis.
    
    Parameters:
    - employee_name: Filter by employee name (supports partial matching)
    - department: Filter by department name (supports partial matching)  
    - start_date_str, end_date_str: Date range (YYYY-MM-DD format)
    - status: Filter by specific attendance status
    - days: Number of days to look back (default: 7)
    """
    try:
            # Build query conditions
        where_conditions = []
        params = []
        
            # Date range logic
        if start_date_str and end_date_str:
            where_conditions.append("a.attendance_date >= %s AND a.attendance_date <= %s")
            params.extend([start_date_str, end_date_str])
            period_start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            period_end = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        elif start_date_str:
            where_conditions.append("a.attendance_date >= %s")
            params.append(start_date_str)
            period_start = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            period_end = date.today()
        else:
                # Default to last N days
            period_start = date.today() - timedelta(days=days)
            period_end = date.today()
            where_conditions.append("a.attendance_date >= %s")
            params.append(period_start)
        
            # Employee filtering
        if employee_name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{employee_name}%")
        
            # Department filtering
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
            # Status filtering
        if status:
            where_conditions.append("a.status = %s")
            params.append(status)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
            # Enhanced query with additional data for trends
        query = f"""
            SELECT a.attendance_date, a.status, e.name as employee_name, 
                   d.name as department_name, e.role, e.id as employee_id,
                   EXTRACT(dow FROM a.attendance_date) as day_of_week,
                   EXTRACT(week FROM a.attendance_date) as week_number
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC, e.name
        """
        
        # Debug logging for query execution
        logger.info(f"Executing attendance query with {len(params)} parameters")
        logger.debug(f"Query: {query}")
        logger.debug(f"Parameters: {params}")
        
        try:
            records = execute_query(query, params)
            logger.info(f"Query returned {len(records) if records else 0} records")
            
            # Log sample data for debugging
            if records and len(records) > 0:
                sample_record = records[0]
                logger.debug(f"Sample record: {sample_record}")
                
                # Check for day_of_week values
                dow_values = [r.get('day_of_week') for r in records[:5]]
                logger.debug(f"First 5 day_of_week values: {dow_values}")
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return f"Database error occurred while fetching attendance data: {e}"
        
        if not records:
            logger.warning(f"No attendance records found for period {period_start} to {period_end}")
            return "No attendance records found for the specified criteria. This could indicate:\n" + \
                   "• No data exists for the specified date range\n" + \
                   "• Filters are too restrictive\n" + \
                   "• Database connectivity issues\n" + \
                   f"Use the debug endpoint: /debug/attendance?days={days} for more details."
        
        try:
            # Basic statistics with error handling
            total_records = len(records)
            unique_employees = len(set(rec.get('employee_name', 'Unknown') for rec in records if rec.get('employee_name')))
            unique_dates = len(set(rec.get('attendance_date') for rec in records if rec.get('attendance_date')))
            unique_departments = len(set(rec.get('department_name', 'Unknown') for rec in records if rec.get('department_name')))
            
            logger.info(f"Processing attendance data: {total_records} records, {unique_employees} employees, {unique_dates} dates")
            
        except Exception as e:
            logger.error(f"Error processing basic statistics: {e}")
            return f"Error processing attendance data: {e}. Please check the debug endpoint for more details."
        
            # Status distribution
        status_counts = {}
        for record in records:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
            # Calculate attendance rate with standardized status classification
        present_count = (status_counts.get('Present', 0) + 
                        status_counts.get('Work from home', 0) + 
                        status_counts.get('Work from out of Rise', 0))
        attendance_rate = round((present_count / total_records) * 100, 1) if total_records > 0 else 0
        
            # Build report based on type
        if report_type == "summary":
                # Summary Report
            report_lines = [
                "📊 **Attendance Summary Report**",
                "=" * 15,
                f"**Period:** {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                f"**Filters:** Employee: {employee_name or 'All'}, Department: {department or 'All'}, Status: {status or 'All'}",
                "",
                f"📈 **Key Metrics:**",
                f"  • Total Records: {total_records}",
                f"  • Employees Covered: {unique_employees}",
                f"  • Departments: {unique_departments}",
                f"  • Days Covered: {unique_dates}",
                f"  • Overall Attendance Rate: {attendance_rate}%",
                "",
                f"📊 **Status Breakdown:**"
            ]
            
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = round((count / total_records) * 100, 1)
                emoji = "✅" if status == "Present" else "🏠" if status == "Work from home" else "📅"
                report_lines.append(f"  {emoji} {status}: {count} ({percentage}%)")
            
        elif report_type == "trends":
                # Trends Report
            report_lines = [
                "📈 **Attendance Trends Report**",
                "=" * 45,
                f"**Analysis Period:** {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                ""
            ]
            
                # Day of week analysis with proper debugging
            dow_analysis = {}
            dow_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                        4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
            
            # Debug: Log total records being processed
            logger.info(f"Processing {len(records)} records for day-of-week analysis")
            
            for record in records:
                # Handle potential None values for day_of_week
                if record.get('day_of_week') is None:
                    logger.warning(f"Missing day_of_week for record: {record.get('attendance_date')}")
                    continue
                    
                try:
                    dow = int(record['day_of_week'])
                    dow_name = dow_names.get(dow, f'Unknown_DOW_{dow}')
                    
                    if dow_name not in dow_analysis:
                        dow_analysis[dow_name] = {'total': 0, 'present': 0}
                    
                    dow_analysis[dow_name]['total'] += 1
                    
                    # Standardized present status check (includes Work from home)
                    if record['status'] in ['Present', 'Work from home', 'Work from out of Rise']:
                        dow_analysis[dow_name]['present'] += 1
                        
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing day_of_week for record {record.get('attendance_date')}: {e}")
                    continue
            
            # Debug: Log day-of-week analysis results
            for day_name, data in dow_analysis.items():
                logger.info(f"{day_name}: {data['present']}/{data['total']} records")
            
            report_lines.append("📅 **Day-of-Week Trends:**")
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                if day in dow_analysis:
                    data = dow_analysis[day]
                    rate = round((data['present'] / data['total']) * 100, 1) if data['total'] > 0 else 0
                    report_lines.append(f"  • {day}: {rate}% attendance ({data['present']}/{data['total']} records)")
                else:
                    report_lines.append(f"  • {day}: No records found")
            
                # Department trends with standardized status classification
            if unique_departments > 1:
                dept_analysis = {}
                for record in records:
                    dept = record['department_name']
                    if dept not in dept_analysis:
                        dept_analysis[dept] = {'total': 0, 'present': 0}
                    
                    dept_analysis[dept]['total'] += 1
                    # Standardized present status check (consistent with above)
                    if record['status'] in ['Present', 'Work from home', 'Work from out of Rise']:
                        dept_analysis[dept]['present'] += 1
                
                report_lines.extend(["", "🏢 **Department Trends:**"])
                for dept, data in sorted(dept_analysis.items(), key=lambda x: x[1]['present']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True):
                    rate = round((data['present'] / data['total']) * 100, 1) if data['total'] > 0 else 0
                    report_lines.append(f"  • {dept}: {rate}% attendance ({data['present']}/{data['total']} records)")
            
        elif report_type == "comparative":
                # Comparative Report
            report_lines = [
                "⚖️ **Comparative Attendance Report**",
                "=" * 50,
                f"**Comparison Period:** {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                ""
            ]
            
                # Employee comparison with standardized status classification
            if unique_employees > 1:
                emp_analysis = {}
                for record in records:
                    emp = record['employee_name']
                    if emp not in emp_analysis:
                        emp_analysis[emp] = {'total': 0, 'present': 0, 'department': record['department_name'], 'role': record['role']}
                    
                    emp_analysis[emp]['total'] += 1
                    # Standardized present status check (consistent with above)
                    if record['status'] in ['Present', 'Work from home', 'Work from out of Rise']:
                        emp_analysis[emp]['present'] += 1
                
                report_lines.append("👥 **Employee Comparison (Top Performers):**")
                sorted_employees = sorted(emp_analysis.items(), key=lambda x: x[1]['present']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
                
                for emp_name, data in sorted_employees[:10]:  # Top 10
                    rate = round((data['present'] / data['total']) * 100, 1) if data['total'] > 0 else 0
                    report_lines.append(f"  • {emp_name} ({data['department']}): {rate}% ({data['present']}/{data['total']})")
            
        else:  # detailed report
                # Group by date for detailed view
            report_by_date = {}
            for record in records:
                date_str = record['attendance_date'].strftime('%Y-%m-%d (%A)')
                if date_str not in report_by_date:
                    report_by_date[date_str] = []
                
                entry = f"  • {record['employee_name']} ({record['department_name']}) [{record['role']}]: **{record['status']}**"
                report_by_date[date_str].append(entry)
            
                # Build detailed report
            report_lines = [
                "📊 **Detailed Attendance Report**",
                "=" * 50,
                f"**Period:** {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                f"**Summary:** {total_records} records • {unique_employees} employees • {attendance_rate}% attendance rate",
                ""
            ]
            
            for date_str, entries in sorted(report_by_date.items(), reverse=True):
                daily_present = len([e for e in entries if '**Present**' in e or '**Work from home**' in e])
                daily_total = len(entries)
                daily_rate = round((daily_present / daily_total) * 100, 1) if daily_total > 0 else 0
                
                report_lines.append(f"**📅 {date_str}** ({daily_rate}% attendance)")
                report_lines.extend(entries)
                report_lines.append("")
        
            # Add trend insights if requested
        if include_trends and report_type != "trends":
            report_lines.extend(["", "🔍 **Trend Insights:**"])
            
            if attendance_rate >= 90:
                report_lines.append("  • ⭐ Excellent overall attendance performance")
            elif attendance_rate >= 75:
                report_lines.append("  • ✅ Good attendance performance")
            else:
                report_lines.append("  • ⚠️ Attendance performance needs improvement")
            
                # WFH trends
            wfh_count = status_counts.get('Work from home', 0)
            if wfh_count > 0:
                wfh_percentage = round((wfh_count / total_records) * 100, 1)
                report_lines.append(f"  • 🏠 Work from home adoption: {wfh_percentage}% of records")
            
                # Leave patterns
            leave_types = ['Planned Leave', 'Sudden Leave', 'Medical Leave', 'Holiday Leave', 'Lieu leave']
            total_leaves = sum(status_counts.get(leave_type, 0) for leave_type in leave_types)
            if total_leaves > 0:
                leave_percentage = round((total_leaves / total_records) * 100, 1)
                report_lines.append(f"  • 📅 Leave utilization: {leave_percentage}% of records")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"Error generating attendance report: {e}"

@tool
def validate_attendance_data(employee_identifier: str, status: str, attendance_date: str = "") -> Dict[str, Any]:
    """
    Validates attendance data before marking or updating records. Helps prevent errors and provides suggestions.
    employee_identifier can be either employee ID (number) or employee name.
    status: The attendance status to validate
    attendance_date: Date to validate (YYYY-MM-DD format, optional - defaults to today)
    """
    try:
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": [],
            "employee_info": None
        }
        
            # Validate status
        valid_statuses = ["Present", "Work from home", "Planned Leave", "Sudden Leave", 
                        "Medical Leave", "Holiday Leave", "Lieu leave", "Work from out of Rise"]
        if status not in valid_statuses:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Invalid status '{status}'. Valid options: {', '.join(valid_statuses)}")
            
                # Suggest closest match
            status_lower = status.lower()
            for valid_status in valid_statuses:
                if status_lower in valid_status.lower() or valid_status.lower() in status_lower:
                    validation_results["suggestions"].append(f"Did you mean '{valid_status}'?")
                    break
        
            # Validate and parse date
        target_date = None
        if attendance_date:
            try:
                target_date = datetime.strptime(attendance_date, '%Y-%m-%d').date()
            except ValueError:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Invalid date format. Please use YYYY-MM-DD format.")
                validation_results["suggestions"].append(f"Example: {date.today().strftime('%Y-%m-%d')}")
        else:
            target_date = date.today()
        
            # Validate date is not too far in the future
        if target_date and target_date > date.today() + timedelta(days=7):
            validation_results["warnings"].append("Date is more than a week in the future. Are you sure this is correct?")
        
            # Validate date is not too far in the past (more than 3 months)
        if target_date and target_date < date.today() - timedelta(days=90):
            validation_results["warnings"].append("Date is more than 3 months in the past. Consider if this historical update is necessary.")
        
            # Validate employee
        employee = None
        if employee_identifier.isdigit():
                # Lookup by ID
            employee = execute_query(
                "SELECT id, name, email, role FROM employees WHERE id = %s AND is_active = true",
                (int(employee_identifier),)
            )
        else:
                # Lookup by name (fuzzy matching)
            employee = execute_query(
                "SELECT id, name, email, role FROM employees WHERE name ILIKE %s AND is_active = true",
                (f"%{employee_identifier}%",)
            )
        
        if not employee:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Employee '{employee_identifier}' not found or inactive.")
            
                # Try to find similar names
            similar_employees = execute_query(
                "SELECT name FROM employees WHERE name ILIKE %s AND is_active = true LIMIT 3",
                (f"%{employee_identifier[:3]}%",)
            )
            if similar_employees:
                similar_names = [emp['name'] for emp in similar_employees]
                validation_results["suggestions"].append(f"Similar names found: {', '.join(similar_names)}")
            
        elif len(employee) > 1:
            validation_results["is_valid"] = False
            matches = [f"{emp['name']} (ID: {emp['id']})" for emp in employee]
            validation_results["errors"].append(f"Multiple employees found matching '{employee_identifier}'. Please be more specific.")
            validation_results["suggestions"].extend([f"Use: {match}" for match in matches])
        
        else:
                # Valid employee found
            employee_data = employee[0]
            validation_results["employee_info"] = {
                "id": employee_data['id'],
                "name": employee_data['name'],
                "email": employee_data['email'],
                "role": employee_data['role']
            }
            
            employee_id = employee_data['id']
            employee_name = employee_data['name']
            
                # Check if attendance already exists for the date
            if target_date:
                existing = execute_query(
                    "SELECT id, status FROM attendances WHERE employee_id = %s AND attendance_date = %s",
                    (employee_id, target_date)
                )
                
                if existing:
                    current_status = existing[0]['status']
                    if current_status == status:
                        validation_results["warnings"].append(f"Attendance for {employee_name} on {target_date.strftime('%Y-%m-%d')} is already marked as '{status}'.")
                    else:
                        validation_results["warnings"].append(f"Attendance for {employee_name} on {target_date.strftime('%Y-%m-%d')} is currently '{current_status}'. This will update it to '{status}'.")
                
                    # Get recent attendance pattern for context
                recent_attendance = execute_query(
                    """SELECT attendance_date, status FROM attendances 
                       WHERE employee_id = %s AND attendance_date >= %s AND attendance_date < %s
                       ORDER BY attendance_date DESC LIMIT 5""",
                    (employee_id, target_date - timedelta(days=7), target_date)
                )
                
                if recent_attendance:
                    validation_results["context"] = {
                        "recent_pattern": [
                            {"date": rec['attendance_date'].strftime('%Y-%m-%d'), "status": rec['status']}
                            for rec in recent_attendance
                        ]
                    }
                    
                        # Check for patterns
                    recent_statuses = [rec['status'] for rec in recent_attendance]
                    if status == "Sudden Leave" and all(s in ["Present", "Work from home"] for s in recent_statuses):
                        validation_results["suggestions"].append("Employee has been consistently present recently. Sudden leave might need approval.")
                    elif status in ["Present", "Work from home"] and all(s.endswith("Leave") for s in recent_statuses):
                        validation_results["suggestions"].append("Employee is returning from leave period.")
        
            # Business rule validations
        if target_date and validation_results["employee_info"]:
                # Check for weekend work (if company policy)
            if target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                if status == "Present":
                    validation_results["warnings"].append("Marking attendance for weekend day. Verify if this is correct.")
            
                # Check for role-specific rules
            employee_role = validation_results["employee_info"]["role"]
            if employee_role == "hr" and status in ["Sudden Leave", "Medical Leave"]:
                validation_results["suggestions"].append("HR personnel leave might require special handling or coverage.")
            elif employee_role == "leader" and status == "Sudden Leave":
                validation_results["suggestions"].append("Leader absence might require delegation or team notification.")
        
            # Generate summary message
        if validation_results["is_valid"]:
            employee_info = validation_results.get("employee_info", {})
            message = f"✅ Validation passed for {employee_info.get('name', employee_identifier)}"
            if target_date:
                message += f" on {target_date.strftime('%Y-%m-%d')}"
            message += f" with status '{status}'"
            
            if validation_results["warnings"]:
                message += f" ({len(validation_results['warnings'])} warnings)"
        else:
            message = f"❌ Validation failed: {len(validation_results['errors'])} error(s) found"
        
        return {
            "success": validation_results["is_valid"],
            "message": message,
            "validation_details": validation_results,
            "employee_identifier": employee_identifier,
            "status": status,
            "date": target_date.strftime('%Y-%m-%d') if target_date else None
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error validating attendance data: {e}"}

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

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}."

@tool
def get_attendance_statistics(days: int = 7, department: str = None, role: str = None) -> Dict[str, Any]:
    """
    Calculate comprehensive attendance statistics with rates, trends, and insights.
    Provides both quantitative data and human-friendly analysis with actionable recommendations.
    """
    try:
        start_date = date.today() - timedelta(days=days)
        where_conditions = ["a.attendance_date >= %s"]
        params = [start_date]
        
        title_filter_info = "for All Employees"
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
            title_filter_info = f"for '{department}' Department"
        
        if role:
            where_conditions.append("e.role = %s")
            params.append(role.lower())
            title_filter_info = f"for employees with role '{role}'"
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Enhanced query with employee and date information
        detailed_query = f"""
            SELECT a.status, a.attendance_date, e.name as employee_name, 
                   e.role, d.name as department_name,
                   EXTRACT(dow FROM a.attendance_date) as day_of_week
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC
        """
        
        detailed_results = execute_query(detailed_query, params)
        
        if not detailed_results:
            return {
                "success": False,
                "error": "No records found.",
                "human_readable_report": f"I couldn't find any attendance records {title_filter_info} in the last {days} days. This could mean either no one has marked attendance, or the filtering criteria is too specific."
            }
        
        # Calculate comprehensive statistics
        total_records = len(detailed_results)
        unique_employees = len(set(r['employee_name'] for r in detailed_results))
        unique_days = len(set(r['attendance_date'] for r in detailed_results))
        
        # Status distribution
        status_counts = {}
        for record in detailed_results:
            status = record['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate attendance rate
        present_statuses = ['Present', 'Work from home', 'Work from out of Rise']
        present_count = sum(status_counts.get(status, 0) for status in present_statuses)
        attendance_rate = round((present_count / total_records) * 100, 1) if total_records > 0 else 0
        
        # Leave analysis
        leave_statuses = ['Planned Leave', 'Sudden Leave', 'Medical Leave', 'Holiday Leave', 'Lieu leave']
        leave_count = sum(status_counts.get(status, 0) for status in leave_statuses)
        leave_rate = round((leave_count / total_records) * 100, 1) if total_records > 0 else 0
        
        # Day of week analysis
        dow_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                    4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        dow_analysis = {}
        
        for record in detailed_results:
            dow = int(record['day_of_week'])
            dow_name = dow_names[dow]
            if dow_name not in dow_analysis:
                dow_analysis[dow_name] = {'total': 0, 'present': 0}
            
            dow_analysis[dow_name]['total'] += 1
            if record['status'] in present_statuses:
                dow_analysis[dow_name]['present'] += 1
        
        # Find best and worst days
        best_day = None
        worst_day = None
        best_rate = 0
        worst_rate = 100
        
        for day, data in dow_analysis.items():
            if data['total'] > 0:
                day_rate = (data['present'] / data['total']) * 100
                if day_rate > best_rate:
                    best_rate = day_rate
                    best_day = day
                if day_rate < worst_rate:
                    worst_rate = day_rate
                    worst_day = day
        
        # Performance insights
        performance_level = ""
        performance_emoji = ""
        recommendations = []
        
        if attendance_rate >= 90:
            performance_level = "Excellent"
            recommendations.append("Maintain current high standards")
            recommendations.append("Consider recognizing top performers")
        elif attendance_rate >= 80:
            performance_level = "Good"
            recommendations.append("Focus on consistent improvement")
            if leave_rate > 15:
                recommendations.append("Review leave patterns for optimization")
        elif attendance_rate >= 70:
            performance_level = "Needs Improvement"
            recommendations.append("Implement attendance improvement initiatives")
            recommendations.append("Consider flexible work arrangements")
        else:
            performance_level = "Critical"
            recommendations.append("Urgent intervention required")
            recommendations.append("Review HR policies and support systems")
        
        # Work from home insights
        wfh_count = status_counts.get('Work from home', 0)
        wfh_rate = round((wfh_count / total_records) * 100, 1) if total_records > 0 else 0
        
        # Build comprehensive statistics
        statistics_data = []
        for status, count in status_counts.items():
            percentage = round((count / total_records) * 100, 1)
            statistics_data.append({
                "status": status,
                "count": count,
                "percentage": percentage
            })
        
        # Generate human-friendly report
        report_lines = [
            f"Attendance Analysis {title_filter_info}",
            "-" * 50,
            f"Analysis Period: Last {days} days ({start_date.strftime('%B %d')} to {date.today().strftime('%B %d, %Y')})",
            f"Coverage: {unique_employees} employees across {unique_days} working days",
            "",
            f"Overall Performance: {performance_level} ({attendance_rate}%)",
            f"- Total Records Analyzed: {total_records:,}",
            f"- Overall Attendance Rate: {attendance_rate}% ({present_count:,} present days)",
            f"- Leave Utilization: {leave_rate}% ({leave_count:,} leave days)",
            f"- Work from Home: {wfh_rate}% ({wfh_count:,} WFH days)",
            "",
            "Daily Performance:"
        ]
        
        # Add day-wise breakdown
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if day in dow_analysis and dow_analysis[day]['total'] > 0:
                data = dow_analysis[day]
                day_rate = round((data['present'] / data['total']) * 100, 1)
                trend_indicator = "(Best)" if day == best_day else "(Worst)" if day == worst_day else ""
                report_lines.append(f"- {day}: {day_rate}% attendance ({data['present']}/{data['total']} records) {trend_indicator}")
        
        report_lines.extend([
            "",
            "Status Breakdown:"
        ])
        
        # Add detailed status breakdown
        for item in sorted(statistics_data, key=lambda x: x['count'], reverse=True):
            report_lines.append(f"- {item['status']}: {item['count']:,} days ({item['percentage']}%)")
        
        # Add insights and recommendations
        if best_day and worst_day and best_day != worst_day:
            report_lines.extend([
                "",
                "Key Insights:",
                f"- Best Day: {best_day} with {best_rate:.1f}% attendance",
                f"- Challenging Day: {worst_day} with {worst_rate:.1f}% attendance"
            ])
        
        if recommendations:
            report_lines.extend([
                "",
                "Recommendations:"
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        return {
            "success": True,
            "title": f"Attendance Statistics {title_filter_info}",
            "total_records": total_records,
            "unique_employees": unique_employees,
            "unique_days": unique_days,
            "attendance_rate": attendance_rate,
            "leave_rate": leave_rate,
            "wfh_rate": wfh_rate,
            "performance_level": performance_level,
            "best_day": best_day,
            "worst_day": worst_day,
            "statistics": statistics_data,
            "day_of_week_analysis": dow_analysis,
            "recommendations": recommendations,
            "human_readable_report": "\n".join(report_lines)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Database error: {e}",
            "human_readable_report": f"I encountered an issue while analyzing the attendance data: {str(e)}. Please try again or contact your system administrator if the problem persists."
        }

@tool
def get_comprehensive_trend_analysis(days: int = 30, department: str = None) -> Dict[str, Any]:
    """
    Provides deep trend analysis with predictive insights, patterns, and actionable intelligence.
    This is the most advanced analytics tool for understanding attendance patterns and forecasting.
    """
    try:
        start_date = date.today() - timedelta(days=days)
        where_conditions = ["a.attendance_date >= %s"]
        params = [start_date]
        
        scope_info = "Company-wide"
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
            scope_info = f"{department} Department"
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Comprehensive data query
        trend_query = f"""
            SELECT 
                a.attendance_date,
                a.status,
                e.name as employee_name,
                e.role,
                d.name as department_name,
                EXTRACT(dow FROM a.attendance_date) as day_of_week,
                EXTRACT(week FROM a.attendance_date) as week_number,
                EXTRACT(month FROM a.attendance_date) as month_number
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC
        """
        
        results = execute_query(trend_query, params)
        
        if not results:
            return {
                "success": False,
                "message": f"No attendance data available for {scope_info} in the last {days} days for trend analysis."
            }
        
        # Organize data for trend analysis
        daily_trends = {}
        weekly_trends = {}
        employee_patterns = {}
        department_trends = {}
        
        present_statuses = ['Present', 'Work from home', 'Work from out of Rise']
        
        for record in results:
            date_key = record['attendance_date'].strftime('%Y-%m-%d')
            week_key = f"Week {int(record['week_number'])}"
            employee = record['employee_name']
            dept = record['department_name']
            
            # Daily trends
            if date_key not in daily_trends:
                daily_trends[date_key] = {'total': 0, 'present': 0, 'date': record['attendance_date']}
            daily_trends[date_key]['total'] += 1
            if record['status'] in present_statuses:
                daily_trends[date_key]['present'] += 1
            
            # Weekly trends
            if week_key not in weekly_trends:
                weekly_trends[week_key] = {'total': 0, 'present': 0}
            weekly_trends[week_key]['total'] += 1
            if record['status'] in present_statuses:
                weekly_trends[week_key]['present'] += 1
            
            # Employee patterns
            if employee not in employee_patterns:
                employee_patterns[employee] = {'total': 0, 'present': 0, 'dept': dept, 'role': record['role']}
            employee_patterns[employee]['total'] += 1
            if record['status'] in present_statuses:
                employee_patterns[employee]['present'] += 1
            
            # Department trends
            if dept not in department_trends:
                department_trends[dept] = {'total': 0, 'present': 0}
            department_trends[dept]['total'] += 1
            if record['status'] in present_statuses:
                department_trends[dept]['present'] += 1
        
        # Calculate trend metrics
        daily_rates = []
        for day_data in daily_trends.values():
            rate = (day_data['present'] / day_data['total']) * 100 if day_data['total'] > 0 else 0
            daily_rates.append(rate)
        
        # Trend direction analysis
        if len(daily_rates) >= 7:
            recent_week = sum(daily_rates[-7:]) / 7
            previous_week = sum(daily_rates[-14:-7]) / 7 if len(daily_rates) >= 14 else recent_week
            trend_direction = "Improving" if recent_week > previous_week + 2 else "Declining" if recent_week < previous_week - 2 else "Stable"
            trend_change = round(recent_week - previous_week, 1)
        else:
            trend_direction = "Insufficient data"
            trend_change = 0
        
        # Best and worst performers
        top_performers = sorted(
            [(emp, data['present']/data['total']*100) for emp, data in employee_patterns.items() if data['total'] >= 3],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        concerning_patterns = sorted(
            [(emp, data['present']/data['total']*100) for emp, data in employee_patterns.items() if data['total'] >= 3],
            key=lambda x: x[1]
        )[:3]
        
        # Department comparison
        dept_performance = []
        for dept, data in department_trends.items():
            rate = (data['present'] / data['total']) * 100 if data['total'] > 0 else 0
            dept_performance.append((dept, rate, data['total']))
        dept_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Generate insights
        insights = []
        
        # Overall trend insight
        overall_rate = sum(daily_rates) / len(daily_rates) if daily_rates else 0
        if overall_rate >= 90:
            insights.append("Exceptional attendance performance across the organization")
        elif overall_rate >= 80:
            insights.append("Strong attendance performance with room for optimization")
        else:
            insights.append("Attendance performance requires immediate attention")
        
        # Trend insights
        if "Improving" in trend_direction:
            insights.append(f"Positive trend: Attendance improved by {trend_change}% in the last week")
        elif "Declining" in trend_direction:
            insights.append(f"Attention needed: Attendance declined by {abs(trend_change)}% in the last week")
        
        # Weekly pattern insights
        if len(weekly_trends) > 2:
            week_rates = [(week, (data['present']/data['total'])*100) for week, data in weekly_trends.items() if data['total'] > 0]
            if week_rates:
                best_week = max(week_rates, key=lambda x: x[1])
                worst_week = min(week_rates, key=lambda x: x[1])
                insights.append(f"Best performing week: {best_week[0]} ({best_week[1]:.1f}% attendance)")
                if best_week[1] - worst_week[1] > 10:
                    insights.append(f"High weekly variation detected (best: {best_week[1]:.1f}%, worst: {worst_week[1]:.1f}%)")
        
        # Build comprehensive report
        report_lines = [
            f"Comprehensive Trend Analysis - {scope_info}",
            "-" * 50,
            f"Analysis Period: {days} days ({start_date.strftime('%B %d')} - {date.today().strftime('%B %d, %Y')})",
            f"Data Points: {len(results):,} attendance records analyzed",
            "",
            f"Overall Performance: {overall_rate:.1f}% Average Attendance",
            f"Trend Direction: {trend_direction}",
            "",
            "Key Insights:"
        ]
        
        for insight in insights:
            report_lines.append(f"- {insight}")
        
        # Top performers section
        if top_performers:
            report_lines.extend([
                "",
                "Top Performers:"
            ])
            for i, (emp, rate) in enumerate(top_performers, 1):
                emp_data = employee_patterns[emp]
                report_lines.append(f"{i}. {emp} ({emp_data['dept']}) - {rate:.1f}% attendance")
        
        # Concerning patterns
        if concerning_patterns and concerning_patterns[0][1] < 80:
            report_lines.extend([
                "",
                "Attention Required:"
            ])
            for emp, rate in concerning_patterns:
                if rate < 80:
                    emp_data = employee_patterns[emp]
                    report_lines.append(f"- {emp} ({emp_data['dept']}) - {rate:.1f}% attendance")
        
        # Department comparison
        if len(dept_performance) > 1:
            report_lines.extend([
                "",
                "Department Performance:"
            ])
            for dept, rate, total in dept_performance:
                report_lines.append(f"- {dept}: {rate:.1f}% ({total:,} records)")
        
        # Weekly trends
        if len(weekly_trends) > 1:
            report_lines.extend([
                "",
                "Weekly Trends:"
            ])
            for week in sorted(weekly_trends.keys(), key=lambda x: int(x.split()[1])):
                data = weekly_trends[week]
                rate = (data['present'] / data['total']) * 100 if data['total'] > 0 else 0
                trend_indicator = "(Above avg)" if rate >= overall_rate else "(Below avg)"
                report_lines.append(f"- {week}: {rate:.1f}% attendance {trend_indicator}")
        
        # Recommendations
        recommendations = []
        if overall_rate < 85:
            recommendations.append("Implement attendance improvement program")
            recommendations.append("Review and address underlying issues")
        if "Declining" in trend_direction:
            recommendations.append("Investigate recent changes affecting attendance")
            recommendations.append("Consider employee engagement initiatives")
        if concerning_patterns:
            recommendations.append("Provide targeted support for underperforming employees")
        if len(dept_performance) > 1 and dept_performance[0][1] - dept_performance[-1][1] > 15:
            recommendations.append("Address departmental attendance disparities")
        
        if recommendations:
            report_lines.extend([
                "",
                "Strategic Recommendations:"
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        return {
            "success": True,
            "scope": scope_info,
            "analysis_period": days,
            "overall_rate": round(overall_rate, 1),
            "trend_direction": trend_direction,
            "trend_change": trend_change,
            "top_performers": top_performers[:3],
            "concerning_patterns": [emp for emp, rate in concerning_patterns if rate < 80],
            "department_performance": dept_performance,
            "insights": insights,
            "recommendations": recommendations,
            "human_readable_report": "\n".join(report_lines)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "human_readable_report": f"I encountered an issue while performing the trend analysis: {str(e)}. This might be due to insufficient data or a temporary system issue."
        }

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
# ENHANCED ATTENDANCE TOOLS WITH HUMAN LOOP FUNCTIONALITY
# ==============================================================================

@tool
def get_enhanced_attendance_summary(
    date_filter: str = "this_month",  
    start_date: str = "",  
    end_date: str = "",    
    role_filter: str = "all"  
) -> Dict[str, Any]:
    """
    Get a comprehensive attendance summary with human loop functionality.
    First shows summary statistics, then asks for user confirmation before showing detailed data.
    
    Parameters:
    - date_filter: "today", "this_week", "this_month", "date_range", "last_week", "last_month"
    - start_date: Custom start date (YYYY-MM-DD format, required if date_filter="date_range")
    - end_date: Custom end date (YYYY-MM-DD format, required if date_filter="date_range")  
    - role_filter: "all", "leader", "hr", "employee" (excludes labour as they're in separate table)
    
    Returns summary first, then generates follow-up questions for detailed data.
    """
    try:
        # Calculate date range based on filter
        today = date.today()
        
        if date_filter == "today":
            query_start_date = today
            query_end_date = today
            period_desc = "Today"
        elif date_filter == "this_week":
            # Start from Monday of current week
            days_since_monday = today.weekday()
            query_start_date = today - timedelta(days=days_since_monday)
            query_end_date = today
            period_desc = "This Week"
        elif date_filter == "last_week":
            # Start from Monday of last week
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            query_start_date = last_monday
            query_end_date = last_monday + timedelta(days=6)
            period_desc = "Last Week"
        elif date_filter == "this_month":
            query_start_date = today.replace(day=1)
            query_end_date = today
            period_desc = "This Month"
        elif date_filter == "last_month":
            # Get first day of last month
            first_day_this_month = today.replace(day=1)
            query_end_date = first_day_this_month - timedelta(days=1)
            query_start_date = query_end_date.replace(day=1)
            period_desc = "Last Month"
        elif date_filter == "date_range":
            if not start_date or not end_date:
                return {
                    "success": False,
                    "error": "start_date and end_date are required when using date_range filter",
                    "requires_confirmation": False
                }
            try:
                query_start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                query_end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                period_desc = f"{start_date} to {end_date}"
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid date format. Use YYYY-MM-DD",
                    "requires_confirmation": False
                }
        else:
            # Default to this month
            query_start_date = today.replace(day=1)
            query_end_date = today
            period_desc = "This Month"
        
        # Build query conditions
        where_conditions = ["a.attendance_date >= %s", "a.attendance_date <= %s"]
        params = [query_start_date, query_end_date]
        
        role_desc = "All Employees"
        if role_filter != "all":
            valid_roles = ["leader", "hr", "employee"]
            if role_filter.lower() not in valid_roles:
                return {
                    "success": False,
                    "error": f"Invalid role filter. Valid options: {', '.join(valid_roles)}",
                    "requires_confirmation": False
                }
            where_conditions.append("e.role = %s")
            params.append(role_filter.lower())
            role_desc = f"{role_filter.title()} Role"
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Execute summary query
        summary_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT e.id) as unique_employees,
                COUNT(DISTINCT a.attendance_date) as unique_days,
                COUNT(DISTINCT d.name) as departments_involved,
                a.status,
                e.role,
                d.name as department_name,
                COUNT(*) as status_count
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id  
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            GROUP BY a.status, e.role, d.name
        """
        
        summary_results = execute_query(summary_query, params)
        
        if not summary_results:
            return {
                "success": False,
                "message": f"No attendance records found for {role_desc} in {period_desc}",
                "requires_confirmation": False,
                "period": period_desc,
                "role_filter": role_desc
            }
        
        # Process summary statistics
        total_records = 0
        unique_employees = set()
        unique_days = set()
        departments = set()
        status_summary = {}
        role_breakdown = {}
        dept_breakdown = {}
        
        for record in summary_results:
            total_records += record['status_count']
            unique_employees.add(record.get('unique_employees'))
            departments.add(record['department_name'])
            
            # Status summary
            status = record['status']
            if status not in status_summary:
                status_summary[status] = 0
            status_summary[status] += record['status_count']
            
            # Role breakdown
            role = record['role']
            if role not in role_breakdown:
                role_breakdown[role] = {'total': 0, 'by_status': {}}
            role_breakdown[role]['total'] += record['status_count']
            if status not in role_breakdown[role]['by_status']:
                role_breakdown[role]['by_status'][status] = 0
            role_breakdown[role]['by_status'][status] += record['status_count']
            
            # Department breakdown  
            dept = record['department_name']
            if dept not in dept_breakdown:
                dept_breakdown[dept] = 0
            dept_breakdown[dept] += record['status_count']
        
        # Calculate attendance rate
        present_statuses = ['Present', 'Work from home', 'Work from out of Rise']
        present_count = sum(status_summary.get(status, 0) for status in present_statuses)
        attendance_rate = round((present_count / total_records) * 100, 1) if total_records > 0 else 0
        
        # Get actual unique counts
        unique_employee_count = len(unique_employees) if unique_employees != {None} else 0
        unique_days_count = (query_end_date - query_start_date).days + 1
        departments_count = len(departments)
        
        # Create summary response
        summary_lines = [
            f"Attendance Summary - {role_desc} ({period_desc})",
            "-" * 50,
            f"Overview:",
            f"- Total Records: {total_records:,} attendance entries",
            "",
            f"Status Breakdown:"
        ]
        
        # Add status breakdown
        for status, count in sorted(status_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = round((count / total_records) * 100, 1)
            summary_lines.append(f"- {status}: {count:,} ({percentage}%)")
        
        # Add department breakdown if multiple departments
        if len(dept_breakdown) > 1:
            summary_lines.extend([
                "",
                f"Top Departments:"
            ])
            sorted_depts = sorted(dept_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]
            for dept, count in sorted_depts:
                percentage = round((count / total_records) * 100, 1)
                summary_lines.append(f"- {dept}: {count:,} records ({percentage}%)")
        
        return {
            "success": True,
            "summary_type": "overview",
            "period": period_desc,
            "role_filter": role_desc,
            "total_records": total_records,
            "unique_employees": unique_employee_count,
            "departments_involved": departments_count,
            "attendance_rate": attendance_rate,
            "status_summary": status_summary,
            "requires_confirmation": True,
            "follow_up_question": "Would you like me to show detailed breakdown with individual employee data and trends?",
            "human_readable_report": "\n".join(summary_lines),
            "query_params": {
                "date_filter": date_filter,
                "start_date": str(query_start_date),
                "end_date": str(query_end_date),
                "role_filter": role_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_enhanced_attendance_summary: {e}")
        return {
            "success": False,
            "error": f"Failed to generate attendance summary: {str(e)}",
            "requires_confirmation": False
        }

@tool
def get_role_attendance_report(
    role: str,  
    period: str = "this_month",
    include_task_leaders: bool = True,
    start_date: str = "",
    end_date: str = ""
) -> Dict[str, Any]:
    """
    Get detailed attendance report specifically filtered by employee role.
    Supports special handling for task group leaders.
    
    Parameters:
    - role: "leader", "hr", "employee" (excludes labour as they're in separate table)
    - period: "today", "this_week", "this_month", "last_week", "last_month", "date_range"
    - include_task_leaders: Whether to include employees who are task group leaders
    - start_date: Custom start date (YYYY-MM-DD, required if period="date_range")
    - end_date: Custom end date (YYYY-MM-DD, required if period="date_range")
    """
    try:
        # Validate role
        valid_roles = ["leader", "hr", "employee"]
        if role.lower() not in valid_roles:
            return {
                "success": False,
                "error": f"Invalid role '{role}'. Valid options: {', '.join(valid_roles)}"
            }
        
        # Calculate date range
        today = date.today()
        
        if period == "today":
            query_start_date = today
            query_end_date = today
            period_desc = "Today"
        elif period == "this_week":
            days_since_monday = today.weekday()
            query_start_date = today - timedelta(days=days_since_monday)
            query_end_date = today
            period_desc = "This Week"
        elif period == "last_week":
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            query_start_date = last_monday
            query_end_date = last_monday + timedelta(days=6)
            period_desc = "Last Week"
        elif period == "this_month":
            query_start_date = today.replace(day=1)
            query_end_date = today
            period_desc = "This Month"
        elif period == "last_month":
            first_day_this_month = today.replace(day=1)
            query_end_date = first_day_this_month - timedelta(days=1)
            query_start_date = query_end_date.replace(day=1)
            period_desc = "Last Month"
        elif period == "date_range":
            if not start_date or not end_date:
                return {
                    "success": False,
                    "error": "start_date and end_date are required when using date_range period"
                }
            try:
                query_start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                query_end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                period_desc = f"{start_date} to {end_date}"
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid date format. Use YYYY-MM-DD"
                }
        else:
            query_start_date = today.replace(day=1)
            query_end_date = today
            period_desc = "This Month"
        
        # Build base query
        where_conditions = [
            "a.attendance_date >= %s", 
            "a.attendance_date <= %s",
            "e.role = %s"
        ]
        params = [query_start_date, query_end_date, role.lower()]
        
        # Add task leader filter if requested
        task_leader_join = ""
        if include_task_leaders and role.lower() == "leader":
            task_leader_join = "LEFT JOIN task_group_leaders tgl ON e.id = tgl.employee_id"
            # Note: We don't filter here, just join to get the data
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Execute detailed query
        detailed_query = f"""
            SELECT 
                e.id as employee_id,
                e.name as employee_name,
                e.email as employee_email,
                e.role as employee_role,
                d.name as department_name,
                a.attendance_date,
                a.status,
                a.notes,
                a.check_in_time,
                a.check_out_time,
                CASE WHEN tgl.employee_id IS NOT NULL THEN true ELSE false END as is_task_leader
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {task_leader_join}
            {where_clause}
            ORDER BY e.name, a.attendance_date DESC
        """
        
        detailed_results = execute_query(detailed_query, params)
        
        if not detailed_results:
            return {
                "success": False,
                "message": f"No attendance records found for {role} role in {period_desc}",
                "role": role,
                "period": period_desc
            }
        
        # Process results
        employee_data = {}
        total_records = len(detailed_results)
        status_summary = {}
        department_breakdown = {}
        task_leader_count = 0
        
        present_statuses = ['Present', 'Work from home', 'Work from out of Rise']
        
        for record in detailed_results:
            emp_id = record['employee_id']
            emp_name = record['employee_name']
            
            # Initialize employee data
            if emp_id not in employee_data:
                employee_data[emp_id] = {
                    'name': emp_name,
                    'email': record['employee_email'],
                    'department': record['department_name'],
                    'is_task_leader': record.get('is_task_leader', False),
                    'attendance_records': [],
                    'total_days': 0,
                    'present_days': 0,
                    'status_breakdown': {}
                }
                
                if record.get('is_task_leader'):
                    task_leader_count += 1
            
            # Add attendance record
            employee_data[emp_id]['attendance_records'].append({
                'date': record['attendance_date'],
                'status': record['status'],
                'notes': record.get('notes', ''),
                'check_in_time': record.get('check_in_time'),
                'check_out_time': record.get('check_out_time')
            })
            
            # Update employee stats
            employee_data[emp_id]['total_days'] += 1
            status = record['status']
            if status in present_statuses:
                employee_data[emp_id]['present_days'] += 1
            
            # Update employee status breakdown
            if status not in employee_data[emp_id]['status_breakdown']:
                employee_data[emp_id]['status_breakdown'][status] = 0
            employee_data[emp_id]['status_breakdown'][status] += 1
            
            # Update overall status summary
            if status not in status_summary:
                status_summary[status] = 0
            status_summary[status] += 1
            
            # Update department breakdown
            dept = record['department_name']
            if dept not in department_breakdown:
                department_breakdown[dept] = {'total': 0, 'present': 0, 'employees': set()}
            department_breakdown[dept]['total'] += 1
            department_breakdown[dept]['employees'].add(emp_name)
            if status in present_statuses:
                department_breakdown[dept]['present'] += 1
        
        # Calculate overall statistics
        unique_employees = len(employee_data)
        total_present = sum(status_summary.get(status, 0) for status in present_statuses)
        overall_attendance_rate = round((total_present / total_records) * 100, 1) if total_records > 0 else 0
        
        # Generate employee performance ranking
        employee_performance = []
        for emp_id, data in employee_data.items():
            attendance_rate = round((data['present_days'] / data['total_days']) * 100, 1) if data['total_days'] > 0 else 0
            employee_performance.append({
                'name': data['name'],
                'department': data['department'],
                'attendance_rate': attendance_rate,
                'total_days': data['total_days'],
                'present_days': data['present_days'],
                'is_task_leader': data['is_task_leader'],
                'status_breakdown': data['status_breakdown']
            })
        
        # Sort by attendance rate
        employee_performance.sort(key=lambda x: x['attendance_rate'], reverse=True)
        
        # Build report
        role_title = role.title()
        report_lines = [
            f"{role_title} Attendance Report - {period_desc}",
            "-" * 50,
            f"Overview:",
            f"- Total {role_title}s: {unique_employees} employees",
            f"- Total Records: {total_records:,} attendance entries",
            f"- Overall Attendance Rate: {overall_attendance_rate}% ({'Excellent' if overall_attendance_rate >= 90 else 'Good' if overall_attendance_rate >= 80 else 'Needs attention'})",
            f"- Departments Involved: {len(department_breakdown)} departments"
        ]
        
        # Add task leader info for leaders
        if role.lower() == "leader" and include_task_leaders:
            report_lines.append(f"- Task Group Leaders: {task_leader_count} out of {unique_employees} leaders")
        
        # Status breakdown
        report_lines.extend([
            "",
            "Status Breakdown:"
        ])
        
        for status, count in sorted(status_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = round((count / total_records) * 100, 1)
            report_lines.append(f"- {status}: {count:,} ({percentage}%)")
        
        # Department performance
        if len(department_breakdown) > 1:
            report_lines.extend([
                "",
                "Department Performance:"
            ])
            
            dept_performance = []
            for dept, data in department_breakdown.items():
                dept_rate = round((data['present'] / data['total']) * 100, 1) if data['total'] > 0 else 0
                dept_performance.append((dept, dept_rate, len(data['employees']), data['total']))
            
            dept_performance.sort(key=lambda x: x[1], reverse=True)
            
            for dept, rate, emp_count, records in dept_performance:
                report_lines.append(f"- {dept}: {rate}% ({emp_count} {role}s, {records} records)")
        
        # Top and concerning performers
        top_performers = employee_performance[:5]
        concerning_performers = [emp for emp in employee_performance if emp['attendance_rate'] < 80]
        
        if top_performers:
            report_lines.extend([
                "",
                f"Top Performing {role_title}s:"
            ])
            for i, emp in enumerate(top_performers, 1):
                task_leader_indicator = " (Task Leader)" if emp.get('is_task_leader') else ""
                report_lines.append(f"{i}. {emp['name']}{task_leader_indicator} ({emp['department']}) - {emp['attendance_rate']}%")
        
        if concerning_performers:
            report_lines.extend([
                "",
                f"{role_title}s Needing Attention:"
            ])
            for emp in concerning_performers[:5]:  # Show top 5 concerning
                task_leader_indicator = " (Task Leader)" if emp.get('is_task_leader') else ""
                report_lines.append(f"- {emp['name']}{task_leader_indicator} ({emp['department']}) - {emp['attendance_rate']}%")
        
        return {
            "success": True,
            "role": role,
            "period": period_desc,
            "total_employees": unique_employees,
            "total_records": total_records,
            "overall_attendance_rate": overall_attendance_rate,
            "task_leaders_count": task_leader_count if role.lower() == "leader" else 0,
            "status_summary": status_summary,
            "department_breakdown": department_breakdown,
            "employee_performance": employee_performance,
            "top_performers": top_performers,
            "concerning_performers": concerning_performers,
            "human_readable_report": "\n".join(report_lines)
        }
        
    except Exception as e:
        logger.error(f"Error in get_role_attendance_report: {e}")
        return {
            "success": False,
            "error": f"Failed to generate role attendance report: {str(e)}"
        }

@tool
def get_employees_by_attendance_status(
    attendance_date: str = "",  
    date_period: str = "",      
    status_filter: str = "all", 
    role_filter: str = "all"    
) -> Dict[str, Any]:
    """
    Get employee names and details filtered by attendance status for specific dates or periods.
    Useful for finding who was present, absent, on leave, etc. for specific dates or time periods.
    
    Parameters:
    - attendance_date: Specific date (YYYY-MM-DD format) or "today" - use this OR date_period, not both
    - date_period: "this_week", "this_month", "last_week", "last_month" - use this OR attendance_date
    - status_filter: "all", "Present", "Work from home", "Planned Leave", "Sudden Leave", etc.
    - role_filter: "all", "leader", "hr", "employee" (excludes labour as they're in separate table)
    
    Returns list of employees with their attendance details for the specified criteria.
    """
    try:
        # Validate inputs
        if attendance_date and date_period:
            return {
                "success": False,
                "error": "Please specify either attendance_date OR date_period, not both"
            }
        
        if not attendance_date and not date_period:
            # Default to today
            attendance_date = "today"
        
        # Calculate date range
        today = date.today()
        
        if attendance_date:
            if attendance_date.lower() == "today":
                query_start_date = today
                query_end_date = today
                period_desc = "Today"
            else:
                try:
                    query_start_date = datetime.strptime(attendance_date, "%Y-%m-%d").date()
                    query_end_date = query_start_date
                    period_desc = attendance_date
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid date format. Use YYYY-MM-DD or 'today'"
                    }
        else:
            # Handle date period
            if date_period == "this_week":
                days_since_monday = today.weekday()
                query_start_date = today - timedelta(days=days_since_monday)
                query_end_date = today
                period_desc = "This Week"
            elif date_period == "last_week":
                days_since_monday = today.weekday()
                last_monday = today - timedelta(days=days_since_monday + 7)
                query_start_date = last_monday
                query_end_date = last_monday + timedelta(days=6)
                period_desc = "Last Week"
            elif date_period == "this_month":
                query_start_date = today.replace(day=1)
                query_end_date = today
                period_desc = "This Month"
            elif date_period == "last_month":
                first_day_this_month = today.replace(day=1)
                query_end_date = first_day_this_month - timedelta(days=1)
                query_start_date = query_end_date.replace(day=1)
                period_desc = "Last Month"
            else:
                return {
                    "success": False,
                    "error": "Invalid date_period. Valid options: 'this_week', 'this_month', 'last_week', 'last_month'"
                }
        
        # Build query conditions
        where_conditions = ["a.attendance_date >= %s", "a.attendance_date <= %s"]
        params = [query_start_date, query_end_date]
        
        # Add status filter
        status_desc = "All Statuses"
        if status_filter != "all":
            valid_statuses = ["Present", "Work from home", "Planned Leave", "Sudden Leave", 
                            "Medical Leave", "Holiday Leave", "Lieu leave", "Work from out of Rise"]
            if status_filter not in valid_statuses:
                return {
                    "success": False,
                    "error": f"Invalid status filter. Valid options: {', '.join(valid_statuses)}"
                }
            where_conditions.append("a.status = %s")
            params.append(status_filter)
            status_desc = status_filter
        
        # Add role filter
        role_desc = "All Employees"
        if role_filter != "all":
            valid_roles = ["leader", "hr", "employee"]
            if role_filter.lower() not in valid_roles:
                return {
                    "success": False,
                    "error": f"Invalid role filter. Valid options: {', '.join(valid_roles)}"
                }
            where_conditions.append("e.role = %s")
            params.append(role_filter.lower())
            role_desc = f"{role_filter.title()} Role"
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Execute query
        query = f"""
            SELECT 
                e.id as employee_id,
                e.name as employee_name,
                e.email as employee_email,
                e.role as employee_role,
                e.phone_number,
                d.name as department_name,
                a.attendance_date,
                a.status,
                a.notes,
                a.check_in_time,
                a.check_out_time,
                CASE WHEN tgl.employee_id IS NOT NULL THEN true ELSE false END as is_task_leader
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            LEFT JOIN task_group_leaders tgl ON e.id = tgl.employee_id
            {where_clause}
            ORDER BY a.attendance_date DESC, e.name
        """
        
        results = execute_query(query, params)
        
        if not results:
            return {
                "success": False,
                "message": f"No employees found with {status_desc} status for {role_desc} in {period_desc}",
                "period": period_desc,
                "status_filter": status_desc,
                "role_filter": role_desc
            }
        
        # Process results
        employee_list = []
        unique_employees = {}
        date_breakdown = {}
        department_summary = {}
        status_summary = {}
        
        for record in results:
            emp_id = record['employee_id']
            emp_name = record['employee_name']
            attendance_date = record['attendance_date']
            status = record['status']
            dept = record['department_name']
            
            # Track unique employees
            if emp_id not in unique_employees:
                unique_employees[emp_id] = {
                    'name': emp_name,
                    'email': record['employee_email'],
                    'role': record['employee_role'],
                    'department': dept,
                    'phone_number': record.get('phone_number', ''),
                    'is_task_leader': record.get('is_task_leader', False),
                    'attendance_dates': [],
                    'status_count': {}
                }
            
            # Add attendance record
            unique_employees[emp_id]['attendance_dates'].append({
                'date': attendance_date,
                'status': status,
                'notes': record.get('notes', ''),
                'check_in_time': record.get('check_in_time'),
                'check_out_time': record.get('check_out_time')
            })
            
            # Update status count for this employee
            if status not in unique_employees[emp_id]['status_count']:
                unique_employees[emp_id]['status_count'][status] = 0
            unique_employees[emp_id]['status_count'][status] += 1
            
            # Track date breakdown
            date_str = attendance_date.strftime('%Y-%m-%d')
            if date_str not in date_breakdown:
                date_breakdown[date_str] = {'employees': set(), 'total': 0}
            date_breakdown[date_str]['employees'].add(emp_name)
            date_breakdown[date_str]['total'] += 1
            
            # Track department summary
            if dept not in department_summary:
                department_summary[dept] = {'employees': set(), 'total': 0}
            department_summary[dept]['employees'].add(emp_name)
            department_summary[dept]['total'] += 1
            
            # Track status summary
            if status not in status_summary:
                status_summary[status] = {'employees': set(), 'total': 0}
            status_summary[status]['employees'].add(emp_name)
            status_summary[status]['total'] += 1
        
        # Convert to list format
        for emp_id, data in unique_employees.items():
            employee_list.append({
                'employee_id': emp_id,
                'name': data['name'],
                'email': data['email'],
                'role': data['role'],
                'department': data['department'],
                'phone_number': data['phone_number'],
                'is_task_leader': data['is_task_leader'],
                'total_attendance_days': len(data['attendance_dates']),
                'attendance_records': data['attendance_dates'],
                'status_breakdown': data['status_count']
            })
        
        # Sort by name
        employee_list.sort(key=lambda x: x['name'])
        
        # Build report
        report_lines = [
            f"Employee List - {status_desc} ({period_desc})",
            "-" * 50,
            f"Summary:",
            f"- Total Employees Found: {len(unique_employees)} {role_desc.lower()}",
            f"- Total Records: {len(results)} attendance entries",
            f"- Period: {period_desc}",
            f"- Status Filter: {status_desc}",
            f"- Role Filter: {role_desc}"
        ]
        
        # Add department breakdown if multiple departments
        if len(department_summary) > 1:
            report_lines.extend([
                "",
                "By Department:"
            ])
            sorted_depts = sorted(department_summary.items(), key=lambda x: len(x[1]['employees']), reverse=True)
            for dept, data in sorted_depts:
                report_lines.append(f"- {dept}: {len(data['employees'])} employees ({data['total']} records)")
        
        # Add status breakdown if showing all statuses
        if status_filter == "all" and len(status_summary) > 1:
            report_lines.extend([
                "",
                "By Status:"
            ])
            for status, data in sorted(status_summary.items(), key=lambda x: len(x[1]['employees']), reverse=True):
                report_lines.append(f"- {status}: {len(data['employees'])} employees ({data['total']} records)")
        
        # Add employee list
        report_lines.extend([
            "",
            "Employee Details:"
        ])
        
        for employee in employee_list:
            task_leader_indicator = " (Task Leader)" if employee['is_task_leader'] else ""
            
            report_lines.append(f"- {employee['name']}{task_leader_indicator}")
            report_lines.append(f"  Email: {employee['email']} | Department: {employee['department']} | Role: {employee['role']}")
            
            if employee['phone_number']:
                report_lines.append(f"  Phone: {employee['phone_number']}")
            
            # Show attendance details if multiple dates
            if employee['total_attendance_days'] > 1:
                report_lines.append(f"  Total attendance days: {employee['total_attendance_days']}")
                # Show status breakdown
                status_parts = []
                for status, count in employee['status_breakdown'].items():
                    if count > 0:
                        status_parts.append(f"{status}: {count}")
                if status_parts:
                    report_lines.append(f"  Status breakdown: {', '.join(status_parts)}")
            else:
                # Single day - show the specific details
                record = employee['attendance_records'][0]
                report_lines.append(f"  Date: {record['date']} - Status: {record['status']}")
                if record['notes']:
                    report_lines.append(f"  Notes: {record['notes']}")
            
            report_lines.append("")  # Add spacing between employees
        
        return {
            "success": True,
            "period": period_desc,
            "status_filter": status_desc,
            "role_filter": role_desc,
            "total_employees": len(unique_employees),
            "total_records": len(results),
            "employee_list": employee_list,
            "department_breakdown": {dept: len(data['employees']) for dept, data in department_summary.items()},
            "status_breakdown": {status: len(data['employees']) for status, data in status_summary.items()},
            "date_breakdown": {date: len(data['employees']) for date, data in date_breakdown.items()},
            "human_readable_report": "\n".join(report_lines)
        }
        
    except Exception as e:
        logger.error(f"Error in get_employees_by_attendance_status: {e}")
        return {
            "success": False,
            "error": f"Failed to get employees by attendance status: {str(e)}"
        }

@tool
def generate_attendance_followup_queries(
    context_data: Dict[str, Any]
) -> List[str]:
    """
    Automatically generate relevant follow-up questions based on current attendance data patterns.
    This helps create a human loop by suggesting next actions or queries the user might want to explore.
    
    Parameters:
    - context_data: Dictionary containing attendance analysis results from previous queries
    
    Returns list of contextually relevant follow-up questions.
    """
    try:
        queries = []
        
        # Extract context information
        attendance_rate = context_data.get('attendance_rate', 0)
        total_employees = context_data.get('total_employees', 0)
        period = context_data.get('period', '')
        role_filter = context_data.get('role_filter', '')
        status_summary = context_data.get('status_summary', {})
        concerning_performers = context_data.get('concerning_performers', [])
        top_performers = context_data.get('top_performers', [])
        department_breakdown = context_data.get('department_breakdown', {})
        
        # Base queries that are always relevant
        base_queries = [
            "Show me detailed attendance trends over the last 3 months",
            "Which employees have perfect attendance this month?",
            "Get attendance statistics by department",
            "Show me attendance patterns by day of the week"
        ]
        
        # Context-specific queries based on attendance rate
        if attendance_rate < 70:
            queries.extend([
                "What are the main reasons for low attendance this period?",
                "Show me employees who need immediate attention for attendance",
                "Generate an attendance improvement action plan",
                "Compare this period's attendance with the previous period",
                "Which departments have the lowest attendance rates?"
            ])
        elif attendance_rate < 85:
            queries.extend([
                "Identify patterns in leave requests and absences",
                "Show me attendance comparison between departments",
                "Which employees are showing declining attendance trends?",
                "Generate targeted interventions for attendance improvement"
            ])
        else:
            queries.extend([
                "Recognize top performing employees for their excellent attendance",
                "Share best practices from high-performing departments",
                "Create attendance excellence report for management"
            ])
        
        # Role-specific follow-ups
        if 'leader' in role_filter.lower():
            queries.extend([
                "Show task group leaders' attendance and their team impact",
                "Compare attendance rates between task group leaders and regular leaders",
                "Which leaders maintain consistent attendance for team guidance?"
            ])
        elif 'hr' in role_filter.lower():
            queries.extend([
                "Show HR team attendance patterns and coverage",
                "Identify backup coverage needs for HR functions",
                "Compare HR attendance with company averages"
            ])
        
        # Status-based follow-ups
        if status_summary:
            wfh_count = status_summary.get('Work from home', 0)
            leave_statuses = ['Planned Leave', 'Sudden Leave', 'Medical Leave', 'Holiday Leave', 'Lieu leave']
            total_leaves = sum(status_summary.get(status, 0) for status in leave_statuses)
            total_records = sum(status_summary.values())
            
            if wfh_count > 0 and total_records > 0:
                wfh_percentage = (wfh_count / total_records) * 100
                if wfh_percentage > 30:
                    queries.append("Analyze work from home patterns and productivity correlation")
                elif wfh_percentage > 15:
                    queries.append("Review work from home policy effectiveness")
            
            if total_leaves > 0:
                leave_percentage = (total_leaves / total_records) * 100
                if leave_percentage > 25:
                    queries.append("Investigate high leave utilization patterns")
                    queries.append("Check if leave policies need adjustment")
        
        # Performance-based follow-ups
        if concerning_performers:
            queries.extend([
                f"Create individual improvement plans for {len(concerning_performers)} underperforming employees",
                "Schedule one-on-one meetings with employees needing attendance support",
                "Identify root causes for attendance issues"
            ])
        
        if top_performers:
            queries.extend([
                f"Prepare recognition for {len(top_performers)} top-performing employees",
                "Document best practices from high-attendance employees",
                "Consider top performers for attendance mentorship roles"
            ])
        
        # Department-based follow-ups
        if len(department_breakdown) > 1:
            queries.extend([
                "Compare attendance rates across all departments",
                "Identify best-performing department practices to replicate",
                "Schedule department-specific attendance reviews"
            ])
        
        # Time-based follow-ups
        if period:
            queries.extend([
                f"Compare {period.lower()} attendance with the same period last year",
                "Show attendance trends leading up to this period",
                "Predict attendance patterns for the next month"
            ])
        
        # Employee count-based follow-ups
        if total_employees > 20:
            queries.append("Generate executive summary for large team attendance")
        elif total_employees > 0:
            queries.append("Create detailed individual attendance profiles")
        
        # Data quality and insights
        queries.extend([
            "Validate attendance data for any anomalies or missing entries",
            "Generate attendance forecast for workforce planning",
            "Create automated attendance alerts for proactive management"
        ])
        
        # Remove duplicates and limit to most relevant
        queries = list(dict.fromkeys(queries))  # Remove duplicates while preserving order
        
        # Add base queries if we don't have many context-specific ones
        if len(queries) < 5:
            queries.extend([q for q in base_queries if q not in queries])
        
        # Limit to top 8-10 most relevant queries
        return queries[:10]
        
    except Exception as e:
        logger.error(f"Error generating follow-up queries: {e}")
        # Return some basic queries as fallback
        return [
            "Show detailed attendance report for this month",
            "Get attendance statistics by role and department", 
            "Identify employees needing attendance improvement",
            "Compare current attendance with previous periods",
            "Generate attendance trends analysis"
        ]

# ==============================================================================
# LANGCHAIN GRAPH SETUP
# ==============================================================================

tools = [
    mark_attendance, update_attendance_history, get_employee_attendance_summary,
    get_attendance_insights, get_attendance_report, validate_attendance_data,
    get_all_employees_overview, get_tasks_report, get_current_datetime, 
    get_attendance_statistics, get_comprehensive_trend_analysis, 
    escalate_to_human, intelligent_decision_check, ask_company_documents,
    # New enhanced attendance tools with human loop functionality
    get_enhanced_attendance_summary, get_role_attendance_report, 
    get_employees_by_attendance_status, generate_attendance_followup_queries
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

**📋 Attendance Management Made Easy:**
- Mark attendance naturally - just tell me "I'm working from home today" or "Mark John as present"
- Update past records with context - "Yesterday I was actually on medical leave"
- Validate data before changes - I'll catch potential issues and suggest fixes
- Handle complex scenarios - "Everyone in IT worked from home last Friday due to the server maintenance"

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

## Enhanced Attendance Capabilities 🚀

**Role-Based Queries:**
- Query attendance by specific roles: leaders, HR staff, employees
- Special handling for task group leaders and their team responsibilities  
- Compare performance across different employee roles

**Smart Date Filtering:**
- Support for "today", "this week", "this month", "last month", custom date ranges
- Automatic period comparisons and trend analysis
- Flexible date queries like "show me who was present yesterday"

**Contextual Follow-ups:**
When I analyze attendance data, I automatically suggest relevant questions like:
- "Would you like to see which employees have perfect attendance?"
- "Should I compare this with previous months?"
- "Want me to identify employees needing attention?"

---

Ready to dive in? Whether you need to update attendance, analyze trends, or just understand what's happening with your team, I'm here to help in whatever way makes most sense for your situation. What would you like to explore?
"""

# ==============================================================================
# LANGCHAIN-POWERED UTILITY FUNCTIONS
# ==============================================================================

def mark_attendance_direct(employee_id: int, status: str = "Present", notes: str = "") -> Dict[str, Any]:
    """Direct attendance marking using LangChain tool - for API endpoints"""
    return mark_attendance(str(employee_id), status, notes)

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

@app.post("/attendance/mark", response_model=APIResponse)
async def mark_employee_attendance(request: AttendanceMarkRequest):
    """Mark attendance for an employee"""
    try:
        result = mark_attendance_direct(request.employee_id, request.status.value, request.notes)
        return APIResponse(
            success=result["success"],
            message=result["message"],
            data=result if result["success"] else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance", response_model=List[dict])
async def get_attendance_records(
    employee_name: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    status: Optional[str] = Query(None),
    days: int = Query(7)
):
    """Get attendance records with filtering"""
    try:
        where_conditions = []
        params = []
        
        if start_date and end_date:
            where_conditions.append("a.attendance_date >= %s AND a.attendance_date <= %s")
            params.extend([start_date, end_date])
        else:
            start_date = date.today() - timedelta(days=days)
            where_conditions.append("a.attendance_date >= %s")
            params.append(start_date)
        
        if employee_name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{employee_name}%")
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
        
        if status:
            where_conditions.append("a.status = %s")
            params.append(status)
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
            SELECT a.id, a.attendance_date, a.status, e.id as employee_id,
                   e.name as employee_name, d.name as department_name
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC, e.name
        """
        
        records = execute_query(query, params)
        return [dict(record) for record in records]
        
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

@app.get("/attendance/statistics")
async def get_attendance_stats(
    days: int = Query(7),
    department: Optional[str] = Query(None),
    role: Optional[str] = Query(None)
):
    """Get attendance statistics"""
    try:
        start_date = date.today() - timedelta(days=days)
        where_conditions = ["a.attendance_date >= %s"]
        params = [start_date]
        
        title_filter_info = "for All Employees"
        
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
            title_filter_info = f"for '{department}' Department"
        
        if role:
            where_conditions.append("e.role = %s")
            params.append(role.lower())
            title_filter_info = f"for employees with role '{role}'"
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        query = f"""
            SELECT a.status, COUNT(*) as count
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            GROUP BY a.status
        """
        
        results = execute_query(query, params)
        
        if not results:
            return {
                "error": "No records found.",
                "human_readable_report": f"No attendance records found {title_filter_info} in the last {days} days."
            }
        
        total_records = sum(item['count'] for item in results)
        statistics_data = [
            {
                "status": item['status'],
                "count": item['count'],
                "percentage": round((item['count']/total_records)*100, 2)
            }
            for item in results
        ]
        
        return {
            "title": f"Attendance Statistics {title_filter_info}",
            "total_records": total_records,
            "statistics": statistics_data
        }
        
    except Exception as e:
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
            
            # Async stream the response with memory
            async for chunk in langgraph_app.astream({"messages": messages}, config):
                if "call_model" in chunk:
                    ai_message = chunk["call_model"]["messages"][-1]
                    if hasattr(ai_message, 'content') and ai_message.content:
                        response_obj = {"type": "text", "content": ai_message.content}
                        yield f"data: {json.dumps(response_obj)}\n\n"
                        final_response.append(response_obj)
            
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(generate_response(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Critical chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/debug/attendance")
async def debug_attendance(
    days: int = Query(7),
    department: Optional[str] = Query(None),
    employee_name: Optional[str] = Query(None),
    detailed: bool = Query(False)
):
    """Debug endpoint for attendance calculation issues"""
    try:
        logger.info(f"Debug attendance endpoint called with days={days}, dept={department}, employee={employee_name}")
        
        # Build query conditions for debugging
        where_conditions = []
        params = []
        
        # Date range logic
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        where_conditions.append("a.attendance_date >= %s AND a.attendance_date <= %s")
        params.extend([start_date, end_date])
        
        # Filter conditions
        if department:
            where_conditions.append("d.name ILIKE %s")
            params.append(f"%{department}%")
            
        if employee_name:
            where_conditions.append("e.name ILIKE %s")
            params.append(f"%{employee_name}%")
            
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Debug query - same as main attendance query
        debug_query = f"""
            SELECT a.attendance_date, a.status, e.name as employee_name, 
                   d.name as department_name, e.role, e.id as employee_id,
                   EXTRACT(dow FROM a.attendance_date) as day_of_week,
                   EXTRACT(week FROM a.attendance_date) as week_number,
                   to_char(a.attendance_date, 'Day') as day_name
            FROM attendances a
            JOIN employees e ON a.employee_id = e.id
            JOIN departments d ON e.department_id = d.id
            {where_clause}
            ORDER BY a.attendance_date DESC, e.name
            LIMIT 100
        """
        
        logger.info("Executing debug query...")
        records = execute_query(debug_query, params)
        
        # Analyze the data
        debug_info = {
            "query_executed": debug_query,
            "parameters": params,
            "total_records": len(records) if records else 0,
            "date_range": f"{start_date} to {end_date}",
            "filters": {
                "department": department,
                "employee_name": employee_name
            }
        }
        
        if records:
            # Status distribution analysis
            status_counts = {}
            dow_counts = {}
            dow_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                        4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
            
            for record in records:
                # Status analysis
                status = record['status']
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Day of week analysis
                if record.get('day_of_week') is not None:
                    dow = int(record['day_of_week'])
                    dow_name = dow_names.get(dow, f'Unknown_{dow}')
                    if dow_name not in dow_counts:
                        dow_counts[dow_name] = {'total': 0, 'present': 0, 'sample_dates': []}
                    
                    dow_counts[dow_name]['total'] += 1
                    if record['status'] in ['Present', 'Work from home', 'Work from out of Rise']:
                        dow_counts[dow_name]['present'] += 1
                    
                    # Add sample dates for debugging
                    if len(dow_counts[dow_name]['sample_dates']) < 3:
                        dow_counts[dow_name]['sample_dates'].append(str(record['attendance_date']))
            
            debug_info.update({
                "status_distribution": status_counts,
                "day_of_week_analysis": dow_counts,
                "sample_records": records[:5] if detailed else records[:2],
                "unique_employees": len(set(r['employee_name'] for r in records)),
                "unique_departments": len(set(r['department_name'] for r in records)),
                "date_span": {
                    "earliest": str(min(r['attendance_date'] for r in records)),
                    "latest": str(max(r['attendance_date'] for r in records))
                }
            })
            
            # Calculate attendance rates
            present_count = sum(
                count for status, count in status_counts.items()
                if status in ['Present', 'Work from home', 'Work from out of Rise']
            )
            overall_rate = round((present_count / len(records)) * 100, 1) if records else 0
            debug_info["calculated_attendance_rate"] = f"{overall_rate}% ({present_count}/{len(records)})"
            
        else:
            debug_info.update({
                "message": "No records found",
                "possible_reasons": [
                    "No attendance data in the specified date range",
                    "Department/employee filters are too restrictive",
                    "Database connection issues",
                    "Table structure changes"
                ]
            })
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        return {
            "error": str(e),
            "message": "Debug endpoint failed - check logs for details"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)