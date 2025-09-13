from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import date, datetime
from enum import Enum

class AttendanceStatus(str, Enum):
    PRESENT = "Present"
    WORK_FROM_HOME = "Work from home"
    PLANNED_LEAVE = "Planned Leave"
    SUDDEN_LEAVE = "Sudden Leave"
    MEDICAL_LEAVE = "Medical Leave"
    HOLIDAY_LEAVE = "Holiday Leave"
    LIEU_LEAVE = "Lieu leave"
    WORK_FROM_OUT = "Work from out of Rise"

class TaskPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"

class EmployeeRole(str, Enum):
    HR = "hr"
    LEADER = "leader"
    EMPLOYEE = "employee"

# Employee Models
class EmployeeBase(BaseModel):
    name: str
    email: str
    role: EmployeeRole
    phone_number: Optional[str] = None
    address: Optional[str] = None
    is_active: bool = True

class EmployeeCreate(EmployeeBase):
    department_id: int

class Employee(EmployeeBase):
    id: int
    department_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class EmployeeWithDepartment(Employee):
    department_name: Optional[str] = None

# Department Models
class DepartmentBase(BaseModel):
    name: str

class Department(DepartmentBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Attendance Models
class AttendanceBase(BaseModel):
    attendance_date: date
    status: AttendanceStatus
    check_in_time: Optional[datetime] = None
    check_out_time: Optional[datetime] = None
    notes: Optional[str] = None

class AttendanceCreate(AttendanceBase):
    employee_id: int

class Attendance(AttendanceBase):
    id: int
    employee_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class AttendanceWithEmployee(Attendance):
    employee_name: Optional[str] = None
    department_name: Optional[str] = None

class AttendanceMarkRequest(BaseModel):
    employee_id: int
    status: AttendanceStatus = AttendanceStatus.PRESENT
    notes: Optional[str] = ""

# Task Models
class TaskGroupBase(BaseModel):
    group_name: str

class TaskGroup(TaskGroupBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TaskBase(BaseModel):
    task_title: str
    location: Optional[str] = None
    expected_days: Optional[int] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    notes: Optional[str] = None

class TaskCreate(TaskBase):
    task_group_id: int

class Task(TaskBase):
    id: int
    task_group_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class TaskWithDetails(Task):
    group_name: Optional[str] = None
    leaders: List[str] = []
    labourers: List[str] = []

# Leave Models
class LeaveRequestBase(BaseModel):
    leave_date: date
    reason: Optional[str] = None

class LeaveRequestCreate(LeaveRequestBase):
    employee_id: int

class LeaveRequest(LeaveRequestBase):
    id: int
    employee_id: int
    status: str = "pending"
    
    class Config:
        from_attributes = True

class LeaveBalance(BaseModel):
    id: int
    employee_id: int
    year: int
    total_days: int
    days_used: int
    
    class Config:
        from_attributes = True

# Labour Models
class LabourBase(BaseModel):
    name: str
    skill: str

class LabourCreate(LabourBase):
    pass

class Labour(LabourBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# HR Policy Models
class HRPolicyBase(BaseModel):
    document_name: str
    chunk_text: str

class HRPolicyCreate(HRPolicyBase):
    pass

class HRPolicy(HRPolicyBase):
    id: int
    embedding: Optional[List[float]] = None
    
    class Config:
        from_attributes = True

# Document Models
class DocumentBase(BaseModel):
    content: str
    metadata: Optional[dict] = None

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Junction Table Models
class TaskGroupLeader(BaseModel):
    task_group_id: int
    employee_id: int
    
    class Config:
        from_attributes = True

class TaskLabour(BaseModel):
    task_id: int
    labour_id: int
    
    class Config:
        from_attributes = True

# API Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# Enhanced Employee Query Response Models
class EmployeeQueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    response_type: str  # "individual_profile", "department_list", "all_employees_summary", "small_group", etc.
    total_found: int
    follow_up_questions: List[str] = []

class DepartmentSummaryResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Department data with employee counts and roles
    total_employees: int
    total_departments: int
    response_type: str = "department_summary"
    follow_up_questions: List[str] = []

class IndividualEmployeeResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Complete employee profile
    response_type: str = "individual_profile"
    search_type: Optional[str] = None  # "id", "email", "name"
    follow_up_questions: List[str] = []

class EmployeeSummaryData(BaseModel):
    department: str
    employees: List[dict]
    role_groups: dict

class MultipleMatchesResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]  # List of matching employees
    response_type: str = "multiple_matches"
    total_found: int
    follow_up_questions: List[str] = []

# Employee Management Response Models (Create/Update)
class EmployeeFormRequest(BaseModel):
    success: bool
    message: str
    response_type: str = "form_request"
    action: str  # "create_employee" or "update_employee"
    form_data: dict  # Contains pre_filled values, departments, roles, etc.
    follow_up_questions: List[str] = []

class EmployeeValidationError(BaseModel):
    success: bool = False
    message: str
    response_type: str = "validation_error"
    suggestions: List[str] = []
    follow_up_questions: List[str] = []

class EmployeeCreatedResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "success_created"
    data: dict  # Created employee data
    follow_up_questions: List[str] = []

class EmployeeUpdatedResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "success_updated"
    data: dict  # Updated employee data
    changes_made: List[str] = []
    update_reason: Optional[str] = None
    follow_up_questions: List[str] = []


class EmployeeUpdateFormRequest(BaseModel):
    success: bool = True
    message: str
    response_type: str = "employee_found_update_form"
    action: str = "update_employee"
    data: dict  # Current employee data
    form_data: dict  # Form configuration with current values
    follow_up_questions: List[str] = []

class SimilarEmployeesFoundResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "similar_employees_found"
    searched_name: str
    data: List[dict]  # Similar employees data
    total_found: int
    follow_up_questions: List[str] = []

class SimilarEmployeeFoundForUpdateResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "similar_employee_found_for_update"
    searched_name: str
    suggested_employee: dict  # Best match employee data
    all_similar: List[dict]  # All similar employees
    follow_up_questions: List[str] = []

class HumanLoopQuestionResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "human_loop_question"
    conversation_state: dict  # Track what action to take on confirmation
    follow_up_questions: List[str] = []
    pending_action: str  # "create_employee", "update_employee", etc.
    suggested_data: Optional[dict] = None  # Data to use when confirmed

class FormDataResponse(BaseModel):
    success: bool = True
    message: str
    data: dict  # Contains departments, roles, validation_rules, field_labels

class AttendanceReport(BaseModel):
    title: str
    total_records: int
    date_range: str
    records: List[AttendanceWithEmployee]

class AttendanceStatistics(BaseModel):
    total_records: int
    statistics: List[dict]
    human_readable_report: str

# Chat Models
class ChatMessage(BaseModel):
    message: str

# Form-based Employee Management Models
class EmployeeCreateRequest(BaseModel):
    query: str = ""
    name: str
    email: str
    role: str
    department: str
    phone_number: Optional[str] = ""
    address: Optional[str] = ""

class EmployeeUpdateRequest(BaseModel):
    query: str = ""
    employee_identifier: str
    field_updates: dict[str, Any]
    update_reason: str = ""

class ChatResponse(BaseModel):
    type: str
    content: Optional[str] = None
    payload: Optional[dict] = None

# Task Assignment Models
class TaskGroupLeaderAssignment(BaseModel):
    task_group_id: int
    employee_id: int

class TaskLabourerAssignment(BaseModel):
    task_id: int
    labour_id: int

# Query Models
class AttendanceQuery(BaseModel):
    employee_name: Optional[str] = None
    department: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[AttendanceStatus] = None
    days: int = 7

class TaskQuery(BaseModel):
    leader_name: Optional[str] = None
    department: Optional[str] = None
    priority: Optional[TaskPriority] = None
    location: Optional[str] = None
    days: int = 30

class EmployeeQuery(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[EmployeeRole] = None
    is_active: Optional[bool] = None

# Enhanced Task Query and Response Models
class TaskTimeFilter(str, Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    LAST_3_MONTHS = "last_3_months"
    LAST_6_MONTHS = "last_6_months"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"

class TaskWithFullDetails(Task):
    group_name: Optional[str] = None
    leader_names: List[str] = []
    leader_departments: List[str] = []
    labour_names: List[str] = []
    labour_skills: List[str] = []
    department_names: List[str] = []

class TaskQueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    response_type: str  # "individual_task", "task_group_summary", "all_tasks_summary", "small_group", etc.
    total_found: int
    follow_up_questions: List[str] = []
    time_period: Optional[str] = None
    filters_applied: dict = {}

class TaskGroupSummaryResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Task group data with task counts and priorities
    total_tasks: int
    total_groups: int
    response_type: str = "task_group_summary"
    follow_up_questions: List[str] = []
    time_period: Optional[str] = None

class IndividualTaskResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Complete task details
    response_type: str = "individual_task"
    search_type: Optional[str] = None  # "id", "title", "priority"
    follow_up_questions: List[str] = []

class TaskSummaryData(BaseModel):
    group_name: str
    tasks: List[dict]
    priority_breakdown: dict
    leader_info: List[dict]

class MultipleTasksResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]  # List of matching tasks
    response_type: str = "multiple_tasks"
    total_found: int
    follow_up_questions: List[str] = []
    time_period: Optional[str] = None
    filters_applied: dict = {}

class TaskStatisticsResponse(BaseModel):
    success: bool
    message: str
    response_type: str = "task_statistics"
    data: dict  # Contains statistics breakdown
    time_period: Optional[str] = None
    total_tasks: int
    priority_breakdown: dict
    completion_trends: dict
    follow_up_questions: List[str] = []

class TaskNotFoundResponse(BaseModel):
    success: bool = False
    message: str
    response_type: str = "task_not_found"
    suggestions: List[str] = []
    follow_up_questions: List[str] = []
    alternative_queries: List[str] = []

class TaskHumanLoopQuestionResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "task_human_loop_question"
    conversation_state: dict  # Track what action to take on confirmation
    follow_up_questions: List[str] = []
    pending_action: str  # "filter_tasks", "show_details", etc.
    suggested_filters: Optional[dict] = None

# Enhanced Task Query Models
class TaskIntelligentQuery(BaseModel):
    query: str
    time_filter: Optional[TaskTimeFilter] = None
    priority_filter: Optional[List[TaskPriority]] = None
    leader_names: Optional[List[str]] = None
    labour_skills: Optional[List[str]] = None
    departments: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    include_completed: bool = True
    limit: Optional[int] = None

# Enhanced Attendance Query and Response Models
class AttendanceQueryClarificationResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "attendance_clarification"
    query_analysis: dict  # Contains parsed query components and what's missing/ambiguous
    clarification_needed: List[str]  # What needs clarification: ["department", "date_range", "specific_date"]
    suggestions: dict  # Contains suggested departments, date options, etc.
    follow_up_questions: List[str] = []
    user_query: str  # Original user query for context

class AttendanceAmbiguousQueryResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "ambiguous_attendance_query"
    detected_intent: str  # "department_attendance", "employee_attendance", etc.
    ambiguous_elements: List[str]  # ["unknown_department", "vague_date", etc.]
    suggested_clarifications: List[dict]  # List of clarification options
    conversation_state: dict  # Track what action to take on confirmation
    follow_up_questions: List[str] = []
    pending_action: str = "attendance_query"

class AttendanceDepartmentNotFoundResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "attendance_department_not_found"
    searched_department: str
    similar_departments: List[str] = []
    available_departments: List[str] = []
    conversation_state: dict
    follow_up_questions: List[str] = []
    pending_action: str = "attendance_query_with_department"

# Enhanced Labour Query and Response Models
class LabourWorkloadStatus(str, Enum):
    AVAILABLE = "available"
    LIGHT_LOAD = "light_load"
    MODERATE_LOAD = "moderate_load"
    HEAVY_LOAD = "heavy_load"
    OVERLOADED = "overloaded"

class LabourWithFullDetails(Labour):
    current_task_count: int = 0
    current_tasks: List[str] = []
    task_groups: List[str] = []
    working_under_leaders: List[str] = []
    leader_departments: List[str] = []
    workload_status: Optional[LabourWorkloadStatus] = None
    recent_task_history: List[dict] = []
    skill_utilization: Optional[float] = None

class LabourQueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    response_type: str  # "individual_labour", "skill_group", "workload_analysis", "multiple_labourers", etc.
    total_found: int
    follow_up_questions: List[str] = []
    time_period: Optional[str] = None
    filters_applied: dict = {}

class IndividualLabourResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Complete labour profile with task assignments
    response_type: str = "individual_labour"
    search_type: Optional[str] = None  # "name", "skill", "id"
    follow_up_questions: List[str] = []
    workload_analysis: dict = {}

class LabourSkillGroupResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]  # Labour grouped by skills
    response_type: str = "skill_group"
    skill_name: str
    total_found: int
    workload_distribution: dict = {}
    follow_up_questions: List[str] = []

class LabourWorkloadAnalysisResponse(BaseModel):
    success: bool
    message: str
    data: dict  # Workload statistics and analysis
    response_type: str = "workload_analysis"
    total_labourers: int
    workload_breakdown: dict  # available, light, moderate, heavy, overloaded counts
    skill_distribution: dict
    recommendations: List[str] = []
    follow_up_questions: List[str] = []

class MultipleLabourersResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]  # List of matching labourers
    response_type: str = "multiple_labourers"
    total_found: int
    skill_breakdown: dict = {}
    workload_summary: dict = {}
    follow_up_questions: List[str] = []
    time_period: Optional[str] = None
    filters_applied: dict = {}

class LabourStatisticsResponse(BaseModel):
    success: bool
    message: str
    response_type: str = "labour_statistics"
    data: dict  # Contains statistics breakdown
    time_period: Optional[str] = None
    total_labourers: int
    skill_breakdown: dict
    workload_trends: dict
    task_assignment_patterns: dict
    follow_up_questions: List[str] = []

class LabourNotFoundResponse(BaseModel):
    success: bool = False
    message: str
    response_type: str = "labour_not_found"
    suggestions: List[str] = []
    follow_up_questions: List[str] = []
    alternative_queries: List[str] = []
    similar_skills: List[str] = []

class LabourHumanLoopQuestionResponse(BaseModel):
    success: bool = True
    message: str
    response_type: str = "labour_human_loop_question"
    conversation_state: dict  # Track what action to take on confirmation
    follow_up_questions: List[str] = []
    pending_action: str  # "filter_labourers", "show_workload", "assign_tasks", etc.
    suggested_filters: Optional[dict] = None
    labour_recommendations: List[str] = []

# Labour Query Models
class LabourIntelligentQuery(BaseModel):
    query: str
    skill_filter: Optional[List[str]] = None
    workload_filter: Optional[List[LabourWorkloadStatus]] = None
    labour_names: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    leader_names: Optional[List[str]] = None
    departments: Optional[List[str]] = None
    time_filter: Optional[TaskTimeFilter] = None
    include_task_history: bool = True
    include_workload_analysis: bool = True
    limit: Optional[int] = None

class LabourTaskAssignmentSummary(BaseModel):
    labour_id: int
    labour_name: str
    skill: str
    assigned_tasks: List[dict]
    task_count: int
    workload_status: LabourWorkloadStatus
    average_task_priority: Optional[str] = None
    leader_relationships: List[str] = []