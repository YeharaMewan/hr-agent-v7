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

class EmployeeNotFoundCreateOffer(BaseModel):
    success: bool = True
    message: str
    response_type: str = "employee_not_found_create_offer"
    suggested_name: str = ""
    follow_up_questions: List[str] = []

class EmployeeUpdateFormRequest(BaseModel):
    success: bool = True
    message: str
    response_type: str = "employee_found_update_form"
    action: str = "update_employee"
    data: dict  # Current employee data
    form_data: dict  # Form configuration with current values
    follow_up_questions: List[str] = []

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