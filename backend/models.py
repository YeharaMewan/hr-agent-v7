from pydantic import BaseModel
from typing import Optional, List
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
    LABOURER = "labourer"

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
    
    class Config:
        from_attributes = True

class EmployeeWithDepartment(Employee):
    department_name: Optional[str] = None

# Department Models
class DepartmentBase(BaseModel):
    name: str

class Department(DepartmentBase):
    id: int
    
    class Config:
        from_attributes = True

# Attendance Models
class AttendanceBase(BaseModel):
    attendance_date: date
    status: AttendanceStatus

class AttendanceCreate(AttendanceBase):
    employee_id: int

class Attendance(AttendanceBase):
    id: int
    employee_id: int
    
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

# API Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

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
    employee_id: int

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