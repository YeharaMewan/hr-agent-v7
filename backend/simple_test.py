#!/usr/bin/env python3
"""
Simple test server to verify employee management workflow
without complex LangChain dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import asyncio
import time

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test data
EMPLOYEES = [
    {"id": 1, "name": "John Doe", "email": "john@company.com", "role": "Software Engineer", "department": "IT", "phone_number": "+1234567890", "address": "123 Main St"},
    {"id": 2, "name": "Jane Smith", "email": "jane@company.com", "role": "HR Manager", "department": "Human Resources", "phone_number": "+1234567891", "address": "456 Oak Ave"},
    {"id": 3, "name": "Simon Torres", "email": "simon@company.com", "role": "Team Leader", "department": "AI", "phone_number": "+1234567892", "address": "789 Pine St"}
]

DEPARTMENTS = [
    {"id": 1, "name": "IT", "label": "Information Technology", "value": 1},
    {"id": 2, "name": "Human Resources", "label": "Human Resources", "value": 2}, 
    {"id": 3, "name": "AI", "label": "Artificial Intelligence", "value": 3},
    {"id": 4, "name": "Construction", "label": "Construction", "value": 4}
]

ROLES = [
    {"value": "Software Engineer", "label": "Software Engineer"},
    {"value": "HR Manager", "label": "HR Manager"},
    {"value": "Team Leader", "label": "Team Leader"},
    {"value": "Project Manager", "label": "Project Manager"},
    {"value": "Developer", "label": "Developer"}
]

class ChatMessage(BaseModel):
    message: str

class EmployeeCreateRequest(BaseModel):
    query: str
    name: str
    email: str
    role: str
    department: str
    phone_number: Optional[str] = ""
    address: Optional[str] = ""

class EmployeeUpdateRequest(BaseModel):
    query: str
    employee_identifier: str
    field_updates: Dict[str, Any]
    update_reason: str

@app.get("/")
async def root():
    return {
        "service": "HR Management System - Test Server",
        "status": "running",
        "version": "test-1.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

async def simulate_ai_response(message: str):
    """Simulate AI processing and determine response type"""
    message_lower = message.lower()
    
    # Test scenarios
    if "update" in message_lower and ("simon" in message_lower or "hasitha" in message_lower):
        # Find employee
        employee = None
        for emp in EMPLOYEES:
            if "simon" in message_lower and "simon" in emp["name"].lower():
                employee = emp
                break
            elif "hasitha" in message_lower:
                # Simulate employee not found, but we'll pretend we found one for testing
                employee = EMPLOYEES[0]  # Use John as placeholder
                break
        
        if employee:
            content = f"I found employee {employee['name']}. "
            yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
            await asyncio.sleep(0.5)
            
            # Send form request
            form_data = {
                "action": "update_employee",
                "employee_id": employee["id"],
                "current_values": {
                    "name": employee["name"],
                    "email": employee["email"], 
                    "role": employee["role"],
                    "department": employee["department"],
                    "phone_number": employee.get("phone_number", ""),
                    "address": employee.get("address", "")
                },
                "departments": DEPARTMENTS,
                "roles": ROLES
            }
            
            yield f"data: {json.dumps({'type': 'form_request', 'action': 'update_employee', 'form_data': form_data, 'content': 'Please review and update the employee details:'})}\n\n"
            return
    
    elif "create" in message_lower and "employee" in message_lower:
        # Simulate new employee creation
        yield f"data: {json.dumps({'type': 'text', 'content': 'I can help you create a new employee record. '})}\n\n"
        await asyncio.sleep(0.5)
        
        # Send form request for creation
        form_data = {
            "action": "create_employee", 
            "pre_filled": {},
            "departments": DEPARTMENTS,
            "roles": ROLES
        }
        
        yield f"data: {json.dumps({'type': 'form_request', 'action': 'create_employee', 'form_data': form_data, 'content': 'Please fill in the new employee details:'})}\n\n"
        return
    
    elif "new user" in message_lower or "add employee" in message_lower:
        # Simulate employee not found scenario  
        yield f"data: {json.dumps({'type': 'text', 'content': 'I could not find that employee in the system. '})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'employee_not_found_create_offer', 'content': 'This appears to be a new user. Would you like to add them as an employee?'})}\n\n"
        await asyncio.sleep(1)
        
        # After user acceptance (simulated), show form
        form_data = {
            "action": "create_employee",
            "pre_filled": {},
            "departments": DEPARTMENTS, 
            "roles": ROLES
        }
        
        yield f"data: {json.dumps({'type': 'form_request', 'action': 'create_employee', 'form_data': form_data, 'content': 'Opening employee creation form...'})}\n\n"
        return
    
    else:
        # Default response
        yield f"data: {json.dumps({'type': 'text', 'content': 'I understand you want to work with employee records. Try asking me to:'})}\n\n"
        await asyncio.sleep(0.5)
        yield f"data: {json.dumps({'type': 'text', 'content': '\\n• \"Update Simon Torres details\" - to test employee updates'})}\n\n"
        await asyncio.sleep(0.3)
        yield f"data: {json.dumps({'type': 'text', 'content': '\\n• \"Create new employee\" - to test employee creation'})}\n\n"
        await asyncio.sleep(0.3)
        yield f"data: {json.dumps({'type': 'text', 'content': '\\n• \"Add new user John Smith\" - to test not found scenario'})}\n\n"
    
    # Final completion
    yield f"data: {json.dumps({'type': 'complete'})}\n\n"

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Simulate chat with form triggering"""
    
    async def generate_response():
        async for chunk in simulate_ai_response(message.message):
            yield chunk
            await asyncio.sleep(0.1)  # Small delay between chunks
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/tools/create_employee_intelligent")
async def create_employee(request: EmployeeCreateRequest):
    """Simulate employee creation"""
    new_id = max(emp["id"] for emp in EMPLOYEES) + 1
    new_employee = {
        "id": new_id,
        "name": request.name,
        "email": request.email,
        "role": request.role,
        "department": request.department,
        "phone_number": request.phone_number,
        "address": request.address
    }
    
    EMPLOYEES.append(new_employee)
    
    return {
        "success": True,
        "message": f"Employee {request.name} created successfully with ID {new_id}",
        "employee_id": new_id,
        "employee": new_employee
    }

@app.post("/tools/update_employee_intelligent") 
async def update_employee(request: EmployeeUpdateRequest):
    """Simulate employee update"""
    employee_id = int(request.employee_identifier)
    
    # Find employee
    employee = None
    for i, emp in enumerate(EMPLOYEES):
        if emp["id"] == employee_id:
            employee = emp
            employee_index = i
            break
    
    if not employee:
        return {
            "success": False,
            "message": f"Employee with ID {employee_id} not found"
        }
    
    # Update fields
    for field, value in request.field_updates.items():
        if field in employee:
            employee[field] = value
    
    EMPLOYEES[employee_index] = employee
    
    return {
        "success": True,
        "message": f"Employee {employee['name']} updated successfully",
        "employee": employee
    }

@app.post("/tools/get_form_data_for_frontend")
async def get_form_data():
    """Return form configuration data"""
    return {
        "departments": DEPARTMENTS,
        "roles": ROLES,
        "validation_rules": {
            "required_fields": ["name", "email", "role", "department"],
            "email_format": "^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting HR Management Test Server...")
    print("Backend will be available at: http://localhost:8000")
    print("Frontend should be running at: http://localhost:5173")
    print("\nTest scenarios:")
    print("1. 'Update Simon Torres details' - Tests update form")
    print("2. 'Create new employee' - Tests creation form") 
    print("3. 'Add new user John Smith' - Tests not found scenario")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)