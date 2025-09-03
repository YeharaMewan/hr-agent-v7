#!/usr/bin/env python3
"""
Initialize SQLite database with tables and sample data
"""
import sqlite3
import os
from datetime import datetime, date

def init_database(db_path='hr_management.db'):
    """Initialize SQLite database with tables and sample data"""
    
    if os.path.exists(db_path):
        # Remove existing database to start fresh
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Creating database: {db_path}")
    
    # Create departments table
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            role TEXT,
            department_id INTEGER REFERENCES departments(id),
            phone_number TEXT,
            address TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create attendances table
    cursor.execute("""
        CREATE TABLE attendances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id),
            attendance_date DATE NOT NULL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            check_in_time TIMESTAMP,
            check_out_time TIMESTAMP,
            notes TEXT
        )
    """)
    
    # Create other necessary tables
    cursor.execute("""
        CREATE TABLE leave_balances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id),
            year INTEGER NOT NULL,
            total_days INTEGER NOT NULL,
            days_used INTEGER NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE hr_policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            chunk_text TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id),
            leave_date DATE NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'pending'
        )
    """)
    
    cursor.execute("""
        CREATE TABLE task_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE task_group_leaders (
            task_group_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            PRIMARY KEY (task_group_id, employee_id),
            FOREIGN KEY (task_group_id) REFERENCES task_groups(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_group_id INTEGER REFERENCES task_groups(id),
            task_title TEXT NOT NULL,
            location TEXT,
            expected_days INTEGER,
            priority TEXT DEFAULT 'Medium',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE labours (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            skill TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE task_labours (
            task_id INTEGER NOT NULL,
            labour_id INTEGER NOT NULL,
            PRIMARY KEY (task_id, labour_id),
            FOREIGN KEY (task_id) REFERENCES tasks(id),
            FOREIGN KEY (labour_id) REFERENCES labours(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    print("Created all tables successfully")
    
    # Insert sample departments
    departments = [
        ('AI',),
        ('Marketing',),
        ('HR',),
        ('Operations',),
        ('Finance',),
        ('Tech',)
    ]
    
    cursor.executemany("INSERT INTO departments (name) VALUES (?)", departments)
    print(f"Inserted {len(departments)} departments")
    
    # Insert sample employees
    employees = [
        ('Simon Johnson', 'simon.johnson@company.com', 'leader', 1, '+1-555-0101', '123 Main St'),
        ('Alice Smith', 'alice.smith@company.com', 'employee', 2, '+1-555-0102', '456 Oak Ave'),
        ('John Doe', 'john.doe@company.com', 'hr', 3, '+1-555-0103', '789 Pine Rd'),
        ('Bob Wilson', 'bob.wilson@company.com', 'employee', 4, '+1-555-0104', '321 Elm St'),
        ('Sarah Davis', 'sarah.davis@company.com', 'leader', 2, '+1-555-0105', '654 Maple Dr'),
        ('Mike Brown', 'mike.brown@company.com', 'employee', 1, '+1-555-0106', '987 Cedar Ln'),
    ]
    
    cursor.executemany("""
        INSERT INTO employees (name, email, role, department_id, phone_number, address) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, employees)
    
    print(f"Inserted {len(employees)} employees")
    
    # Insert some sample attendance records
    attendance_records = [
        (1, date.today().isoformat(), 'Present', datetime.now().isoformat(), None, 'Regular day'),
        (2, date.today().isoformat(), 'Work from home', datetime.now().isoformat(), None, 'WFH today'),
        (3, date.today().isoformat(), 'Present', datetime.now().isoformat(), None, 'In office'),
    ]
    
    cursor.executemany("""
        INSERT INTO attendances (employee_id, attendance_date, status, check_in_time, check_out_time, notes) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, attendance_records)
    
    print(f"Inserted {len(attendance_records)} attendance records")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Database initialization completed: {db_path}")
    return True

def main():
    """Main function"""
    init_database()
    print("SQLite database setup complete!")

if __name__ == "__main__":
    main()