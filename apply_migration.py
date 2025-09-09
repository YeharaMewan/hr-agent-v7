#!/usr/bin/env python3
"""
Apply database migration script to add description column
"""

import os
import sys
import psycopg2
from urllib.parse import urlparse

def apply_migration():
    """Apply database migration to add description column to departments table"""
    # Database URL from environment
    database_url = "postgresql://postgres.ovtkppkbfkdldjkfbetb:tharusha123%23@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
    
    try:
        # Parse the database URL
        parsed_url = urlparse(database_url)
        
        # Create connection
        conn = psycopg2.connect(
            host=parsed_url.hostname,
            database=parsed_url.path[1:],  # Remove leading slash
            user=parsed_url.username,
            password=parsed_url.password,
            port=parsed_url.port
        )
        
        cur = conn.cursor()
        
        print("Connecting to database...")
        
        # Check if description column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'departments' 
            AND column_name = 'description'
            AND table_schema = 'public'
        """)
        
        existing_column = cur.fetchone()
        
        if existing_column:
            print("PASS: Description column already exists in departments table")
        else:
            print("Adding description column to departments table...")
            
            # Add description column
            cur.execute("ALTER TABLE public.departments ADD COLUMN description text;")
            
            # Update existing departments with sample descriptions
            cur.execute("""
                UPDATE public.departments SET description = 
                    CASE 
                        WHEN LOWER(name) LIKE '%ai%' OR LOWER(name) LIKE '%artificial%' THEN 'Artificial Intelligence and Machine Learning Department'
                        WHEN LOWER(name) LIKE '%marketing%' THEN 'Marketing and Business Development Department'
                        WHEN LOWER(name) LIKE '%hr%' OR LOWER(name) LIKE '%human%' THEN 'Human Resources Management Department'
                        WHEN LOWER(name) LIKE '%finance%' OR LOWER(name) LIKE '%accounting%' THEN 'Finance and Accounting Department'
                        WHEN LOWER(name) LIKE '%operations%' OR LOWER(name) LIKE '%ops%' THEN 'Operations and Logistics Department'
                        WHEN LOWER(name) LIKE '%tech%' OR LOWER(name) LIKE '%it%' OR LOWER(name) LIKE '%engineering%' THEN 'Technology and Engineering Department'
                        WHEN LOWER(name) LIKE '%construction%' OR LOWER(name) LIKE '%building%' THEN 'Construction and Infrastructure Department'
                        WHEN LOWER(name) LIKE '%ramstudios%' OR LOWER(name) LIKE '%studios%' THEN 'Creative Studios and Media Production Department'
                        ELSE name || ' Department'
                    END
                WHERE description IS NULL;
            """)
            
            conn.commit()
            print("PASS: Successfully added description column and updated existing departments")
        
        # Verify the migration
        cur.execute("SELECT COUNT(*) FROM public.departments WHERE description IS NOT NULL;")
        count = cur.fetchone()[0]
        print(f"PASS: Found {count} departments with descriptions")
        
        # Show sample department data
        cur.execute("""
            SELECT name, description 
            FROM public.departments 
            LIMIT 3
        """)
        
        departments = cur.fetchall()
        if departments:
            print("\nSample departments after migration:")
            for name, desc in departments:
                print(f"  - {name}: {desc}")
        
        cur.close()
        conn.close()
        
        print("\nPASS: Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Error applying migration: {e}")
        return False

if __name__ == "__main__":
    success = apply_migration()
    sys.exit(0 if success else 1)