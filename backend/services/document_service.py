"""
Document Service - Core document generation logic
Handles service letters, confirmation letters, and other official documents.
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import asyncio

from .template_service import TemplateService
from .pdf_service import PDFService
from database import execute_query, get_db_connection

logger = logging.getLogger(__name__)

class DocumentType:
    """Document type constants"""
    SERVICE_LETTER = "service_letter"
    CONFIRMATION_LETTER = "confirmation_letter"
    OFFER_LETTER = "offer_letter"
    REFERENCE_LETTER = "reference_letter"

class DocumentService:
    """Core document generation service"""
    
    def __init__(self):
        self.template_service = TemplateService()
        self.pdf_service = PDFService()
        self.storage_path = Path("storage/documents")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def generate_document(
        self,
        document_type: str,
        employee_id: int,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a document for an employee
        
        Args:
            document_type: Type of document to generate
            employee_id: ID of the employee
            additional_data: Additional data for document generation
            
        Returns:
            Dictionary containing document info and file path
        """
        try:
            logger.info(f"Generating {document_type} for employee {employee_id}")
            
            # Fetch employee data
            employee_data = await self._get_employee_data(employee_id)
            if not employee_data:
                raise ValueError(f"Employee with ID {employee_id} not found")
            
            # Generate document ID first
            document_id = str(uuid.uuid4())
            
            # Prepare template data including document ID for web preview
            template_data = await self._prepare_template_data(
                employee_data, document_type, additional_data or {}
            )
            template_data['document_id'] = document_id  # Add document ID for download button
            template_data['download_url'] = f"/documents/download/{document_id}"
            
            # Generate HTML from template
            html_content = await self.template_service.render_template(
                document_type, template_data
            )
            
            # Create user-friendly filename with employee name
            employee_name_safe = "".join(c for c in employee_data.get("name", "").replace(" ", "_") if c.isalnum() or c in "_-")
            document_type_clean = document_type.replace("_", " ").title().replace(" ", "_")
            
            # Internal filename (for storage)
            pdf_filename = f"{document_type}_{employee_id}_{document_id}.pdf"
            
            # User-friendly filename (for download)
            user_filename = f"{document_type_clean}_{employee_name_safe}.pdf"
            
            pdf_path = self.storage_path / pdf_filename
            
            await self.pdf_service.generate_pdf(html_content, str(pdf_path))
            
            # Save document record
            document_record = await self._save_document_record(
                employee_id, document_type, str(pdf_path), document_id
            )
            
            logger.info(f"Document generated successfully: {pdf_filename}")
            
            return {
                "success": True,
                "document_id": document_id,
                "file_path": str(pdf_path),
                "filename": user_filename,  # User-friendly filename for download
                "internal_filename": pdf_filename,  # Internal filename for storage
                "document_type": document_type,
                "employee_name": employee_data.get("name", ""),
                "generated_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "document_type": document_type,
                "employee_id": employee_id
            }
    
    async def _get_employee_data(self, employee_id: int) -> Optional[Dict[str, Any]]:
        """Fetch employee data from database"""
        try:
            query = """
            SELECT 
                e.id, e.name, e.email, e.role, e.phone_number, e.address,
                e.created_at, e.is_active,
                d.name as department_name,
                COALESCE(d.description, '') as department_description
            FROM employees e
            LEFT JOIN departments d ON e.department_id = d.id
            WHERE e.id = %s
            """
            
            result = execute_query(query, (employee_id,))
            
            if result and len(result) > 0:
                row = result[0]
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "email": row["email"],
                    "role": row["role"],
                    "phone_number": row["phone_number"] or "",
                    "address": row["address"] or "",
                    "created_at": row["created_at"],
                    "is_active": row["is_active"],
                    "department_name": row["department_name"] or "",
                    "department_description": row["department_description"] or "",
                    "joining_date": row["created_at"].strftime("%B %d, %Y") if row["created_at"] else "",
                    "current_date": datetime.now().strftime("%B %d, %Y"),
                    "formatted_role": row["role"].title().replace("_", " ") if row["role"] else "Employee"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching employee data: {str(e)}")
            return None
    
    async def _prepare_template_data(
        self,
        employee_data: Dict[str, Any],
        document_type: str,
        additional_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for template rendering"""
        
        # Base template data
        template_data = {
            **employee_data,
            **additional_data,
            "company_name": "Rise Tech Village",
            "company_address": "123 Tech Street, Innovation City, IC 12345",
            "company_phone": "+1 (555) 123-4567",
            "company_email": "hr@risetechvillage.com",
            "current_date": datetime.now().strftime("%B %d, %Y"),
            "document_type": document_type
        }
        
        # Document-specific data preparation
        if document_type == DocumentType.SERVICE_LETTER:
            template_data.update({
                "letter_title": "Service Letter",
                "subject": f"Service Letter for {employee_data.get('name', '')}",
                "purpose": additional_data.get("purpose", "To Whom It May Concern"),
                "service_duration": self._calculate_service_duration(employee_data.get("created_at"))
            })
            
        elif document_type == DocumentType.CONFIRMATION_LETTER:
            template_data.update({
                "letter_title": "Employment Confirmation Letter",
                "subject": f"Employment Confirmation for {employee_data.get('name', '')}",
                "confirmation_date": additional_data.get("confirmation_date", template_data["current_date"]),
                "probation_period": additional_data.get("probation_period", "3 months")
            })
            
        return template_data
    
    def _calculate_service_duration(self, joining_date) -> str:
        """Calculate service duration from joining date"""
        if not joining_date:
            return "N/A"
        
        try:
            if isinstance(joining_date, str):
                joining_date = datetime.fromisoformat(joining_date.replace('Z', '+00:00'))
            
            duration = datetime.now() - joining_date.replace(tzinfo=None)
            years = duration.days // 365
            months = (duration.days % 365) // 30
            
            if years > 0 and months > 0:
                return f"{years} year{'s' if years > 1 else ''} and {months} month{'s' if months > 1 else ''}"
            elif years > 0:
                return f"{years} year{'s' if years > 1 else ''}"
            elif months > 0:
                return f"{months} month{'s' if months > 1 else ''}"
            else:
                return f"{duration.days} day{'s' if duration.days > 1 else ''}"
        except:
            return "N/A"
    
    async def _save_document_record(
        self,
        employee_id: int,
        document_type: str,
        file_path: str,
        document_id: str
    ) -> Dict[str, Any]:
        """Save document generation record to database"""
        try:
            query = """
            INSERT INTO generated_documents 
            (id, employee_id, document_type, file_path, generated_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            generated_at = datetime.now(timezone.utc)
            expires_at = generated_at + timedelta(hours=24)
            
            execute_query(query, (
                document_id,
                employee_id,
                document_type,
                file_path,
                generated_at,
                expires_at
            ), fetch=False)
            
            return {
                "id": document_id,
                "employee_id": employee_id,
                "document_type": document_type,
                "generated_at": generated_at,
                "expires_at": expires_at
            }
            
        except Exception as e:
            logger.error(f"Error saving document record: {str(e)}")
            raise
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document information by ID"""
        try:
            query = """
            SELECT 
                gd.id, gd.employee_id, gd.document_type, gd.file_path,
                gd.generated_at, gd.expires_at, gd.downloaded_at,
                e.name as employee_name
            FROM generated_documents gd
            LEFT JOIN employees e ON gd.employee_id = e.id
            WHERE gd.id = %s AND gd.expires_at > %s
            """
            
            result = execute_query(query, (document_id, datetime.now(timezone.utc)))
            
            if result and len(result) > 0:
                row = result[0]
                
                # Create user-friendly filename
                employee_name_safe = "".join(c for c in (row["employee_name"] or "").replace(" ", "_") if c.isalnum() or c in "_-")
                document_type_clean = row["document_type"].replace("_", " ").title().replace(" ", "_")
                user_filename = f"{document_type_clean}_{employee_name_safe}.pdf"
                
                return {
                    "id": row["id"],
                    "employee_id": row["employee_id"],
                    "document_type": row["document_type"],
                    "file_path": row["file_path"],
                    "generated_at": row["generated_at"],
                    "expires_at": row["expires_at"],
                    "downloaded_at": row["downloaded_at"],
                    "employee_name": row["employee_name"],
                    "user_filename": user_filename,  # User-friendly filename for download
                    "is_expired": row["expires_at"] < datetime.now(timezone.utc) if row["expires_at"] else True,
                    "file_exists": os.path.exists(row["file_path"]) if row["file_path"] else False
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching document info: {str(e)}")
            return None
    
    async def mark_document_downloaded(self, document_id: str) -> bool:
        """Mark document as downloaded"""
        try:
            query = """
            UPDATE generated_documents 
            SET downloaded_at = %s 
            WHERE id = %s
            """
            
            execute_query(query, (datetime.now(timezone.utc), document_id), fetch=False)
            return True
            
        except Exception as e:
            logger.error(f"Error marking document as downloaded: {str(e)}")
            return False
    
    async def cleanup_expired_documents(self) -> int:
        """Clean up expired documents from storage"""
        try:
            # Find expired documents
            query = """
            SELECT id, file_path 
            FROM generated_documents 
            WHERE expires_at < %s
            """
            
            expired_docs = execute_query(query, (datetime.now(timezone.utc),))
            cleaned_count = 0
            
            for doc in expired_docs:
                doc_id, file_path = doc[0], doc[1]
                
                # Delete file if exists
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted expired document file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                
                # Delete database record
                delete_query = "DELETE FROM generated_documents WHERE id = %s"
                execute_query(delete_query, (doc_id,), fetch=False)
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired documents")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0

# Singleton instance
document_service = DocumentService()