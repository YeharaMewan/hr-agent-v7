"""
Services package for HR Management System
Contains document generation, template management, and PDF services.
"""

# Make services importable as a package
from .document_service import document_service, DocumentType
from .template_service import template_service  
from .pdf_service import pdf_service

__all__ = [
    'document_service',
    'template_service', 
    'pdf_service',
    'DocumentType'
]